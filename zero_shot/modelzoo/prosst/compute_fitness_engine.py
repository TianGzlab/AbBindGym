from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from prosst.structure.quantizer import PdbQuantizer

from dms_utils import (
    build_cache_dir,
    build_output_path,
    compute_overlap_weights,
    iter_mutations,
    preprocess_dataframe,
    sanitize_sequence,
    sha256_upper,
)

# Supported structure file extensions in priority order
STRUCTURE_EXTENSIONS: Tuple[str, ...] = (".pdb", ".cif", ".mmcif")


def resolve_structure_file(
    pdb_dir: Path,
    identifier: str,
    extensions: Tuple[str, ...] = STRUCTURE_EXTENSIONS,
) -> Optional[Path]:
    """Resolve a structure file path from a PDB directory using an identifier.

    Searches for files matching `{identifier}{ext}` in pdb_dir, trying extensions
    in priority order (.pdb first, then .cif, .mmcif).

    Args:
        pdb_dir: Root directory containing structure files.
        identifier: Unique identifier (e.g., pdb_id like "1ABC") from the CSV.
        extensions: Tuple of file extensions to try, in priority order.

    Returns:
        Path to the structure file if found, None otherwise.
    """
    for ext in extensions:
        candidate = pdb_dir / f"{identifier}{ext}"
        if candidate.exists():
            return candidate
    return None


def infer_window_size_from_model_path(model_path: str) -> int:
    """Infer window size from model path (e.g., ProSST-1024 -> 1024)."""
    match = re.search(r"(\d+)$", model_path.rstrip("/"))
    if match:
        return int(match.group(1))
    return 1024  # default


class ProSSTEngine:
    """ProSST inference engine for DMS scoring.

    ProSST is a structure-aware protein language model that requires both
    residue sequences and structure sequences (from PdbQuantizer) as input.

    It supports two inference modes:
        - wt: one-pass inference per window and return per-position log-probabilities
              using an overlapping window aggregation for long sequences.
        - masked: per-position masked inference with the "optimal window" rule.

    Cached artifacts are stored under:
        {cache_dir}/wt/{sha256(sequence)}.pt
        {cache_dir}/masked/{sha256(sequence)}.pt
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        *,
        device: Optional[str] = None,
        use_fp16_infer: bool = False,
    ) -> None:
        """Initialize the ProSST inference engine.

        Args:
            model_path: Local path to the ProSST model.
            cache_dir: Directory for caching log-probabilities.
            device: Device to run inference on (e.g., "cuda:0", "cpu").
            use_fp16_infer: Whether to use FP16 inference on GPU.
        """
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16_infer = bool(use_fp16_infer and self.device.type == "cuda")

        print(f"Loading ProSST model from: {model_path}")
        print(f"Device: {self.device}. FP16 inference: {self.use_fp16_infer}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.model = (
            AutoModelForMaskedLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        if self.use_fp16_infer:
            self.model = self.model.half()

        self.cache_dir = cache_dir
        os.makedirs(os.path.join(cache_dir, "wt"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "masked"), exist_ok=True)

        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size: int = int(self.tokenizer.vocab_size)
        self.mask_token_id: int = int(self.tokenizer.mask_token_id)

        # ProSST window size: try config, then infer from model path
        config_window = getattr(self.model.config, "max_position_embeddings", 0)
        if config_window and config_window > 0:
            self.window_size = int(config_window)
        else:
            self.window_size = infer_window_size_from_model_path(model_path)

        print(f"Window size: {self.window_size}")

        w = compute_overlap_weights(self.window_size)
        self.window_weights = torch.tensor(w, device=self.device, dtype=torch.float32)

    def _tokenize_structure_sequence(self, structure_sequence: List[int]) -> torch.Tensor:
        """Tokenize a structure sequence for ProSST input.

        Args:
            structure_sequence: List of integer structure tokens.

        Returns:
            Tensor of shape [T] with shifted structure token IDs.
        """
        # Shift structure tokens by 3 and add special tokens
        shift_structure_sequence = [i + 3 for i in structure_sequence]
        shift_structure_sequence = [1, *shift_structure_sequence, 2]
        return torch.tensor(shift_structure_sequence, dtype=torch.long, device=self.device)

    def get_log_probs(
        self,
        residue_sequence: str,
        structure_sequence: List[int],
        mode: str,
    ) -> torch.Tensor:
        """Get (and cache) per-residue log-probabilities.

        Args:
            residue_sequence: Wildtype sequence string (no special tokens).
            structure_sequence: List of integer structure tokens.
            mode: "wt" or "masked".

        Returns:
            CPU tensor of shape [L, V] in float16 for storage efficiency.
            Special tokens (CLS/EOS) have been removed.
        """
        seq = sanitize_sequence(residue_sequence)
        # Create a combined hash for caching (both sequence and structure)
        combined_key = f"{seq}|{','.join(map(str, structure_sequence))}"
        seq_hash = sha256_upper(combined_key)

        cache_path = os.path.join(self.cache_dir, mode, f"{seq_hash}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        if mode == "wt":
            log_probs = self._compute_wt_overlapping(seq, structure_sequence)
        elif mode == "masked":
            log_probs = self._compute_masked_optimal_batched(seq, structure_sequence)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Cache as float16 on CPU
        log_probs_cpu = log_probs.detach().to("cpu").half()
        torch.save(log_probs_cpu, cache_path)
        return log_probs_cpu

    def _model_forward_log_probs(
        self,
        input_ids_1d: torch.Tensor,
        ss_input_ids_1d: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model on a 1D token id tensor and return log-probabilities.

        Args:
            input_ids_1d: Tensor of shape [T] on self.device.
            ss_input_ids_1d: Tensor of shape [T] on self.device (structure sequence).

        Returns:
            Log-probabilities tensor of shape [T, V] on self.device, float32.
        """
        with torch.inference_mode():
            if self.use_fp16_infer:
                logits = (
                    self.model(
                        input_ids=input_ids_1d.unsqueeze(0),
                        ss_input_ids=ss_input_ids_1d.unsqueeze(0),
                    )
                    .logits[0]
                    .float()
                )
            else:
                logits = self.model(
                    input_ids=input_ids_1d.unsqueeze(0),
                    ss_input_ids=ss_input_ids_1d.unsqueeze(0),
                ).logits[0]
            return torch.log_softmax(logits, dim=-1)

    def _compute_wt_overlapping(
        self,
        sequence: str,
        structure_sequence: List[int],
    ) -> torch.Tensor:
        """WT mode with overlapping window aggregation for long sequences.

        Returns:
            GPU tensor [L, V] in float32.
        """
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"][0].to(self.device)  # [T]
        ss_input_ids = self._tokenize_structure_sequence(structure_sequence)  # [T]
        T = int(input_ids.size(0))

        # Short sequence
        if T <= self.window_size:
            lprobs = self._model_forward_log_probs(input_ids, ss_input_ids)
            return lprobs[1:-1, :]  # remove CLS/EOS

        # Long sequence overlapping aggregation
        token_accum = torch.zeros((T, self.vocab_size), device=self.device, dtype=torch.float32)
        weight_accum = torch.zeros((T,), device=self.device, dtype=torch.float32)

        stride = max(1, (self.window_size // 2) - 1)
        start_left = 0
        end_left = self.window_size - 1

        start_right = (T - 1) - self.window_size + 1
        end_right = T - 1

        while True:
            # Left window
            chunk_left = input_ids[start_left : end_left + 1]
            ss_chunk_left = ss_input_ids[start_left : end_left + 1]
            w_left = self.window_weights[: chunk_left.size(0)]
            lprobs_left = self._model_forward_log_probs(chunk_left, ss_chunk_left)
            token_accum[start_left : end_left + 1] += lprobs_left * w_left.unsqueeze(-1)
            weight_accum[start_left : end_left + 1] += w_left

            # Right window
            chunk_right = input_ids[start_right : end_right + 1]
            ss_chunk_right = ss_input_ids[start_right : end_right + 1]
            w_right = self.window_weights[: chunk_right.size(0)]
            lprobs_right = self._model_forward_log_probs(chunk_right, ss_chunk_right)
            token_accum[start_right : end_right + 1] += lprobs_right * w_right.unsqueeze(-1)
            weight_accum[start_right : end_right + 1] += w_right

            if end_left > start_right:
                break

            start_left += stride
            end_left += stride
            start_right -= stride
            end_right -= stride

        # Center patch if overlap is not wide enough
        final_overlap = end_left - start_right + 1
        if final_overlap < stride:
            start_center = max(0, (T // 2) - (self.window_size // 2))
            end_center = min(T - 1, start_center + self.window_size - 1)
            chunk_center = input_ids[start_center : end_center + 1]
            ss_chunk_center = ss_input_ids[start_center : end_center + 1]
            w_center = self.window_weights[: chunk_center.size(0)]
            lprobs_center = self._model_forward_log_probs(chunk_center, ss_chunk_center)

            token_accum[start_center : end_center + 1] += lprobs_center * w_center.unsqueeze(-1)
            weight_accum[start_center : end_center + 1] += w_center

        weight_accum = torch.clamp(weight_accum, min=1e-6)
        final_lprobs = token_accum / weight_accum.unsqueeze(-1)

        return final_lprobs[1:-1, :]  # remove CLS/EOS

    def _compute_masked_optimal_batched(
        self,
        sequence: str,
        structure_sequence: List[int],
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Masked-marginals with optimal window slicing, batched over positions.

        Returns:
            GPU tensor [L, V] in float32. L excludes special tokens.
        """
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids_full = inputs["input_ids"][0].to(self.device)  # [T]
        ss_input_ids_full = self._tokenize_structure_sequence(structure_sequence)  # [T]
        T = int(input_ids_full.size(0))
        L = len(sequence)
        V = self.vocab_size
        W = self.window_size
        half = W // 2

        final_lprobs = torch.empty((L, V), device=self.device, dtype=torch.float32)

        # residue token indices in [1..L], because 0 is CLS and L+1 is EOS
        token_idx_all = torch.arange(1, L + 1, device=self.device, dtype=torch.long)

        # Pre-create arange for window offsets
        win_offsets = torch.arange(W, device=self.device, dtype=torch.long)

        for b0 in range(0, L, batch_size):
            tok = token_idx_all[b0 : b0 + batch_size]  # [B]
            B = int(tok.numel())

            if T <= W:
                # Full-sequence batching
                batch_ids = input_ids_full.unsqueeze(0).expand(B, T).clone()  # [B, T]
                batch_ss_ids = ss_input_ids_full.unsqueeze(0).expand(B, T).clone()  # [B, T]
                batch_ids[torch.arange(B, device=self.device), tok] = self.mask_token_id

                with torch.inference_mode():
                    logits = self.model(
                        input_ids=batch_ids,
                        ss_input_ids=batch_ss_ids,
                    ).logits  # [B, T, V]
                    logits_at_mask = logits[torch.arange(B, device=self.device), tok, :]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)  # [B, V]

                final_lprobs[b0 : b0 + B] = lprobs
            else:
                # Optimal-window batching
                starts = torch.clamp(tok - half, min=0, max=T - W)  # [B]
                idx = starts.unsqueeze(1) + win_offsets.unsqueeze(0)  # [B, W]
                batch_ids = input_ids_full[idx].clone()  # [B, W]
                batch_ss_ids = ss_input_ids_full[idx].clone()  # [B, W]
                mask_pos = tok - starts  # [B] in [0..W-1]
                batch_ids[torch.arange(B, device=self.device), mask_pos] = self.mask_token_id

                with torch.inference_mode():
                    logits = self.model(
                        input_ids=batch_ids,
                        ss_input_ids=batch_ss_ids,
                    ).logits  # [B, W, V]
                    logits_at_mask = logits[
                        torch.arange(B, device=self.device), mask_pos, :
                    ]  # [B, V]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)  # [B, V]

                final_lprobs[b0 : b0 + B] = lprobs

        return final_lprobs


class StructureSequenceManager:
    """Manages structure sequence generation from PDB files.

    This class handles:
    - Loading PdbQuantizer for structure sequence generation
    - Caching generated structure sequences
    - Resolving structure files from --pdb-dir using pdb_id
    """

    def __init__(
        self,
        pdb_dir: str,
        structure_cache_dir: str,
        structure_vocab_size: int = 1024,
        device: Optional[str] = None,
    ) -> None:
        self.pdb_dir = Path(pdb_dir)
        self.cache_dir = Path(structure_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.structure_vocab_size = structure_vocab_size
        self.device = device
        

        # Lazy load PdbQuantizer
        self._quantizer: Optional[PdbQuantizer] = None

    def _get_quantizer(self) -> PdbQuantizer:
        """Lazily initialize the PdbQuantizer."""
        if self._quantizer is None:
            try:
                print(f"Initializing PdbQuantizer with vocab_size={self.structure_vocab_size}")
                self._quantizer = PdbQuantizer(
                    structure_vocab_size=self.structure_vocab_size,
                    subgraph_interval=1,
                    device=self.device,
                )
            except ImportError as e:
                raise ImportError(
                    "Failed to import PdbQuantizer. "
                    "Please ensure prosst package is installed with structure support. "
                    f"Error: {e}"
                ) from e
        return self._quantizer

    def resolve_and_get_structure_sequence(
        self,
        pdb_id: str,
        chains: Optional[List[str]] = None,
    ) -> Tuple[Optional[List[int]], Optional[str]]:
        """Resolve a PDB file from pdb_id and get its structure sequence.

        Searches for {pdb_id}.pdb, {pdb_id}.cif, etc. in the pdb_dir.
        If found, generates and caches the structure sequence.

        Args:
            pdb_id: Unique identifier used to locate the structure file (e.g., "1ABC").
            chains: Optional list of chain IDs to include (in order). If None, use all chains.

        Returns:
            Tuple of:
                - Structure sequence (list of ints), or None if file not found
                - Extracted residue sequence string, or None if file not found
        """
        if not pdb_id or pd.isna(pdb_id):
            return None, None

        pdb_id_str = str(pdb_id).strip()
        if not pdb_id_str:
            return None, None

        # Resolve the structure file path
        pdb_path = resolve_structure_file(self.pdb_dir, pdb_id_str)
        if pdb_path is None:
            return None, None

        # Build cache key including chains to differentiate same PDB with different chain subsets
        chains_key = ",".join(chains) if chains else "_all_"
        cache_key = sha256_upper(f"{pdb_path}|{self.structure_vocab_size}|{chains_key}")
        cache_file = self.cache_dir / f"{cache_key}.pt"

        if cache_file.exists():
            cached = torch.load(cache_file)
            return cached["struct_seq"], cached["residue_seq"]

        # Generate structure sequence
        try:
            quantizer = self._get_quantizer()
            residue_seq, struct_seq = quantizer(
                str(pdb_path), return_residue_seq=True, chains=chains
            )
        except Exception as e:
            print(f"Warning: Failed to generate structure for {pdb_path}: {e}")
            return None, None

        # Cache
        torch.save(
            {"struct_seq": struct_seq, "residue_seq": residue_seq},
            cache_file,
        )

        return struct_seq, residue_seq

    def get_structure_sequence(
        self,
        pdb_file: str,
        chain_order: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List[int]], str]:
        """Get structure sequence for a PDB file (legacy interface).

        Args:
            pdb_file: PDB filename (relative to pdb_dir) or absolute path.
            chain_order: Optional list of chain IDs to extract.

        Returns:
            Tuple of:
                - Dict mapping chain_id to structure sequence (list of ints)
                - Extracted residue sequence string
        """
        # Resolve PDB path
        if os.path.isabs(pdb_file):
            pdb_path = Path(pdb_file)
        else:
            pdb_path = self.pdb_dir / pdb_file

        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        # Check cache
        cache_key = sha256_upper(f"{pdb_path}|{self.structure_vocab_size}")
        cache_file = self.cache_dir / f"{cache_key}.pt"

        if cache_file.exists():
            cached = torch.load(cache_file)
            return cached["struct_seq"], cached["residue_seq"]

        # Generate structure sequence
        quantizer = self._get_quantizer()
        residue_seq, struct_seq = quantizer(str(pdb_path), return_residue_seq=True)

        # For single-chain or simple case, return as single entry
        result_struct = {"_all": struct_seq}
        result_residue = residue_seq

        # Cache
        torch.save(
            {"struct_seq": result_struct, "residue_seq": result_residue},
            cache_file,
        )

        return result_struct, result_residue


def score_mutation_delta_logprob(
    global_mut_str: str,
    *,
    sequence: str,
    log_probs_cpu: torch.Tensor,
    vocab: dict,
    offset_idx: int = 1,
    strict_wt_check: bool = True,
) -> float:
    """Score a (possibly multi-site) mutation string by sum of delta log-probabilities.

    Score definition:
        sum_i [ log P(mut_i | context) - log P(wt_i | context) ]

    Args:
        global_mut_str: Mutation string in global coordinates, e.g. "H91Y:K120A".
        sequence: Wild-type sequence string.
        log_probs_cpu: CPU tensor [L, V]. Values may be float16.
        vocab: Tokenizer vocabulary mapping AA chars to token IDs.
        offset_idx: 1-based offset used in the dataset. Typically 1.
        strict_wt_check: If True, raise an error if WT mismatch is detected.

    Returns:
        A Python float score.
    """
    if not global_mut_str or str(global_mut_str).strip() in {"WT", ""}:
        return 0.0

    L = int(log_probs_cpu.size(0))
    total = 0.0

    lprobs = log_probs_cpu.float()

    for wt, pos, mt in iter_mutations(str(global_mut_str)):
        idx0 = pos - offset_idx
        if idx0 < 0 or idx0 >= L:
            continue

        if strict_wt_check:
            if sequence[idx0].upper() != wt.upper():
                raise AssertionError(
                    f"WT mismatch at position {pos}. Expected {wt}, found {sequence[idx0]}"
                )

        wt_id = vocab.get(wt)
        mt_id = vocab.get(mt)
        if wt_id is None or mt_id is None:
            continue

        total += (lprobs[idx0, mt_id] - lprobs[idx0, wt_id]).item()

    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProSST fitness scoring for DMS datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to a ProSST model (e.g., AI4Protein/ProSST-1024)",
    )

    # Input/Output
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV/TSV file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for result CSVs",
    )

    # PDB directory for structure file resolution
    parser.add_argument(
        "--pdb-dir",
        required=True,
        help=(
            "Root directory containing structure files (PDB/mmCIF). "
            "Files are located as {pdb_id}.pdb or {pdb_id}.cif based on the pdb_id column."
        ),
    )

    parser.add_argument(
        "--pdb-id-col",
        default="pdb_id",
        help="Column name in CSV containing the PDB identifier for structure lookup (default: pdb_id)",
    )

    parser.add_argument(
        "--structure-vocab-size",
        type=int,
        default=1024,
        choices=[20, 64, 128, 512, 1024, 2048, 4096],
        help="Structure vocabulary size (must match model, e.g., 1024 for ProSST-1024)",
    )

    parser.add_argument(
        "--structure-cache-dir",
        default="./prosst_structure_cache",
        help="Cache directory for generated structure sequences",
    )

    # Inference mode
    parser.add_argument(
        "--mode",
        default="wt",
        choices=["wt", "masked"],
        help="Scoring mode: 'wt' for wildtype context, 'masked' for masked marginals",
    )

    # Data preprocessing
    parser.add_argument(
        "--focus",
        type=int,
        default=1,
        help="1=drop silent chains, 0=keep all chains",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="Mutation position offset, usually 1",
    )

    # Caching and device
    parser.add_argument(
        "--cache-dir",
        default="./prosst_logits_cache",
        help="Cache directory for log-probabilities",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g., cuda:0 or cpu",
    )
    parser.add_argument(
        "--fp16-infer",
        action="store_true",
        help="Enable FP16 inference on GPU",
    )

    # Sequence source
    parser.add_argument(
        "--use-pdb-sequence",
        action="store_true",
        help=(
            "Use the residue sequence extracted from PDB file instead of CSV's wildtype_sequence. "
            "This ensures sequence/structure length consistency but may affect mutation position mapping."
        ),
    )

    # Validation
    parser.add_argument(
        "--no-strict-wt-check",
        dest="strict_wt_check",
        action="store_false",
        help="Disable WT mismatch checks",
    )
    parser.set_defaults(strict_wt_check=True)

    args = parser.parse_args()

    # Validate pdb_dir exists
    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.is_dir():
        raise ValueError(f"--pdb-dir must be a valid directory: {args.pdb_dir}")

    # Setup directories
    cache_dir = build_cache_dir(
        args.cache_dir,
        args.model_path,
        fp16=args.fp16_infer,
        focus=(args.focus == 1),
    )
    print(f"Cache dir: {cache_dir}")

    output_path = build_output_path(
        args.output_dir,
        args.input_csv,
        args.model_path,
        mode=args.mode,
        fp16=args.fp16_infer,
        focus=(args.focus == 1),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output file: {output_path}")

    # Initialize engine
    engine = ProSSTEngine(
        args.model_path,
        cache_dir,
        device=args.device,
        use_fp16_infer=args.fp16_infer,
    )

    # Read input
    sep = "\t" if args.input_csv.endswith(".tsv") else ","
    print(f"Reading input file: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=sep)

    # Validate pdb_id column exists
    pdb_id_col = args.pdb_id_col
    if pdb_id_col not in df.columns:
        raise ValueError(
            f"Column '{pdb_id_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}. "
            f"Use --pdb-id-col to specify the correct column name."
        )

    # Initialize structure manager
    struct_manager = StructureSequenceManager(
        pdb_dir=args.pdb_dir,
        structure_cache_dir=args.structure_cache_dir,
        structure_vocab_size=args.structure_vocab_size,
        device=args.device,
    )

    final_scores: List[Optional[float]] = [None] * len(df)
    skipped_count = 0

    processor = preprocess_dataframe(
        df,
        wt_col="wildtype_sequence",
        mutant_col="mutant",
        chain_id_col="chain_id",
        poi_col="POI",
        focus=(args.focus == 1),
    )

    print(f"Starting scoring. Mode: {args.mode}")
    for group in tqdm(processor, desc="Scoring POI groups"):
        wt_seq = group.wt_concat

        if not wt_seq:
            for row_idx in group.row_indices:
                final_scores[row_idx] = 0.0
            continue

        # Get pdb_id from the first row of the group to resolve structure file
        first_row_idx = group.row_indices[0]
        pdb_id = df.loc[first_row_idx, pdb_id_col]

        # Resolve structure file and generate structure sequence with chain filtering
        # group.chains contains the ordered list of chain IDs used to construct wt_concat
        chains_to_use = group.chains if group.chains else None
        struct_seq, pdb_residue_seq = struct_manager.resolve_and_get_structure_sequence(
            pdb_id, chains=chains_to_use
        )

        if struct_seq is None:
            # Structure file not found or failed to process
            pdb_id_str = str(pdb_id) if pdb_id else "(empty)"
            print(
                f"Warning: Structure file not found for pdb_id='{pdb_id_str}' in {args.pdb_dir}. "
                f"Skipping {len(group.row_indices)} row(s)."
            )
            skipped_count += len(group.row_indices)
            for row_idx in group.row_indices:
                final_scores[row_idx] = None
            continue

        # Determine which sequence to use for scoring
        if args.use_pdb_sequence:
            # Use PDB's residue sequence (ensures length matches structure)
            scoring_seq = pdb_residue_seq if pdb_residue_seq else wt_seq
            if pdb_residue_seq and len(pdb_residue_seq) != len(wt_seq):
                print(
                    f"Info: Using PDB sequence (len={len(pdb_residue_seq)}) instead of "
                    f"CSV sequence (len={len(wt_seq)}) for pdb_id='{pdb_id}'."
                )
        else:
            scoring_seq = wt_seq

        # Validate sequence/structure length consistency
        if len(scoring_seq) != len(struct_seq):
            pdb_id_str = str(pdb_id) if pdb_id else "(unknown)"
            print(
                f"\nError: Sequence/structure length mismatch for pdb_id='{pdb_id_str}':\n"
                f"  - Residue sequence length: {len(scoring_seq)}\n"
                f"  - Structure sequence length: {len(struct_seq)}\n"
                f"  - PDB residue sequence length: {len(pdb_residue_seq) if pdb_residue_seq else 'N/A'}\n"
                f"\nThis typically happens when:\n"
                f"  1. CSV's wildtype_sequence contains only subset of chains (e.g., VH only)\n"
                f"  2. PDB file contains full complex (e.g., VH + VL + antigen)\n"
                f"\nSolutions:\n"
                f"  - Use --use-pdb-sequence to use the sequence from PDB file\n"
                f"  - Ensure PDB file contains only the chains matching CSV's sequence\n"
                f"\nSkipping {len(group.row_indices)} row(s)."
            )
            skipped_count += len(group.row_indices)
            for row_idx in group.row_indices:
                final_scores[row_idx] = None
            continue

        log_probs = engine.get_log_probs(scoring_seq, struct_seq, args.mode)

        for row_idx, mut_global in zip(group.row_indices, group.mutant_global):
            final_scores[row_idx] = score_mutation_delta_logprob(
                mut_global,
                sequence=scoring_seq,
                log_probs_cpu=log_probs,
                vocab=engine.vocab,
                offset_idx=args.offset,
                strict_wt_check=args.strict_wt_check,
            )

    df[f"prosst_{args.mode}_score"] = final_scores
    df.to_csv(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")
    if skipped_count > 0:
        print(f"Note: {skipped_count} row(s) were skipped due to missing structure files.")


if __name__ == "__main__":
    main()