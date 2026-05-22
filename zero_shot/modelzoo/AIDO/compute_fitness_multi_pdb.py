from __future__ import annotations

# Note: We set offline mode in the AIDOEngine.__init__ instead of here
# This allows trust_remote_code to work with cached files while still
# minimizing network requests during actual model loading
import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dms_utils import (
    AIDO_Structure_Tokenizer,
    build_cache_dir,
    build_output_path,
    compute_overlap_weights,
    iter_mutations,
    load_msa_a2m,
    preprocess_dataframe,
)
from utils import from_pdb_string


class AIDOEngine:
    """
    Lightweight AIDO inference engine with an API similar to esm/compute_fitness_multi_pdb.py.
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: Optional[str],
        *,
        codebook_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16_infer: bool = False,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16_infer = bool(use_fp16_infer and self.device.type == "cuda")

        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}. FP16 inference: {self.use_fp16_infer}")
        
        cache_home = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_name = f"models--{model_path.replace('/', '--')}"
        model_cache_path = cache_home / model_cache_name

        # Initialize with fallback to model_path in case cache doesn't exist
        resolved_model_path = model_path
        if model_cache_path.exists():
            snapshots_dir = model_cache_path / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    snapshot = snapshots[0]
                    print(f"Using snapshot: {snapshot.name}")
                    # Use the direct snapshot path to avoid network lookups
                    resolved_model_path = str(snapshot)

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                resolved_model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float16 if self.use_fp16_infer else torch.bfloat16,
            )
            .to(self.device)
            .eval()
        )

        if self.use_fp16_infer:
            self.model = self.model.half()

        self.seq_tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Structure tokenizer for encoding 3D protein structures
        structure_encoder_id = "genbio-ai/AIDO.StructureEncoder"
        structure_encoder_cache = cache_home / f"models--{structure_encoder_id.replace('/', '--')}"

        # Initialize with fallback to structure_encoder_id in case cache doesn't exist
        structure_encoder_path = structure_encoder_id
        if structure_encoder_cache.exists():
            structure_snapshots_dir = structure_encoder_cache / "snapshots"
            if structure_snapshots_dir.exists():
                structure_snapshots = list(structure_snapshots_dir.iterdir())
                if structure_snapshots:
                    structure_encoder_path = str(structure_snapshots[0])

        self.str_tokenizer = AIDO_Structure_Tokenizer(
            codebook_path=codebook_path,
            device=str(self.device),
            model_path=structure_encoder_path
        )

        self.cache_dir = cache_dir
        os.makedirs(os.path.join(cache_dir, "wt"), exist_ok=True)

        self.window_size: int = int(getattr(self.model.config, "max_position_embeddings", 1024))
        self.vocab_size: int = int(self.model.config.vocab_size)
        self.mask_token_id: int = int(self.seq_tokenizer.mask_token_id)

        w = compute_overlap_weights(self.window_size)
        self.window_weights = torch.tensor(w, device=self.device, dtype=torch.float32)

    def get_log_probs(
        self,
        q_seq: str,
        prot,
        msa: List[str],
        dms_df: pd.DataFrame,
        start: int,
        *,
        mask_str: bool = False,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Run sliding-window inference; returns (positions, logits_table) as torch Tensor.
        """
        all_poses, logit_table = get_logits_table_sliding(
            q_seq,
            prot,
            msa,
            dms_df,
            self.model,
            self.seq_tokenizer,
            self.str_tokenizer,
            start,
            sliding_window=768,
            sliding_step=768,
            mask_str=mask_str,
            verbose=False,
            disable_tqdm=True,
        )
        return all_poses, torch.from_numpy(logit_table)
    
    def _model_forward_log_probs(self, input_ids_1d: torch.Tensor) -> torch.Tensor:
        """
        Run the model on a 1D token id tensor and return log-probabilities.

        Args:
            input_ids_1d: Tensor of shape [T] on self.device.

        Returns:
            Log-probabilities tensor of shape [T, V] on self.device, float32.
        """
        with torch.inference_mode():
            if self.use_fp16_infer:
                # Keep output in float32 for stable log_softmax
                logits = self.model(input_ids_1d.unsqueeze(0)).logits[0].float()
            else:
                logits = self.model(input_ids_1d.unsqueeze(0)).logits[0]
            return torch.log_softmax(logits, dim=-1)

    def _compute_wt_overlapping(self, sequence: str) -> torch.Tensor:
        """
        WT mode with overlapping window aggregation for long sequences.

        Returns:
            GPU tensor [L, V] in float32.
        """
        inputs = self.seq_tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"][0].to(self.device)  # [T]
        T = int(input_ids.size(0))

        # Short sequence
        if T <= self.window_size:
            lprobs = self._model_forward_log_probs(input_ids)
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
            w_left = self.window_weights[: chunk_left.size(0)]
            lprobs_left = self._model_forward_log_probs(chunk_left)
            token_accum[start_left : end_left + 1] += lprobs_left * w_left.unsqueeze(-1)
            weight_accum[start_left : end_left + 1] += w_left

            # Right window
            chunk_right = input_ids[start_right : end_right + 1]
            w_right = self.window_weights[: chunk_right.size(0)]
            lprobs_right = self._model_forward_log_probs(chunk_right)
            token_accum[start_right : end_right + 1] += lprobs_right * w_right.unsqueeze(-1)
            weight_accum[start_right : end_right + 1] += w_right

            # Overlap check
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
            w_center = self.window_weights[: chunk_center.size(0)]
            lprobs_center = self._model_forward_log_probs(chunk_center)

            token_accum[start_center : end_center + 1] += lprobs_center * w_center.unsqueeze(-1)
            weight_accum[start_center : end_center + 1] += w_center

        # Normalize
        weight_accum = torch.clamp(weight_accum, min=1e-6)
        final_lprobs = token_accum / weight_accum.unsqueeze(-1)

        return final_lprobs[1:-1, :]  # remove CLS/EOS


def score_mutation_delta_logprob(
    global_mut_str: str,
    logits_table: torch.Tensor,
    all_poses: List[int],
    seq_tokenizer,
    *,
    start: int = 1,
    temp_mt: float = 1.0,
    temp_wt: float = 1.5,
) -> float:
    """
    Sum delta log-probabilities for a (possibly multi-site) mutation string.
    
    Matches the reference implementation in get_scores_from_table (utils/misc.py).
    Temperature parameters follow the reference: temp_mt=1.0, temp_wt=1.5
    
    Args:
        global_mut_str: Colon-separated mutation string, e.g., "H91Y:Y92F"
        logits_table: Tensor of shape [num_positions, vocab_size] with logits (not log-probs)
        all_poses: List of positions corresponding to rows in logits_table
        seq_tokenizer: Tokenizer to convert amino acid tokens to IDs
        start: Position offset (1-based by default)
        temp_mt: Temperature scaling for mutant probabilities (default 1.0)
        temp_wt: Temperature scaling for wildtype probabilities (default 1.5)
    
    Returns:
        Delta log probability score (float)
    """
    if not global_mut_str:
        return 0.0

    # Ensure logits_table is on CPU and float32 for consistent computation
    if isinstance(logits_table, np.ndarray):
        logits_table = torch.from_numpy(logits_table).float()
    elif logits_table.is_cuda:
        logits_table = logits_table.cpu().float()
    else:
        logits_table = logits_table.float()

    # Apply temperature scaling and convert to log probabilities
    # This matches: np.log(softmax(logits_table / temp_wt, axis=-1))
    logp_mt = F.log_softmax(logits_table / temp_mt, dim=-1)
    logp_wt = F.log_softmax(logits_table / temp_wt, dim=-1)
    
    total = 0.0
    vocab = seq_tokenizer.get_vocab()
    
    # Convert all_poses to list for consistent indexing
    all_poses_list = list(all_poses) if not isinstance(all_poses, list) else all_poses
    
    for wt_aa, pos_1based, mt_aa in iter_mutations(str(global_mut_str)):
        idx0 = pos_1based - start  # Convert to 0-based index
        
        # Check if position is in range
        if idx0 not in all_poses_list:
            continue
        
        # Get the index in the logits table
        pos_idx = all_poses_list.index(idx0)
        
        # Get token IDs and validate
        if wt_aa not in vocab or mt_aa not in vocab:
            continue
        
        wt_id = vocab[wt_aa]
        mt_id = vocab[mt_aa]
        
        # Validate indices are within bounds
        if wt_id < 0 or wt_id >= logp_wt.size(-1) or mt_id < 0 or mt_id >= logp_mt.size(-1):
            continue
        
        # Accumulate delta log probability: log P(mutant) - log P(wildtype)
        total += (logp_mt[pos_idx, mt_id] - logp_wt[pos_idx, wt_id]).item()

    return float(total)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="genbio-ai/AIDO.Protein-16B",
        help="HuggingFace repo id or local path to a AIDO checkpoint directory",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input DMS CSV (expects accompanying query.fasta, msa_data/, struc_data/ in the same folder)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write scored CSV",
    )
    parser.add_argument(
        "--mode",
        default="wt",
        choices=["wt"],
        help="Scoring mode (AIDO supports wt only)",
    )
    parser.add_argument(
        "--focus",
        type=int,
        default=1,
        help="Unused (for API compatibility)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="Mutation position offset, usually 1",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF cache directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--fp16-infer",
        action="store_true",
        help="Use float16 (default is bfloat16)",
    )
    parser.add_argument(
        "--structure-folder",
        default="data/zero_shot/BindingGYM/structures",
        help="Location of structures",
    )
    parser.add_argument(
        "--codebook-path",
        default=os.environ.get("AIDO_CODEBOOK_PATH"),
        help="Path to the external AIDO structure-tokenizer codebook.pt",
    )
    parser.add_argument(
        "--mask-str",
        action="store_true",
        help="Mask the structure input",
    )
    args = parser.parse_args()

    cache_dir = build_cache_dir(
        args.cache_dir,
        args.model_path,
        fp16=args.fp16_infer,
        focus=(args.focus == 1)
    )

    output_path = build_output_path(
        args.output_dir,
        args.input_csv,
        args.model_path,
        mode=args.mode,
        fp16=args.fp16_infer,
        focus=(args.focus == 1)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output file: {output_path}")

    engine = AIDOEngine(
        args.model_path,
        cache_dir,
        codebook_path=args.codebook_path,
        device=args.device,
        use_fp16_infer=args.fp16_infer,
    )

    sep = "\t" if args.input_csv.endswith(".tsv") else ","
    print(f"Reading input file: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=sep)

    final_scores: List[Optional[float]] = [None] * len(df)

    processor = preprocess_dataframe(
        df,
        wt_col="wildtype_sequence",
        mutant_col="mutant",
        chain_id_col="chain_id",
        poi_col="POI",
        focus=(args.focus == 1),
    )

    print(f"Starting scoring. Mode: {args.mode}")

    # Extract input paths
    input_root = Path(args.input_csv).parent
    dms_id = Path(args.input_csv).stem.replace(".csv", "").replace(".tsv", "")

    for group in tqdm(processor, desc="Scoring POI groups"):
        wt_seq = group.wt_concat

        if not wt_seq:
            for idx in group.row_indices:
                final_scores[idx] = 0.0
            continue

        # Get the subset of df for this group
        poi_df = df.loc[group.row_indices]

        # Use the concatenated wildtype sequence from preprocessing
        q_seq = wt_seq
        start = args.offset

        # Try to load query sequence from fasta if available (for validation)
        fasta_path = input_root / "query.fasta"
        if fasta_path.exists():
            dms2seq, dms2annot = load_fasta(fasta_path, load_annotation=True)
            if dms_id in dms2seq:
                fasta_seq = dms2seq[dms_id]
                if len(fasta_seq) == len(q_seq):
                    q_seq = fasta_seq
                start_end = dms2annot.get(dms_id, "1-0")
                if "-" in start_end:
                    start, _ = [int(x) for x in start_end.split("-")]

        # Load PDB structure first
        if "pdb_file" in poi_df.columns and not pd.isna(poi_df["pdb_file"].iloc[0]):
            pdb_file = poi_df["pdb_file"].iloc[0]
        else:
            print(f"Warning: Cannot determine PDB file (no pdb_file), skipping group")
            for idx in group.row_indices:
                final_scores[idx] = 0.0
            continue

        pdb_path = Path(args.structure_folder) / pdb_file

        if not pdb_path.exists():
            print(f"Warning: PDB file not found at {pdb_path}, skipping group")
            for idx in group.row_indices:
                final_scores[idx] = 0.0
            continue

        with open(pdb_path) as IN:
            text = IN.read()

        prot = from_pdb_string(text, molecular_type="protein", insertion_code_process="insert")

        # Load MSA (only .a2m supported). Use POI (e.g. 1LP1) to build filename,
        # try several candidate directories and use the first match.
        if "POI" in poi_df.columns and not pd.isna(poi_df["POI"].iloc[0]):
            poi_val = str(poi_df["POI"].iloc[0])
        else:
            poi_val = dms_id
        msa_basename = f"{poi_val}.a2m"
        candidates = [
            Path(input_root).parent / "msas" / msa_basename,
            Path(input_root) / "msas" / msa_basename,
            Path(input_root) / "msa_data" / msa_basename,
            Path(input_root).parent / "msa_data" / msa_basename,
        ]
        msa_path = None
        for c in candidates:
            if c.exists():
                msa_path = c
                break

        if msa_path is not None:
            print(f"Using MSA: {msa_path} (derived from POI='{poi_val}')")
            msa = load_msa_a2m(str(msa_path))
            if len(q_seq) != len(msa[0]):
                print(f"Warning: MSA length mismatch ({len(msa[0])} vs {len(q_seq)}), using query only")
                msa = [q_seq]
        else:
            msa = [q_seq]
        prot_seq = prot.seq(True)

        # Validate or adjust sequence matching
        if prot_seq != q_seq:
            if len(prot_seq) == len(q_seq):
                q_seq = prot_seq
            else:
                min_len = min(len(prot_seq), len(q_seq))
                q_seq = q_seq[:min_len]
                prot = prot.slice(0, min_len, by="index")
                msa = [seq[:min_len] for seq in msa]

        # Ensure MSA first sequence matches q_seq (it should by now, but double-check)
        if len(msa) > 0 and msa[0] != q_seq:
            msa[0] = q_seq

        # Create a temporary dataframe for this group with global mutations
        group_df = pd.DataFrame({
            "mutant": group.mutant_global,
            "DMS_score": [df.loc[idx, "DMS_score"] if "DMS_score" in df.columns else 0.0
                         for idx in group.row_indices]
        })

        # Show up to 10 mismatch examples from this group's mutations
        mismatch_examples = []
        for mut in group_df["mutant"].tolist():
            for sub_mutant in str(mut).split(":"):
                s = sub_mutant.strip()
                if len(s) < 3:
                    continue
                try:
                    wt = s[0]
                    idx = int(s[1:-1]) - start
                    actual = q_seq[idx] if 0 <= idx < len(q_seq) else "OUT_OF_RANGE"
                except Exception as e:
                    actual = f"ERR:{e}"
                if actual != wt:
                    mismatch_examples.append((s, idx, wt, actual))
                if len(mismatch_examples) >= 10:
                    break
            if len(mismatch_examples) >= 10:
                break
        if mismatch_examples:
            print("DEBUG: Found mismatch examples (mut, idx, wt, actual):")
            for ex in mismatch_examples:
                print("  ", ex)
        else:
            print("DEBUG: No WT mismatches found in group_df mutations (based on q_seq).")


        # Inference
        all_poses, logits_table = engine.get_log_probs(
            q_seq,
            prot,
            msa,
            group_df,
            start,
            mask_str=args.mask_str,
        )

        # Score mutations
        for idx, (row_idx, mut) in enumerate(zip(group.row_indices, group.mutant_global)):
            score = score_mutation_delta_logprob(
                mut,
                logits_table,
                all_poses,
                engine.seq_tokenizer,
                start=start,
                temp_mt=1.0,
                temp_wt=1.5,
            )
            final_scores[row_idx] = score

    # Add scores to dataframe
    df["AIDO_score"] = final_scores

    # Save results
    df.to_csv(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
