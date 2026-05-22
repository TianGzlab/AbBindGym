from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from filelock import FileLock

from dms_utils import (
    build_cache_dir,
    build_output_path,
    compute_overlap_weights,
    iter_mutations,
    preprocess_dataframe,
    sanitize_sequence,
    sha256_upper,
)

# Maximum sequence length for AbLang2 (512 tokens - 2 special tokens)
ABLANG2_MAX_AA_LEN = 510

class AbLang2Engine:
    """
    High-performance AbLang2 inference engine for DMS scoring.

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
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16_infer = bool(use_fp16_infer and self.device.type == "cuda")

        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}. FP16 inference: {self.use_fp16_infer}")

        # Load base model and tokenizer
        # Note: Do NOT add model_path to sys.path before loading to avoid import conflicts
        base_model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Import and use AbLang2 adapter from local directory
        self.ablang = None
        try:
            # Import adapter from the same directory as this script
            from adapter import AbLang2PairedHuggingFaceAdapter

            self.ablang = AbLang2PairedHuggingFaceAdapter(
                model=base_model,
                tokenizer=self.tokenizer,
                device=str(self.device)
            )

            # Adapter stores model in self.AbLang, apply FP16 if needed
            if self.use_fp16_infer:
                self.ablang.AbLang = self.ablang.AbLang.half()

            # Use the adapter's model for compatibility
            self.model = self.ablang.AbLang
        except ImportError as e:
            print(f"Warning: Could not import adapter from local directory: {e}")
            print("Please keep adapter.py and its dependencies in zero_shot/modelzoo/ablang2/")
            self.model = base_model.to(self.device).eval()
            if self.use_fp16_infer:
                self.model = self.model.half()
            self.ablang = None
        except Exception as e:
            print(f"Warning: Error loading adapter: {e}")
            import traceback
            traceback.print_exc()
            self.model = base_model.to(self.device).eval()
            if self.use_fp16_infer:
                self.model = self.model.half()
            self.ablang = None

        self.cache_dir = cache_dir
        os.makedirs(os.path.join(cache_dir, "wt"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "masked"), exist_ok=True)

        self.window_size: int = int(getattr(self.model.config, "max_position_embeddings", 1024))
        self.vocab_size: int = int(self.model.config.vocab_size)
        self.mask_token_id: int = int(self.tokenizer.mask_token_id)

        w = compute_overlap_weights(self.window_size)
        self.window_weights = torch.tensor(w, device=self.device, dtype=torch.float32)

    def get_log_probs(self, sequence: str, mode: str) -> torch.Tensor:
        """
        Get (and cache) per-residue log-probabilities.

        Args:
            sequence: Wildtype sequence string (no special tokens).
            mode: "wt" or "masked".

        Returns:
            CPU tensor of shape [L, V] in float16 for storage efficiency.
            Special tokens (CLS/EOS) have been removed.
        """
        seq = sanitize_sequence(sequence)
        seq_hash = sha256_upper(seq)

        cache_path = os.path.join(self.cache_dir, mode, f"{seq_hash}.pt")
        lock_path = cache_path + ".lock" # avoid error when using DDP
        with FileLock(lock_path):
            if os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu")

        if mode == "wt":
            log_probs = self._compute_wt_overlapping(seq)
        elif mode == "masked":
            log_probs = self._compute_masked_optimal_batched(seq)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Cache as float16 on CPU
        log_probs_cpu = log_probs.detach().to("cpu").half()
        torch.save(log_probs_cpu, cache_path)
        return log_probs_cpu
    
    def _model_forward_log_probs(self, input_ids_1d: torch.Tensor) -> torch.Tensor:
        """
        Run the model on a 1D token id tensor and return log-probabilities.

        Args:
            input_ids_1d: Tensor of shape [T] on self.device.

        Returns:
            Log-probabilities tensor of shape [T, V] on self.device, float32.
        """
        with torch.inference_mode():
            # Call model directly with token IDs
            outputs = self.model(input_ids_1d.unsqueeze(0))

            # Extract logits from model output
            # For AbLang2 HuggingFace wrapper, logits are in last_hidden_state
            if torch.is_tensor(outputs):
                # Direct logits tensor [1, T, V]
                logits = outputs[0]
            elif hasattr(outputs, 'last_hidden_state'):
                # AbLang2 HuggingFace wrapper returns logits in last_hidden_state
                logits = outputs.last_hidden_state[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits[0]
            else:
                raise AttributeError(
                    f"Cannot extract logits from model output. Output type: {type(outputs)}"
                )

            if self.use_fp16_infer:
                logits = logits.float()

            return torch.log_softmax(logits, dim=-1)


    def _compute_wt_overlapping(self, sequence: str) -> torch.Tensor:
        """
        WT mode with overlapping window aggregation for long sequences.

        Enforces maximum chunk size of ABLANG2_MAX_AA_LEN (510 AA) + 2 special tokens = 512 tokens.

        Returns:
            GPU tensor [L, V] in float32.
        """
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"][0].to(self.device)  # [T]
        T = int(input_ids.size(0))

        # Enforce maximum window size for AbLang2 (512 tokens total including special tokens)
        max_window = ABLANG2_MAX_AA_LEN + 2  # 512 tokens
        effective_window = min(self.window_size, max_window)

        # Short sequence
        if T <= effective_window:
            lprobs = self._model_forward_log_probs(input_ids)
            return lprobs[1:-1, :]  # remove CLS/EOS

        # Long sequence overlapping aggregation (matches the original algorithm pattern)
        token_accum = torch.zeros((T, self.vocab_size), device=self.device, dtype=torch.float32)
        weight_accum = torch.zeros((T,), device=self.device, dtype=torch.float32)

        stride = max(1, (effective_window // 2) - 1)
        start_left = 0
        end_left = effective_window - 1

        start_right = (T - 1) - effective_window + 1
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
            start_center = max(0, (T // 2) - (effective_window // 2))
            end_center = min(T - 1, start_center + effective_window - 1)
            chunk_center = input_ids[start_center : end_center + 1]
            w_center = self.window_weights[: chunk_center.size(0)]
            lprobs_center = self._model_forward_log_probs(chunk_center)

            token_accum[start_center : end_center + 1] += lprobs_center * w_center.unsqueeze(-1)
            weight_accum[start_center : end_center + 1] += w_center

        # Normalize
        weight_accum = torch.clamp(weight_accum, min=1e-6)
        final_lprobs = token_accum / weight_accum.unsqueeze(-1)

        return final_lprobs[1:-1, :]  # remove CLS/EOS

    def _compute_masked_optimal_batched(
        self, sequence: str, batch_size: int = 512
    ) -> torch.Tensor:
        """
        Masked-marginals with optimal window slicing, batched over positions.

        Enforces maximum chunk size of ABLANG2_MAX_AA_LEN (510 AA) + 2 special tokens = 512 tokens.

        Exact semantics:
          - token_idx is in the full token sequence including specials (CLS and EOS)
          - window boundaries match get_optimal_window for T > window_size

        Returns:
            GPU tensor [L, V] in float32. L excludes special tokens.
        """
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids_full = inputs["input_ids"][0].to(self.device)  # [T]
        T = int(input_ids_full.size(0))
        # L is the actual number of residue tokens (excluding CLS at 0 and EOS at T-1)
        L = T - 2
        V = self.vocab_size

        # Enforce maximum window size for AbLang2 (512 tokens total including special tokens)
        max_window = ABLANG2_MAX_AA_LEN + 2  # 512 tokens
        W = min(self.window_size, max_window)
        half = W // 2

        final_lprobs = torch.empty((L, V), device=self.device, dtype=torch.float32)

        # residue token indices in [1..L], because 0 is CLS and L+1 is EOS
        token_idx_all = torch.arange(1, L + 1, device=self.device, dtype=torch.long)

        # Pre-create arange for window offsets for gather
        win_offsets = torch.arange(W, device=self.device, dtype=torch.long)

        for b0 in range(0, L, batch_size):
            tok = token_idx_all[b0 : b0 + batch_size]  # [B]
            B = int(tok.numel())

            if T <= W:
                # Full-sequence batching. One sample per masked position.
                batch_ids = input_ids_full.unsqueeze(0).expand(B, T).clone()  # [B, T]
                batch_ids[torch.arange(B, device=self.device), tok] = self.mask_token_id

                with torch.inference_mode():
                    outputs = self.model(input_ids=batch_ids)
                    # Extract logits - for AbLang2 they're in last_hidden_state
                    if hasattr(outputs, 'last_hidden_state'):
                        logits = outputs.last_hidden_state  # [B, T, V]
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits  # [B, T, V]
                    else:
                        logits = outputs  # Direct tensor
                    # [B, V]
                    logits_at_mask = logits[torch.arange(B, device=self.device), tok, :]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)  # [B, V]

                final_lprobs[b0 : b0 + B] = lprobs
            else:
                # Optimal-window batching without padding.
                # This clamp is equivalent to get_optimal_window(token_idx, T, W) when T > W.
                starts = torch.clamp(tok - half, min=0, max=T - W)  # [B]
                idx = starts.unsqueeze(1) + win_offsets.unsqueeze(0)  # [B, W]
                batch_ids = input_ids_full[idx].clone()  # [B, W]
                mask_pos = tok - starts  # [B] in [0..W-1]
                batch_ids[torch.arange(B, device=self.device), mask_pos] = self.mask_token_id

                with torch.inference_mode():
                    outputs = self.model(input_ids=batch_ids)
                    # Extract logits - for AbLang2 they're in last_hidden_state
                    if hasattr(outputs, 'last_hidden_state'):
                        logits = outputs.last_hidden_state  # [B, W, V]
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits  # [B, W, V]
                    else:
                        logits = outputs  # Direct tensor
                    logits_at_mask = logits[
                        torch.arange(B, device=self.device), mask_pos, :
                    ]  # [B, V]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)  # [B, V]

                final_lprobs[b0 : b0 + B] = lprobs

        return final_lprobs


def score_mutation_delta_logprob(
    global_mut_str: str,
    *,
    sequence: str,
    log_probs_cpu: torch.Tensor,
    tokenizer: AutoTokenizer,
    offset_idx: int = 1,
    strict_wt_check: bool = True,
    input_csv: Optional[str] = None,
    context: Optional[str] = None
) -> float:
    """
    Score a (possibly multi-site) mutation string by sum of delta log-probabilities.

    Score definition:
        sum_i [ log P(mut_i | context) - log P(wt_i | context) ]

    Args:
        global_mut_str: Mutation string in global coordinates, e.g. "H91Y:K120A".
        log_probs_cpu: CPU tensor [L, V]. Values may be float16.
        tokenizer: HF tokenizer used to map AA chars to token ids.
        offset_idx: 1-based offset used in the dataset. Typically 1.
        input_csv: Optional file name to include in WT mismatch errors.

    Returns:
        A Python float score.
    """
    if not global_mut_str or str(global_mut_str).strip() in {"WT"}:
        return 0.0

    L = int(log_probs_cpu.size(0))
    total = 0.0

    # Compute in float32 for stability even if cache is float16.
    lprobs = log_probs_cpu.float()

    for wt, pos, mt in iter_mutations(str(global_mut_str)):
        idx0 = pos - offset_idx
        if idx0 < 0 or idx0 >= L:
            continue

        if strict_wt_check:
            if sequence[idx0].upper() != wt.upper():
                prefix = f"[{context}] " if context else ""
                raise AssertionError(
                    f"[Input file: {input_csv or 'unknown'}] "
                    f"{prefix}WT mismatch at position {pos}. Expected {wt}, found {sequence[idx0]}"
                )

        wt_id = int(tokenizer.convert_tokens_to_ids(wt))
        mt_id = int(tokenizer.convert_tokens_to_ids(mt))
        total += (lprobs[idx0, mt_id] - lprobs[idx0, wt_id]).item()

    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to a HF ESM model",
    )
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
    parser.add_argument(
        "--mode",
        default="wt",
        choices=["wt", "masked"],
        help="Scoring mode",
    )
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
    parser.add_argument(
        "--cache-dir",
        default="./logits_cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--fp16-infer",
        action="store_true",
        help="Enable FP16 inference on GPU",
    )
    parser.add_argument(
        "--no-strict-wt-check",
        dest="strict_wt_check",
        action="store_false",
        help="Disable WT mismatch checks",
    )
    parser.set_defaults(strict_wt_check=True)
    args = parser.parse_args()

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

    engine = AbLang2Engine(
        args.model_path,
        cache_dir,
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
    for group in tqdm(processor, desc="Scoring POI groups"):
        wt_seq = group.wt_concat

        if not wt_seq:
            # Preserve behavior: if no WT sequence, score as 0.0
            for row_idx in group.row_indices:
                final_scores[row_idx] = 0.0
            continue

        log_probs = engine.get_log_probs(wt_seq, args.mode)

        for row_idx, mut_global in zip(group.row_indices, group.mutant_global):
            final_scores[row_idx] = score_mutation_delta_logprob(
                mut_global,
                sequence=wt_seq,
                log_probs_cpu=log_probs,
                tokenizer=engine.tokenizer,
                offset_idx=args.offset,
                strict_wt_check=args.strict_wt_check,
                input_csv = args.input_csv,
                context = f"row {row_idx}" 
            )

    df[f"ablang2_{args.mode}_score"] = final_scores
    df.to_csv(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
