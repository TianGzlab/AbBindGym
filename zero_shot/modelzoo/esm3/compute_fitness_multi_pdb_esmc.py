from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd
import torch
from esm.models.esmc import ESMC
from filelock import FileLock
from tqdm import tqdm

from dms_utils import (
    build_cache_dir,
    build_output_path,
    compute_overlap_weights,
    iter_mutations,
    preprocess_dataframe,
    sanitize_sequence,
    sha256_upper,
)


class ESMEngine:
    """
    Unified Inference Engine for ESM3 and ESMC models.
    Supports:
      - ESMC (e.g., esmc_600m)
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        device: Optional[str] = None,
        use_fp16_infer: bool = False,
    ) -> None:
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16_infer = bool(use_fp16_infer and self.device.type == "cuda")

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}. FP16 inference: {self.use_fp16_infer}")

        self.model_type = "esmc"
        self.model = ESMC.from_pretrained(model_name).to(self.device).eval()

        if self.use_fp16_infer:
            self.model = self.model.half()

        self.tokenizer = self.model.tokenizer

        self.cache_dir = cache_dir
        os.makedirs(os.path.join(cache_dir, "wt"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "masked"), exist_ok=True)

        self.window_size: int = 2048
        self.vocab_size: int = self.tokenizer.vocab_size
        self.mask_token_id: int = self.tokenizer.mask_token_id

        print(f"Window size: {self.window_size}, Vocab size: {self.vocab_size}")

        w = compute_overlap_weights(self.window_size)
        self.window_weights = torch.tensor(w, device=self.device, dtype=torch.float32)

    def get_log_probs(self, sequence: str, mode: str) -> torch.Tensor:
        """
        Get (and cache) per-residue log-probabilities.
        """
        seq = sanitize_sequence(sequence)
        seq_hash = sha256_upper(seq)

        cache_path = os.path.join(self.cache_dir, mode, f"{seq_hash}.pt")
        lock_path = cache_path + ".lock"

        with FileLock(lock_path):
            # Check cache inside lock to avoid race conditions
            if os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu", weights_only=True)

            # Compute inside lock to prevent duplicate computation
            if mode == "wt":
                log_probs = self._compute_wt_overlapping(seq)
            elif mode == "masked":
                log_probs = self._compute_masked_optimal_batched(seq)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            log_probs_cpu = log_probs.detach().to("cpu").half()
            torch.save(log_probs_cpu, cache_path)
            return log_probs_cpu

    def _model_forward_log_probs(self, input_ids_1d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ESMC model.
        Returns log probabilities of shape [T, V] where T is sequence length.
        """
        with torch.inference_mode():
            input_tensor = input_ids_1d.unsqueeze(0)
            output = self.model(sequence_tokens=input_tensor)
            logits = output.sequence_logits[0]  # [T, V]

            if logits.size(-1) > self.vocab_size:
                logits = logits[..., : self.vocab_size]

            if self.use_fp16_infer:
                logits = logits.float()

            return torch.log_softmax(logits, dim=-1)

    def _compute_wt_overlapping(self, sequence: str) -> torch.Tensor:
        """
        WT mode with overlapping window aggregation.
        """
        raw_ids = self.tokenizer.encode(sequence)
        if isinstance(raw_ids, list):
            input_ids = torch.tensor(raw_ids, device=self.device)
        else:
            input_ids = raw_ids.to(self.device)

        T = int(input_ids.size(0))

        if T <= self.window_size:
            lprobs = self._model_forward_log_probs(input_ids)
            return lprobs[1:-1, :]

        # Long sequence handling (Standard overlapping logic)
        token_accum = torch.zeros(
            (T, self.vocab_size), device=self.device, dtype=torch.float32
        )
        weight_accum = torch.zeros((T,), device=self.device, dtype=torch.float32)

        stride = max(1, (self.window_size // 2) - 1)
        W = self.window_size

        # Process windows from left to right with stride
        start = 0
        while start < T:
            end = min(start + W, T)
            # Adjust start if we're at the end and window would be too small
            if end - start < W and start > 0:
                start = max(0, T - W)
                end = T

            chunk = input_ids[start:end]
            chunk_len = chunk.size(0)
            w = self.window_weights[:chunk_len]
            lprobs = self._model_forward_log_probs(chunk)

            # Ensure lprobs is float32 for accumulation
            if lprobs.dtype != torch.float32:
                lprobs = lprobs.float()

            token_accum[start:end] += lprobs * w.unsqueeze(-1)
            weight_accum[start:end] += w

            # Move to next window
            if end >= T:
                break
            start += stride

        weight_accum = torch.clamp(weight_accum, min=1e-6)
        final_lprobs = token_accum / weight_accum.unsqueeze(-1)

        return final_lprobs[1:-1, :]

    def _compute_masked_optimal_batched(
        self, sequence: str, batch_size: int = 64
    ) -> torch.Tensor:
        raw_ids = self.tokenizer.encode(sequence)
        if isinstance(raw_ids, list):
            input_ids_full = torch.tensor(raw_ids, device=self.device)
        else:
            input_ids_full = raw_ids.to(self.device)

        T = int(input_ids_full.size(0))  # Total tokens including BOS/EOS
        L = len(sequence)  # Number of amino acids
        V = self.vocab_size
        W = self.window_size
        half = W // 2

        final_lprobs = torch.empty((L, V), device=self.device, dtype=torch.float32)

        # token_idx_all: positions of amino acids in tokenized sequence (1 to L inclusive)
        # Position 0 is BOS, position L+1 is EOS
        token_idx_all = torch.arange(1, L + 1, device=self.device, dtype=torch.long)
        win_offsets = torch.arange(W, device=self.device, dtype=torch.long)

        for b0 in range(0, L, batch_size):
            tok = token_idx_all[b0 : b0 + batch_size]
            B = int(tok.numel())

            if T <= W:
                # 确保不会因为batch_size过大导致索引越界
                if b0 + batch_size > L:
                    B = L - b0
                    tok = token_idx_all[b0 : b0 + B]

                batch_ids = input_ids_full.unsqueeze(0).expand(B, T).clone()
                batch_ids[torch.arange(B, device=self.device), tok] = self.mask_token_id

                with torch.inference_mode():
                    output = self.model(sequence_tokens=batch_ids)
                    logits = output.sequence_logits

                    logits_at_mask = logits[torch.arange(B, device=self.device), tok, :]
                    if logits_at_mask.size(-1) > self.vocab_size:
                        logits_at_mask = logits_at_mask[..., : self.vocab_size]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)

                final_lprobs[b0 : b0 + B] = lprobs
            else:
                # For long sequences, extract a window around each position
                # starts: where each window begins in the full sequence
                starts = torch.clamp(tok - half, min=0, max=T - W)
                idx = starts.unsqueeze(1) + win_offsets.unsqueeze(0)
                batch_ids = input_ids_full[idx].clone()
                # mask_pos: position within the window where the mask should be placed
                mask_pos = tok - starts
                batch_ids[torch.arange(B, device=self.device), mask_pos] = (
                    self.mask_token_id
                )

                with torch.inference_mode():
                    output = self.model(sequence_tokens=batch_ids)
                    logits = output.sequence_logits

                    logits_at_mask = logits[
                        torch.arange(B, device=self.device), mask_pos, :
                    ]
                    if logits_at_mask.size(-1) > self.vocab_size:
                        logits_at_mask = logits_at_mask[..., : self.vocab_size]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)

                final_lprobs[b0 : b0 + B] = lprobs

        return final_lprobs


def score_mutation_delta_logprob(
    global_mut_str: str,
    *,
    sequence: str,
    log_probs_cpu: torch.Tensor,
    tokenizer,
    offset_idx: int = 1,
    strict_wt_check: bool = True,
    input_csv: Optional[str] = None,
    context: Optional[str] = None,
) -> float:
    if not global_mut_str or str(global_mut_str).strip() in {"WT"}:
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
                prefix = f"[{context}] " if context else ""
                raise AssertionError(
                    f"[Input file: {input_csv or 'unknown'}] "
                    f"{prefix}WT mismatch at position {pos}. Expected {wt}, found {sequence[idx0]}"
                )

        wt_encoded = tokenizer.encode(wt)
        if torch.is_tensor(wt_encoded):
            wt_encoded = wt_encoded.tolist()

        mt_encoded = tokenizer.encode(mt)
        if torch.is_tensor(mt_encoded):
            mt_encoded = mt_encoded.tolist()

        wt_id = wt_encoded[1]
        mt_id = mt_encoded[1]

        total += (lprobs[idx0, mt_id] - lprobs[idx0, wt_id]).item()

    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name or path. E.g., 'esmc_600m' or 'esm3_sm_open_v1'",
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
        "--device",
        default="cuda:0",
        help="running device",
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
        default="./logits_cache_esm",
        help="Cache directory",
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

    # Cache 目录逻辑
    cache_dir = build_cache_dir(
        args.cache_dir,
        args.model_name,
        fp16=args.fp16_infer,
        focus=(args.focus == 1),
    )
    print(f"Cache dir: {cache_dir}")

    output_path = build_output_path(
        args.output_dir,
        args.input_csv,
        args.model_name,
        mode=args.mode,
        fp16=args.fp16_infer,
        focus=(args.focus == 1),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output file: {output_path}")

    engine = ESMEngine(
        args.model_name,
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
                input_csv=args.input_csv,
                context=f"row {row_idx}",
            )

    prefix = "esmc" if "esmc" in args.model_name.lower() else "esm3"
    df[f"{prefix}_{args.mode}_score"] = final_scores

    df.to_csv(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
