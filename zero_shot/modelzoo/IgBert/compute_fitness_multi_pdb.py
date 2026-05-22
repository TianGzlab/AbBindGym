from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer
from filelock import FileLock

from dms_utils import (
    build_cache_dir,
    build_output_path,
    compute_overlap_weights,
    iter_mutations,
    sanitize_sequence,
    sha256_upper,
)


class IgBertEngine:
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

        self.tokenizer = BertTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            do_lower_case=False
        )

        self.model = (
            BertForMaskedLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        if self.use_fp16_infer:
            self.model = self.model.half()

        self.cache_dir = cache_dir

        os.makedirs(os.path.join(cache_dir, "wt"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "masked"), exist_ok=True)

        self.window_size: int = int(getattr(self.model.config, "max_position_embeddings", 1024))
        self.vocab_size: int = int(self.model.config.vocab_size)
        self.mask_token_id: int = int(self.tokenizer.mask_token_id)

        w = compute_overlap_weights(self.window_size)
        self.window_weights = torch.tensor(w, device=self.device, dtype=torch.float32)

    def _format_paired_sequence(self, sequence_heavy: str, sequence_light: str) -> str:
        """
        Format paired sequences following IgBert documentation.

        Example: "V Q L ... S S [SEP] E V V ... I K"

        Args:
            sequence_heavy: Heavy chain sequence.
            sequence_light: Light chain sequence.

        Returns:
            Formatted string with space-separated amino acids and [SEP] separator.
        """
        return ' '.join(sequence_heavy) + ' [SEP] ' + ' '.join(sequence_light)

    def get_log_probs(
        self, sequence_heavy: str, sequence_light: str, mode: str
    ) -> torch.Tensor:
        """
        Get (and cache) per-residue log-probabilities for paired antibody sequences.

        Args:
            sequence_heavy: Heavy chain sequence string.
            sequence_light: Light chain sequence string.
            mode: "wt" or "masked".

        Returns:
            CPU tensor of shape [L, V] in float16 for storage efficiency.
            Special tokens (CLS/SEP at boundaries) have been removed.
        """
        seq_heavy = sanitize_sequence(sequence_heavy)
        seq_light = sanitize_sequence(sequence_light)

        # Create a unique hash for the paired sequence
        combined_seq = seq_heavy + "|" + seq_light
        seq_hash = sha256_upper(combined_seq)

        cache_path = os.path.join(self.cache_dir, mode, f"{seq_hash}.pt")
        lock_path = cache_path + ".lock"

        with FileLock(lock_path):
            if os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu")

            if mode == "wt":
                log_probs = self._compute_wt(seq_heavy, seq_light)
            elif mode == "masked":
                log_probs = self._compute_masked(seq_heavy, seq_light)
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
            if self.use_fp16_infer:
                # Keep output in float32 for stable log_softmax
                logits = self.model(input_ids_1d.unsqueeze(0)).logits[0].float()
            else:
                logits = self.model(input_ids_1d.unsqueeze(0)).logits[0]
            return torch.log_softmax(logits, dim=-1)

    def _compute_wt(self, sequence_heavy: str, sequence_light: str) -> torch.Tensor:
        """
        WT mode: compute log probs for the full wildtype paired sequence.

        Returns:
            GPU tensor [L, V] in float32, where L = len(heavy) + len(light) + 1 (for middle [SEP]).
        """
        # Format with [SEP] separator as per documentation
        paired_sequence = self._format_paired_sequence(sequence_heavy, sequence_light)

        inputs = self.tokenizer(paired_sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"][0].to(self.device)  # [T]
        T = int(input_ids.size(0))

        # Short sequence
        if T <= self.window_size:
            lprobs = self._model_forward_log_probs(input_ids)
            return lprobs[1:-1, :]  # remove CLS and final SEP

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

        # Center patch if needed
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

        return final_lprobs[1:-1, :]  # remove CLS and final SEP

    def _compute_masked(
        self, sequence_heavy: str, sequence_light: str, batch_size: int = 1280
    ) -> torch.Tensor:
        """
        Masked-marginals with optimal window slicing, batched over positions.

        Returns:
            GPU tensor [L, V] in float32, where L = len(heavy) + len(light) + 1 (for middle [SEP]).
        """
        # Format with [SEP] separator
        paired_sequence = self._format_paired_sequence(sequence_heavy, sequence_light)

        inputs = self.tokenizer(paired_sequence, return_tensors="pt", add_special_tokens=True)
        input_ids_full = inputs["input_ids"][0].to(self.device)  # [T]
        T = int(input_ids_full.size(0))

        # L includes the middle [SEP] token and all amino acids
        L = len(sequence_heavy) + len(sequence_light) + 1  # +1 for middle [SEP]
        V = self.vocab_size
        W = self.window_size
        half = W // 2

        final_lprobs = torch.empty((L, V), device=self.device, dtype=torch.float32)

        # Residue token indices: skip [CLS] at position 0, process positions 1 to L
        token_idx_all = torch.arange(1, L + 1, device=self.device, dtype=torch.long)

        # Pre-create arange for window offsets
        win_offsets = torch.arange(W, device=self.device, dtype=torch.long)

        for b0 in range(0, L, batch_size):
            tok = token_idx_all[b0 : b0 + batch_size]  # [B]
            B = int(tok.numel())

            if T <= W:
                # Full-sequence batching
                batch_ids = input_ids_full.unsqueeze(0).expand(B, T).clone()  # [B, T]
                batch_ids[torch.arange(B, device=self.device), tok] = self.mask_token_id

                with torch.inference_mode():
                    logits = self.model(input_ids=batch_ids).logits  # [B, T, V]
                    logits_at_mask = logits[torch.arange(B, device=self.device), tok, :]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)  # [B, V]

                final_lprobs[b0 : b0 + B] = lprobs
            else:
                # Optimal-window batching
                starts = torch.clamp(tok - half, min=0, max=T - W)  # [B]
                idx = starts.unsqueeze(1) + win_offsets.unsqueeze(0)  # [B, W]
                batch_ids = input_ids_full[idx].clone()  # [B, W]
                mask_pos = tok - starts  # [B] in [0..W-1]
                batch_ids[torch.arange(B, device=self.device), mask_pos] = self.mask_token_id

                with torch.inference_mode():
                    logits = self.model(input_ids=batch_ids).logits  # [B, W, V]
                    logits_at_mask = logits[
                        torch.arange(B, device=self.device), mask_pos, :
                    ]  # [B, V]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)  # [B, V]

                final_lprobs[b0 : b0 + B] = lprobs

        return final_lprobs


def score_mutation_delta_logprob(
    global_mut_heavy: str,
    global_mut_light: str,
    *,
    sequence_heavy: str,
    sequence_light: str,
    log_probs_cpu: torch.Tensor,
    tokenizer: BertTokenizer,
    offset_idx: int = 1,
    strict_wt_check: bool = True,
    input_csv: Optional[str] = None,
    context: Optional[str] = None
) -> float:
    """
    Score mutations by sum of delta log-probabilities.

    Note: log_probs_cpu includes the middle [SEP] token at position len(heavy).

    Returns:
        A Python float score.
    """
    if (not global_mut_heavy or str(global_mut_heavy).strip() in {"WT", ""}) and \
       (not global_mut_light or str(global_mut_light).strip() in {"WT", ""}):
        return 0.0

    # Compute in float32
    lprobs = log_probs_cpu.float()
    total = 0.0

    # Process heavy chain mutations
    if global_mut_heavy and str(global_mut_heavy).strip() not in {"WT", ""}:
        for wt, pos, mt in iter_mutations(str(global_mut_heavy)):
            idx0 = pos - offset_idx
            if 0 <= idx0 < len(sequence_heavy):
                if strict_wt_check and sequence_heavy[idx0].upper() != wt.upper():
                    prefix = f"[{context}] " if context else ""
                    raise AssertionError(
                        f"[Input file: {input_csv or 'unknown'}] "
                        f"{prefix}WT mismatch in heavy chain at position {pos}. "
                        f"Expected {wt}, found {sequence_heavy[idx0]}"
                    )

                wt_id = int(tokenizer.convert_tokens_to_ids(wt))
                mt_id = int(tokenizer.convert_tokens_to_ids(mt))
                total += (lprobs[idx0, mt_id] - lprobs[idx0, wt_id]).item()

    # Process light chain mutations (offset by heavy chain length + 1 for middle [SEP])
    if global_mut_light and str(global_mut_light).strip() not in {"WT", ""}:
        for wt, pos, mt in iter_mutations(str(global_mut_light)):
            idx0 = pos - offset_idx
            if 0 <= idx0 < len(sequence_light):
                if strict_wt_check and sequence_light[idx0].upper() != wt.upper():
                    prefix = f"[{context}] " if context else ""
                    raise AssertionError(
                        f"[Input file: {input_csv or 'unknown'}] "
                        f"{prefix}WT mismatch in light chain at position {pos}. "
                        f"Expected {wt}, found {sequence_light[idx0]}"
                    )

                # Light chain index: skip heavy chain + middle [SEP]
                combined_idx = len(sequence_heavy) + 1 + idx0

                wt_id = int(tokenizer.convert_tokens_to_ids(wt))
                mt_id = int(tokenizer.convert_tokens_to_ids(mt))
                total += (lprobs[combined_idx, mt_id] - lprobs[combined_idx, wt_id]).item()

    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to IgBert model",
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

    engine = IgBertEngine(
        args.model_path,
        cache_dir,
        device=args.device,
        use_fp16_infer=args.fp16_infer,
    )

    sep = "\t" if args.input_csv.endswith(".tsv") else ","
    print(f"Reading input file: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=sep)

    final_scores: List[Optional[float]] = [None] * len(df)

    print(f"Starting scoring. Mode: {args.mode}")

    # Process each row
    for idx in tqdm(range(len(df)), desc="Scoring mutations"):
        try:
            # Get wildtype sequences (dict format: {'H': 'seq1', 'L': 'seq2'})
            wt_seq_dict = eval(df.loc[idx, 'wildtype_sequence']) if isinstance(df.loc[idx, 'wildtype_sequence'], str) else df.loc[idx, 'wildtype_sequence']

            # Get mutations (dict format: {'H': 'H91Y:K120A', 'L': ''})
            mutant_dict = eval(df.loc[idx, 'mutant']) if isinstance(df.loc[idx, 'mutant'], str) else df.loc[idx, 'mutant']

            # Extract heavy and light chain sequences
            heavy_chain_id = None
            light_chain_id = None

            for chain_id in wt_seq_dict.keys():
                if chain_id in ['H', 'VH', 'HEAVY']:
                    heavy_chain_id = chain_id
                elif chain_id in ['L', 'VL', 'LIGHT', 'K', 'KAPPA']:
                    light_chain_id = chain_id

            if heavy_chain_id is None or light_chain_id is None:
                final_scores[idx] = 0.0
                continue

            seq_heavy = sanitize_sequence(wt_seq_dict[heavy_chain_id])
            seq_light = sanitize_sequence(wt_seq_dict[light_chain_id])

            mut_heavy = mutant_dict.get(heavy_chain_id, "")
            mut_light = mutant_dict.get(light_chain_id, "")

            if not seq_heavy or not seq_light:
                final_scores[idx] = 0.0
                continue

            # Get log probabilities
            log_probs = engine.get_log_probs(seq_heavy, seq_light, args.mode)

            # Score mutations
            score = score_mutation_delta_logprob(
                mut_heavy,
                mut_light,
                sequence_heavy=seq_heavy,
                sequence_light=seq_light,
                log_probs_cpu=log_probs,
                tokenizer=engine.tokenizer,
                offset_idx=args.offset,
                strict_wt_check=args.strict_wt_check,
                input_csv=args.input_csv,
                context=f"row {idx}"
            )

            final_scores[idx] = score

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            final_scores[idx] = None

    df[f"igbert_{args.mode}_score"] = final_scores
    df.to_csv(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()