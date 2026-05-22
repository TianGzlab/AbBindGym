from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dms_utils import (
    build_cache_dir,
    build_output_path,
    iter_mutations,
    preprocess_dataframe,
    sanitize_sequence,
    sha256_upper,
)


class CausalLMEngine:
    """
    Causal language model inference engine for mutation fitness scoring.

    Core behavior mirrors the original compute_fitness.py:
        - Teacher-forced log-probabilities over the full sequence.
        - Optional mirroring (forward + reverse) with per-chunk averaging.
        - Chunking to respect the model context length.

    Cached sequence-level scores are stored under:
        {cache_dir}/causal_inference/{sha256(sequence)}.pt
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        *,
        device: Optional[str] = None,
        use_fp16_infer: bool = False,
        model_context_len: Optional[int] = None,
    ) -> None:
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16_infer = bool(use_fp16_infer and self.device.type == "cuda")
        self.loss_fn = CrossEntropyLoss(reduction="mean")

        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}. FP16 inference: {self.use_fp16_infer}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        if self.use_fp16_infer:
            self.model = self.model.half()

        # Context length follows the original ProtGPT2 script (1023) unless overridden.
        default_ctx = getattr(
            self.model.config,
            "n_positions",
            getattr(self.model.config, "max_position_embeddings", 1024),
        )
        self.model_context_len: int = int(model_context_len or max(1, default_ctx - 1))

        self.cache_dir = os.path.join(cache_dir, "causal_inference")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _chunk_sequence(self, seq: str) -> List[str]:
        """
        Split a sequence into contiguous chunks within the model context length.
        """
        if len(seq) <= self.model_context_len:
            return [seq]

        chunks: List[str] = []
        start = 0
        while start < len(seq):
            end = min(start + self.model_context_len, len(seq))
            chunks.append(seq[start:end])
            start = end
        return chunks

    def _score_single_sequence(self, sequence: str, *, mirror: bool) -> float:
        """
        Score one sequence with teacher forcing. Returns negative cross entropy averaged
        across chunks (and mirroring if enabled), matching compute_fitness.py semantics.
        """
        seq = sanitize_sequence(sequence)
        if not seq:
            return 0.0

        seq_hash = sha256_upper(seq)
        cache_path = os.path.join(self.cache_dir, f"{seq_hash}.pt")
        if os.path.exists(cache_path):
            return float(torch.load(cache_path, map_location="cpu"))

        total_score = 0.0
        chunks = self._chunk_sequence(seq)
        denom = len(chunks) * (2 if mirror else 1)
        if denom == 0:
            return 0.0

        amp_ctx = (
            torch.amp.autocast(enabled=True, device_type="cuda")
            if self.use_fp16_infer and self.device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode(), amp_ctx:
            for chunk in chunks:
                directions = (chunk, chunk[::-1]) if mirror else (chunk,)
                for p in directions:
                    ids = self.tokenizer.encode(p)
                    if len(ids) < 2:
                        continue
                    ids_tensor = torch.tensor(ids, device=self.device, dtype=torch.long).unsqueeze(
                        0
                    )
                    input_ids = ids_tensor[:, :-1]
                    targets = ids_tensor[:, 1:]

                    logits = self.model(input_ids).logits  # [1, T, V]
                    loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                    )
                    total_score += -loss.item()

        avg_score = total_score / denom
        torch.save(torch.tensor(avg_score, dtype=torch.float32), cache_path)
        return float(avg_score)

    def score_sequences(self, sequences: List[str], *, mirror: bool = True) -> np.ndarray:
        """
        Score a list of sequences. Returns numpy array of scores in float32.
        """
        scores: List[float] = []
        for seq in sequences:
            scores.append(self._score_single_sequence(seq, mirror=mirror))
        return np.array(scores, dtype=np.float32)


def apply_mutations(
    sequence: str,
    mutations: str,
    *,
    offset: int = 1,
    input_csv: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    """
    Apply a global mutation string to a sequence (e.g., "H91Y:Y92F").
    """
    if not mutations or str(mutations).strip().upper() == "WT":
        return sequence

    seq = list(sanitize_sequence(sequence))
    mutated_positions = {}
    for wt, pos, mt in iter_mutations(str(mutations)):
        idx = pos - offset
        if idx < 0 or idx >= len(seq):
            continue
        current = seq[idx]
        if idx in mutated_positions:
            # duplicate mutation at same site; allow if desired residue matches current state
            if current.upper() == mt.upper():
                continue
            else:
                prefix = f"[{context}]" if context else ""
                raise AssertionError(
                    f"[Input file: {input_csv or 'unknown'}] "
                    f"{prefix}Conflicting mutations at position {pos}: already set to {current}, got request {wt}->{mt}"
                )
        if current.upper() != wt.upper():
            # if already differs but matches target, treat as duplicate identical mutation
            if current.upper() == mt.upper():
                mutated_positions[idx] = mt
                continue
            prefix = f"[{context}]" if context else ""
            raise AssertionError(
                f"[Input file: {input_csv or 'unknown'}] "
                f"{prefix}WT mismatch at position {pos}: expected {wt}, found {current}"
            )
        seq[idx] = mt
        mutated_positions[idx] = mt
    return "".join(seq)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to a HF causal LM (e.g., ProtGPT2)",
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
        default="delta",
        choices=["mutant", "delta"],
        help="mutant = score mutated sequences directly (compute_fitness style); delta = mutant - WT",
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
        "--context-len",
        type=int,
        default=None,
        help="Optional override for model context length (tokens).",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Disable reverse-sequence scoring. Enabled by default.",
    )
    parser.set_defaults(mirror=True)

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

    engine = CausalLMEngine(
        args.model_path,
        cache_dir,
        device=args.device,
        use_fp16_infer=args.fp16_infer,
        model_context_len=args.context_len,
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

        mutant_sequences = []
        for mut_global, ridx in zip(group.mutant_global, group.row_indices):
            mutant_sequences.append(
                apply_mutations(
                    wt_seq,
                    mut_global,
                    offset=args.offset,
                    input_csv=args.input_csv,
                    context=f"row {ridx}",
                )
            )

        if args.mode == "delta":
            scores = engine.score_sequences([wt_seq] + mutant_sequences, mirror=args.mirror)
            wt_score = scores[0]
            mut_scores = scores[1:]
            final_batch = mut_scores - wt_score
        else:
            final_batch = engine.score_sequences(mutant_sequences, mirror=args.mirror)

        for row_idx, score_val in zip(group.row_indices, final_batch):
            final_scores[row_idx] = float(score_val)

    score_col = "ProtGPT2_score" if args.mode == "mutant" else "ProtGPT2_delta_score"
    df[score_col] = final_scores
    df.to_csv(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
