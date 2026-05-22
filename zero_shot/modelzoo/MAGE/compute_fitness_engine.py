from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
)

from data_utils import DMS_file_for_LLM
from progen2.progen.modeling_progen import ProGenForCausalLM


def sha256_upper(s: str) -> str:
    return hashlib.sha256(s.upper().encode("utf-8")).hexdigest().upper()


def sanitize_sequence(seq: str) -> str:
    if seq is None:
        return ""
    return "".join(str(seq).split()).upper()


def derive_model_id(model_path: str, *, override: Optional[str] = None) -> str:
    """
    Derive a human-meaningful model identifier for naming cache and output columns.
    Priority:
      1) override if provided
      2) if model_path exists on disk, use basename to avoid leaking absolute paths
      3) otherwise treat as HF repo id, e.g. "hugohrban/progen2-small"
    """
    if override and str(override).strip():
        return str(override).strip().rstrip("/")

    p = str(model_path).strip().rstrip("/")
    if os.path.exists(p):
        return os.path.basename(os.path.normpath(p))
    return p


def slugify_model_id(model_id: str) -> str:
    """
    Filesystem and CSV-header safe identifier.
    Keep alnum, dot, underscore, dash.
    Replace "/" with "__".
    Other chars become "_".
    """
    s = str(model_id).strip()
    s = s.replace("\\", "/").replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_model"


def format_progen2_sequence(seq_wo_terminals: str) -> str:
    s = sanitize_sequence(seq_wo_terminals)
    if len(s) >= 2 and s[0] == "1" and s[-1] == "2":
        return s
    return "1" + s + "2"


def build_cache_dir(root_cache_dir: str, model_id_slug: str, *, fp16: bool, focus: bool) -> str:
    flag = f"fp16{int(fp16)}_focus{int(focus)}"
    out = os.path.join(root_cache_dir, "progen2", model_id_slug, flag)
    os.makedirs(out, exist_ok=True)
    return out


def build_output_path(
    output_dir: str,
    input_csv: str,
    model_id_slug: str,
    *,
    fp16: bool,
    focus: bool,
    reduction: str,
) -> str:
    base = os.path.basename(input_csv)
    stem, ext = os.path.splitext(base)
    suffix = f"progen2_{model_id_slug}_fp16{int(fp16)}_focus{int(focus)}_{reduction}"
    out_name = f"{stem}.{suffix}{ext if ext else '.csv'}"
    return os.path.join(output_dir, out_name)


def create_model(ckpt_dir: str, fp16: bool):
    """
    Load ProGen2 model via transformers AutoModelForCausalLM from a local checkpoint directory (offline-safe).

    Note:
      - ckpt should be a resolved local directory, or a repo id that has already been cached.
      - local_files_only=True ensures no network calls happen.
    """
    model = ProGenForCausalLM.from_pretrained(
        ckpt_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float16 if fp16 else None,
    )
    model.eval()
    return model


def create_tokenizer(ckpt_dir: str) -> Tokenizer:
    return PreTrainedTokenizerFast.from_pretrained(ckpt_dir)


def _chunk_sequence(seq: str, chunk_len: int) -> List[str]:
    """
    Split sequence into non-empty contiguous chunks of length <= chunk_len.
    """
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be > 0, got {chunk_len}")
    if len(seq) <= chunk_len:
        return [seq]
    chunks: List[str] = []
    for start in range(0, len(seq), chunk_len):
        end = min(start + chunk_len, len(seq))
        if end > start:
            chunks.append(seq[start:end])
    return chunks


def calc_fitness(
    model,
    prots,
    tokenizer,
    device: str = "cuda:0",
    model_context_len: int = 1024,
    fp16: bool = False,
    reduction: str = "sum",
):
    """
    Compute fitness score (negative CE loss) for each protein sequence in `prots`.

    Maintains original behavior:
      - chunking for long sequences
      - mirroring score by forward and reversed chunk, then normalize by 2
      - remove terminals if last token is BOS or EOS
      - restrict logits to AA vocab tokens [5..29], shift targets accordingly
      - output is numpy array of per-seq scores
    """
    loss_list: List[float] = []
    loss_fn = CrossEntropyLoss()

    bos_token, eos_token = 3, 4
    first_token, last_token = 5, 29

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=fp16):
            for prot in tqdm(prots, leave=False):
                loss_val = 0.0

                sequence_chunks = _chunk_sequence(str(prot), model_context_len)

                for chunk in sequence_chunks:
                    for p in (chunk, chunk[::-1]):
                        ids = torch.tensor(tokenizer.encode(p), device=device, dtype=torch.long)

                        if ids.numel() < 2:
                            continue

                        input_ids = ids[:-1]
                        targets = ids[1:]

                        out = model(input_ids.unsqueeze(0))
                        logits = out.logits.squeeze(0)

                        if targets.numel() > 0 and targets[-1].item() in (
                            bos_token,
                            eos_token,
                        ):
                            logits = logits[:-1, ...]
                            targets = targets[:-1]

                        if targets.numel() == 0:
                            continue

                        if (targets == bos_token).any().item():
                            raise AssertionError("Targets contain BOS token unexpectedly.")
                        if (targets == eos_token).any().item():
                            raise AssertionError("Targets contain EOS token unexpectedly.")

                        logits = logits[:, first_token : (last_token + 1)]
                        targets = targets - first_token

                        if logits.shape[1] != (last_token - first_token + 1):
                            raise AssertionError("Unexpected restricted vocab size.")

                        loss = loss_fn(
                            input=logits.reshape(-1, logits.size(-1)),
                            target=targets.reshape(-1),
                        )
                        loss_val += -float(loss.item())

                loss_val /= 2.0

                if reduction == "mean":
                    loss_val /= max(1, len(str(prot)))

                loss_list.append(loss_val)

    return np.array(loss_list, dtype=np.float32)


@dataclass(frozen=True)
class CacheStats:
    hit: int = 0
    miss: int = 0


class ProGen2Engine:
    """
    ProGen2 DMS scoring engine with ESM-like pipeline features:
      - offline-safe model and tokenizer load
      - per-sequence fitness caching on disk
      - group scoring by POI with WT baseline subtraction

    Cache layout:
      {cache_dir}/fitness/{sha256(sequence_with_terminals)}.pt

    Note:
      cache_dir itself is already model-specific by design.
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        *,
        device: Optional[str] = None,
        use_fp16_infer: bool = False,
        reduction: str = "sum",
        cache_mutants: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16_infer = bool(use_fp16_infer and str(self.device).startswith("cuda"))

        self.reduction = str(reduction)
        if self.reduction not in {"sum", "mean"}:
            raise ValueError(f"reduction must be 'sum' or 'mean', got {self.reduction}")

        self.cache_mutants = bool(cache_mutants)

        print(f"Loading ProGen2 model from: {model_path}")
        print(
            f"Device: {self.device}. FP16 inference: {self.use_fp16_infer}. Reduction: {self.reduction}"
        )

        self.model = create_model(ckpt_dir=model_path, fp16=self.use_fp16_infer)
        self.tokenizer = create_tokenizer(ckpt_dir=model_path)

        self.model_context_len: int = int(getattr(self.model.config, "n_positions", 1024))

        self.cache_dir = cache_dir
        self.fitness_dir = os.path.join(cache_dir, "fitness")
        os.makedirs(self.fitness_dir, exist_ok=True)

        self._mem_cache: Dict[str, float] = {}

    def _cache_path(self, seq_with_terminals: str) -> str:
        key = sha256_upper(seq_with_terminals)
        return os.path.join(self.fitness_dir, f"{key}.pt")

    def get_fitness(self, seq_with_terminals: str) -> Tuple[float, bool]:
        """
        Return fitness score for a single formatted sequence.
        Returns (score, cache_hit).
        """
        s = sanitize_sequence(seq_with_terminals)

        if s in self._mem_cache:
            return float(self._mem_cache[s]), True

        path = self._cache_path(s)
        if os.path.exists(path):
            val = torch.load(path, map_location="cpu")
            score = float(val.item()) if torch.is_tensor(val) else float(val)
            self._mem_cache[s] = score
            return score, True

        scores = calc_fitness(
            model=self.model,
            prots=np.array([s]),
            tokenizer=self.tokenizer,
            device=str(self.device),
            model_context_len=self.model_context_len,
            fp16=self.use_fp16_infer,
            reduction=self.reduction,
        )
        score = float(scores[0])

        self._mem_cache[s] = score
        torch.save(torch.tensor(score, dtype=torch.float32), path)
        return score, False

    def get_fitness_many(
        self, seqs_with_terminals: List[str]
    ) -> Tuple[Dict[str, float], CacheStats]:
        """
        Compute fitness for multiple sequences with disk caching.
        Returns mapping {seq: score} for sanitized seq keys.
        """
        hit = 0
        miss = 0

        sanitized = [sanitize_sequence(s) for s in seqs_with_terminals]
        uniq = list(dict.fromkeys(sanitized))

        out: Dict[str, float] = {}
        to_compute: List[str] = []

        for s in uniq:
            if s in self._mem_cache:
                out[s] = float(self._mem_cache[s])
                hit += 1
                continue

            path = self._cache_path(s)
            if os.path.exists(path):
                val = torch.load(path, map_location="cpu")
                score = float(val.item()) if torch.is_tensor(val) else float(val)
                out[s] = score
                self._mem_cache[s] = score
                hit += 1
            else:
                to_compute.append(s)
                miss += 1

        if to_compute:
            scores = calc_fitness(
                model=self.model,
                prots=np.array(to_compute),
                tokenizer=self.tokenizer,
                device=str(self.device),
                model_context_len=self.model_context_len,
                fp16=self.use_fp16_infer,
                reduction=self.reduction,
            )
            for s, sc in zip(to_compute, scores.tolist()):
                sc_f = float(sc)
                out[s] = sc_f
                self._mem_cache[s] = sc_f
                torch.save(torch.tensor(sc_f, dtype=torch.float32), self._cache_path(s))

        return out, CacheStats(hit=hit, miss=miss)

    def score_group_delta(
        self,
        wt_seq_wo_terminals: str,
        mutant_seqs_wo_terminals: List[str],
    ) -> List[float]:
        """
        For one POI group:
          delta_i = fitness(mut_i) - fitness(wt)
        """
        wt_in = format_progen2_sequence(wt_seq_wo_terminals)
        mut_in = [format_progen2_sequence(s) for s in mutant_seqs_wo_terminals]

        if not self.cache_mutants:
            wt_score, _ = self.get_fitness(wt_in)
            mut_scores = calc_fitness(
                model=self.model,
                prots=np.array([sanitize_sequence(s) for s in mut_in]),
                tokenizer=self.tokenizer,
                device=str(self.device),
                model_context_len=self.model_context_len,
                fp16=self.use_fp16_infer,
                reduction=self.reduction,
            ).astype(np.float32)
            return [float(x - wt_score) for x in mut_scores.tolist()]

        score_map, _stats = self.get_fitness_many([wt_in] + mut_in)
        wt_score = float(score_map[sanitize_sequence(wt_in)])
        deltas = [float(score_map[sanitize_sequence(s)] - wt_score) for s in mut_in]
        return deltas


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to a HF ProGen2 checkpoint directory (offline-safe) or a cached repo id",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional explicit model id used for naming, e.g. hugohrban/progen2-small",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV or TSV file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for result CSV/TSV",
    )
    parser.add_argument(
        "--focus",
        type=int,
        default=1,
        help="1=drop silent chains, 0=keep all chains, passed into DMS_file_for_LLM",
    )
    parser.add_argument(
        "--cache-dir",
        default="./logits_cache",
        help="Cache root directory",
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
        "--reduction",
        default="sum",
        choices=["sum", "mean"],
        help="Fitness reduction mode, matches calc_fitness",
    )
    parser.add_argument(
        "--cache-mutants",
        action="store_true",
        help="Cache mutant fitness on disk, default is False",
    )

    args = parser.parse_args()

    focus_flag = bool(args.focus == 1)
    effective_fp16 = bool(args.fp16_infer)

    model_id = derive_model_id(args.model_path, override=args.model_id)
    model_id_slug = slugify_model_id(model_id)
    print(f"Model id: {model_id}")
    print(f"Model id slug: {model_id_slug}")

    cache_dir = build_cache_dir(
        args.cache_dir,
        model_id_slug,
        fp16=effective_fp16,
        focus=focus_flag,
    )
    print(f"Cache dir: {cache_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = build_output_path(
        args.output_dir,
        args.input_csv,
        model_id_slug,
        fp16=effective_fp16,
        focus=focus_flag,
        reduction=str(args.reduction),
    )
    print(f"Output file: {output_path}")

    engine = ProGen2Engine(
        model_path=args.model_path,
        cache_dir=cache_dir,
        device=args.device,
        use_fp16_infer=effective_fp16,
        reduction=str(args.reduction),
        cache_mutants=bool(args.cache_mutants),
    )

    sep = "\t" if args.input_csv.endswith(".tsv") else ","
    print(f"Reading input file: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=sep, low_memory=False)

    if "POI" not in df.columns:
        raise ValueError("Input file must contain column: POI")

    final_scores: List[Optional[float]] = [None] * len(df)

    print("Starting scoring. Grouping by POI.")
    for _poi, g in tqdm(df.groupby("POI"), desc="Scoring POI groups"):
        g2 = DMS_file_for_LLM(g, focus=focus_flag)
        if g2 is None or len(g2) == 0:
            for row_idx in g.index.tolist():
                final_scores[row_idx] = 0.0
            continue

        if "wildtype_sequence" not in g2.columns or "mutated_sequence" not in g2.columns:
            raise ValueError(
                "DMS_file_for_LLM output must contain columns: wildtype_sequence, mutated_sequence"
            )

        wt_seq = str(g2["wildtype_sequence"].values[0])
        if not wt_seq or str(wt_seq).strip() == "":
            for row_idx in g2.index.tolist():
                final_scores[row_idx] = 0.0
            continue

        mut_seqs = [str(s) for s in g2["mutated_sequence"].tolist()]
        deltas = engine.score_group_delta(
            wt_seq_wo_terminals=wt_seq,
            mutant_seqs_wo_terminals=mut_seqs,
        )

        for row_idx, sc in zip(g2.index.tolist(), deltas):
            final_scores[row_idx] = float(sc)

    col_name = f"progen2_{model_id_slug}_{str(args.reduction)}_score"
    df[col_name] = final_scores

    df.to_csv(output_path, index=False, sep=sep)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
