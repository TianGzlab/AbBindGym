from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from data_utils import DMS_file_for_LLM
from progen3.batch_preparer import ProGen3BatchPreparer
from progen3.modeling import MoeCausalOutputWithPast, ProGen3ForCausalLM


def sha256_upper(s: str) -> str:
    return hashlib.sha256(s.upper().encode("utf-8")).hexdigest().upper()


def sanitize_sequence(seq: str) -> str:
    if seq is None:
        return ""
    return "".join(str(seq).split()).upper()


def derive_model_id(model_path: str, *, override: Optional[str] = None) -> str:
    if override and str(override).strip():
        return str(override).strip().rstrip("/")
    p = str(model_path).strip().rstrip("/")
    if os.path.exists(p):
        return os.path.basename(os.path.normpath(p))
    return p


def slugify_model_id(model_id: str) -> str:
    s = str(model_id).strip()
    s = s.replace("\\", "/").replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_model"


def build_cache_dir(
    root_cache_dir: str, model_id_slug: str, *, fp16: bool, focus: bool
) -> str:
    flag = f"fp16{int(fp16)}_focus{int(focus)}"
    out = os.path.join(root_cache_dir, "progen3", model_id_slug, flag)
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
    suffix = f"progen3_{model_id_slug}_fp16{int(fp16)}_focus{int(focus)}_{reduction}"
    out_name = f"{stem}.{suffix}{ext if ext else '.csv'}"
    return os.path.join(output_dir, out_name)


class ProGen3ScorerInternal:
    def __init__(
        self, model_path: str, device: str, fp16: bool, reduction: str = "mean"
    ):
        self.device = device
        self.reduction = reduction
        self.fp16 = fp16

        print(f"Loading ProGen3 model from: {model_path} ...")
        dtype = torch.bfloat16
        self.model = ProGen3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.batch_preparer = ProGen3BatchPreparer()

    def _compute_nll(self, model_forward_kwargs: dict[str, Any]) -> torch.Tensor:
        for k, v in model_forward_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_forward_kwargs[k] = v.to(self.device)

        with torch.no_grad():
            output: MoeCausalOutputWithPast = self.model(
                input_ids=model_forward_kwargs["input_ids"],
                labels=model_forward_kwargs["labels"],
                sequence_ids=model_forward_kwargs["sequence_ids"],
                position_ids=model_forward_kwargs["position_ids"],
                return_dict=True,
            )

        labels = model_forward_kwargs["labels"]
        pad_id = self.model.config.pad_token_id
        target_mask = labels != pad_id

        targets = labels[..., 1:].contiguous()
        target_mask = target_mask[..., 1:].contiguous()
        logits = output.logits[..., :-1, :].contiguous().to(torch.float32)

        flat_logits = logits.view(-1, logits.shape[-1])
        nll = nn.functional.cross_entropy(
            flat_logits, targets.view(-1), reduction="none"
        ).view(targets.shape)

        nll = (nll * target_mask.to(nll)).sum(dim=1)

        if self.reduction == "mean":
            nll = nll / target_mask.sum(dim=1)

        return nll

    def score_sequences(self, sequences: List[str], batch_size: int = 8) -> List[float]:
        """
        对一组序列进行打分。
        ProGen3 策略：(Forward LL + Reverse LL) / 2
        """

        scores = []

        for i in tqdm(
            range(0, len(sequences), batch_size), leave=False, desc="Batch Scoring"
        ):
            batch_seqs = sequences[i : i + batch_size]

            # 1. Forward Pass
            kwargs_fwd = self.batch_preparer.get_batch_kwargs(
                batch_seqs, device=self.device, reverse=False
            )
            nll_fwd = self._compute_nll(kwargs_fwd)

            # 2. Reverse Pass
            kwargs_rev = self.batch_preparer.get_batch_kwargs(
                batch_seqs, device=self.device, reverse=True
            )
            nll_rev = self._compute_nll(kwargs_rev)

            # 3. Combine
            # Log Likelihood = -NLL
            # Score = (LL_fwd + LL_rev) / 2
            #       = (-NLL_fwd + -NLL_rev) / 2
            #       = -(NLL_fwd + NLL_rev) / 2
            batch_scores = -(nll_fwd + nll_rev) / 2

            scores.extend(batch_scores.cpu().tolist())

        return scores


@dataclass(frozen=True)
class CacheStats:
    hit: int = 0
    miss: int = 0


class ProGen3Engine:
    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        *,
        device: Optional[str] = None,
        use_fp16_infer: bool = False,
        reduction: str = "mean",
        cache_mutants: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16_infer = bool(
            use_fp16_infer and str(self.device).startswith("cuda")
        )
        self.reduction = reduction
        self.cache_mutants = bool(cache_mutants)
        self.cache_dir = cache_dir
        self.fitness_dir = os.path.join(cache_dir, "fitness")
        os.makedirs(self.fitness_dir, exist_ok=True)

        self.scorer = ProGen3ScorerInternal(
            model_path=model_path,
            device=self.device,
            fp16=self.use_fp16_infer,
            reduction=self.reduction,
        )

        self._mem_cache: Dict[str, float] = {}

    def _cache_path(self, seq: str) -> str:
        key = sha256_upper(seq)
        return os.path.join(self.fitness_dir, f"{key}.pt")

    def get_fitness_many(self, seqs: List[str]) -> Tuple[Dict[str, float], CacheStats]:
        hit = 0
        miss = 0

        # ProGen3 只需要纯序列，不需要像 ProGen2 那样加 terminals
        sanitized = [sanitize_sequence(s) for s in seqs]

        # 去重，保留映射关系
        uniq = list(dict.fromkeys(sanitized))

        out: Dict[str, float] = {}
        to_compute: List[str] = []

        # 检查缓存
        for s in uniq:
            if s in self._mem_cache:
                out[s] = float(self._mem_cache[s])
                hit += 1
                continue

            path = self._cache_path(s)
            if os.path.exists(path):
                try:
                    val = torch.load(path, map_location="cpu", weights_only=False)
                    score = float(val.item()) if torch.is_tensor(val) else float(val)
                    out[s] = score
                    self._mem_cache[s] = score
                    hit += 1
                except Exception:
                    to_compute.append(s)
                    miss += 1
            else:
                to_compute.append(s)
                miss += 1

        if to_compute:
            # 调用 ProGen3 打分
            scores_list = self.scorer.score_sequences(to_compute, batch_size=4)

            for s, sc in zip(to_compute, scores_list):
                sc_f = float(sc)
                out[s] = sc_f
                self._mem_cache[s] = sc_f
                # 写入缓存
                torch.save(torch.tensor(sc_f, dtype=torch.float32), self._cache_path(s))

        return out, CacheStats(hit=hit, miss=miss)

    def score_group_delta(
        self,
        wt_seq: str,
        mutant_seqs: List[str],
    ) -> List[float]:
        """
        计算 delta = fitness(mut) - fitness(wt)
        """
        wt_clean = sanitize_sequence(wt_seq)
        muts_clean = [sanitize_sequence(s) for s in mutant_seqs]

        # 准备所有需要分数的序列
        all_seqs = [wt_clean]
        if self.cache_mutants:
            all_seqs.extend(muts_clean)

        score_map, _stats = self.get_fitness_many(all_seqs)

        wt_score = float(score_map[wt_clean])

        # 如果不缓存 mutants (optionally)，这里单独计算不存盘
        # 但为了简化逻辑，建议总是走 get_fitness_many，除非显存极度受限
        # 这里沿用 ProGen2 脚本逻辑：如果 cache_mutants 为 False，则不从 map 读取?
        # 其实 get_fitness_many 已经处理了计算逻辑。
        # 如果 strict 不缓存 mutant 到磁盘，可以在 get_fitness_many 内部修改，
        # 但通常缓存是有益的。

        deltas = []
        for s in muts_clean:
            if s in score_map:
                mut_score = score_map[s]
            else:
                temp_scores = self.scorer.score_sequences([s], batch_size=1)
                mut_score = temp_scores[0]

            deltas.append(float(mut_score - wt_score))

        return deltas


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to ProGen3 checkpoint directory (containing config.json, model.safetensors, etc.)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional explicit model id used for naming cache",
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV or TSV file (must contain POI column)",
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
        help="1=drop silent chains, 0=keep all chains (passed to DMS_file_for_LLM)",
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
        "--reduction",
        default="mean",
        choices=["sum", "mean"],
        help="Fitness reduction mode (official ProGen3 uses mean usually)",
    )
    parser.add_argument(
        "--cache-mutants",
        action="store_true",
        help="Cache mutant fitness on disk",
    )

    args = parser.parse_args()

    focus_flag = bool(args.focus == 1)
    effective_fp16 = True

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

    # 初始化引擎
    engine = ProGen3Engine(
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

    # 按照 POI 分组处理
    for _poi, g in tqdm(df.groupby("POI"), desc="Scoring POI groups"):
        # 调用 data_utils 处理数据 (保留原有逻辑)
        g2 = DMS_file_for_LLM(g, focus=focus_flag)

        if g2 is None or len(g2) == 0:
            for row_idx in g.index.tolist():
                final_scores[row_idx] = 0.0
            continue

        if (
            "wildtype_sequence" not in g2.columns
            or "mutated_sequence" not in g2.columns
        ):
            raise ValueError(
                "DMS_file_for_LLM output must contain columns: wildtype_sequence, mutated_sequence"
            )

        # 获取序列
        wt_seq = str(g2["wildtype_sequence"].values[0])
        if not wt_seq or str(wt_seq).strip() == "":
            for row_idx in g2.index.tolist():
                final_scores[row_idx] = 0.0
            continue

        mut_seqs = [str(s) for s in g2["mutated_sequence"].tolist()]

        # 计算 Delta
        deltas = engine.score_group_delta(
            wt_seq=wt_seq,
            mutant_seqs=mut_seqs,
        )

        # 填回分数
        for row_idx, sc in zip(g2.index.tolist(), deltas):
            final_scores[row_idx] = float(sc)

    # 保存结果
    col_name = f"progen3_{model_id_slug}_{str(args.reduction)}_score"
    df[col_name] = final_scores

    df.to_csv(output_path, index=False, sep=sep)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
