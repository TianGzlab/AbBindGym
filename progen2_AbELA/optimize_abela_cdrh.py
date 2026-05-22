import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from progen.abela import (
    STANDARD_AMINO_ACIDS,
    cdr_positions,
    format_mutation,
    load_abela_records,
    serialize_antibody,
)
from progen.lora import apply_qv_lora, load_lora_adapter
from progen.runtime import autocast_for, load_progen_model, select_device, str_to_bool
from progen.tokenization import load_tokenizer


FIRST_AA_TOKEN = 5
LAST_AA_TOKEN = 29
TERMINAL_TOKENS = {3, 4}


@dataclass(frozen=True)
class Candidate:
    seed_id: str
    target: str
    epitope: str
    vh: str
    vl: str
    hcdr_ranges: tuple
    mutation_positions: tuple
    mutations: tuple
    score: float = float("-inf")

    @property
    def serialized(self):
        return serialize_antibody(self.epitope, self.vh, self.vl)

    def with_score(self, score):
        return Candidate(
            seed_id=self.seed_id,
            target=self.target,
            epitope=self.epitope,
            vh=self.vh,
            vl=self.vl,
            hcdr_ranges=self.hcdr_ranges,
            mutation_positions=self.mutation_positions,
            mutations=self.mutations,
            score=float(score),
        )


def load_model(args, device):
    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required")
    checkpoint_dir = args.checkpoint_dir
    model = load_progen_model(checkpoint_dir, args.fp16)
    if args.adapter_dir:
        adapter_config = Path(args.adapter_dir) / "adapter_config.json"
        if adapter_config.exists():
            with adapter_config.open() as handle:
                config = json.load(handle)
            args.lora_rank = int(config.get("lora_rank", args.lora_rank))
            args.lora_alpha = float(config.get("lora_alpha", args.lora_alpha))
            args.lora_dropout = float(config.get("lora_dropout", args.lora_dropout))
        apply_qv_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
        load_lora_adapter(model, args.adapter_dir)
    model.to(device)
    model.eval()
    return model


def score_serialized_batch(model, tokenizer, serialized, device, batch_size=4, fp16=True):
    scores = []
    for start in range(0, len(serialized), batch_size):
        chunk = serialized[start : start + batch_size]
        encoded = [torch.tensor(tokenizer.encode(item).ids, dtype=torch.long) for item in chunk]
        max_len = max(ids.numel() for ids in encoded)
        input_ids = torch.zeros((len(encoded), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        lengths = []
        for row, ids in enumerate(encoded):
            length = ids.numel()
            input_ids[row, :length] = ids
            attention_mask[row, :length] = 1
            lengths.append(length)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            with autocast_for(device, fp16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        for row, length in enumerate(lengths):
            row_logits = logits[row, : length - 1, FIRST_AA_TOKEN : LAST_AA_TOKEN + 1]
            target = input_ids[row, 1:length]
            if int(target[-1].item()) in TERMINAL_TOKENS:
                row_logits = row_logits[:-1]
                target = target[:-1]
            if torch.any((target < FIRST_AA_TOKEN) | (target > LAST_AA_TOKEN)):
                bad = sorted(set(target.detach().cpu().tolist()) - set(range(FIRST_AA_TOKEN, LAST_AA_TOKEN + 1)))
                raise ValueError(f"encountered non-amino-acid token(s) inside score region: {bad}")
            target = target - FIRST_AA_TOKEN
            log_probs = F.log_softmax(row_logits, dim=-1)
            score = log_probs.gather(1, target.unsqueeze(1)).sum().item()
            scores.append(score)
    return scores


def mutate_parent(parent, index_base):
    positions = cdr_positions(parent.hcdr_ranges)
    mutated = set(parent.mutation_positions)
    for position in positions:
        if position in mutated:
            continue
        original = parent.vh[position]
        for amino_acid in STANDARD_AMINO_ACIDS:
            if amino_acid == original:
                continue
            vh = parent.vh[:position] + amino_acid + parent.vh[position + 1 :]
            yield Candidate(
                seed_id=parent.seed_id,
                target=parent.target,
                epitope=parent.epitope,
                vh=vh,
                vl=parent.vl,
                hcdr_ranges=parent.hcdr_ranges,
                mutation_positions=parent.mutation_positions + (position,),
                mutations=parent.mutations + (format_mutation(position, original, amino_acid, index_base=index_base),),
            )


def beam_optimize(records, model, tokenizer, args, device):
    parents = []
    for record in records:
        if not record.hcdr_ranges:
            raise ValueError("at least one record has no HCDR ranges; cannot run CDR-H mutagenesis")
        parents.append(
            Candidate(
                seed_id=record.record_id,
                target=record.target,
                epitope=record.epitope,
                vh=record.vh,
                vl=record.vl,
                hcdr_ranges=record.hcdr_ranges,
                mutation_positions=(),
                mutations=(),
            )
        )

    all_scored = {}
    for round_index in range(1, args.rounds + 1):
        generated = {}
        for parent in parents:
            for child in mutate_parent(parent, index_base=args.cdr_index_base):
                generated.setdefault(child.serialized, child)

        candidates = list(generated.values())
        if not candidates:
            break

        scores = score_serialized_batch(
            model,
            tokenizer,
            [candidate.serialized for candidate in candidates],
            device=device,
            batch_size=args.score_batch_size,
            fp16=args.fp16,
        )
        scored = [candidate.with_score(score) for candidate, score in zip(candidates, scores)]
        scored.sort(key=lambda item: item.score, reverse=True)

        for candidate in scored:
            all_scored.setdefault(candidate.serialized, candidate)

        parents = scored[: args.beam_size]
        if args.private_log:
            print(f"round={round_index} completed")
        else:
            print(
                f"round={round_index} generated={len(candidates)} "
                f"best_score={parents[0].score:.4f} retained={len(parents)}"
            )

    ranked = sorted(all_scored.values(), key=lambda item: item.score, reverse=True)
    return ranked[: args.final_size]


def source_aliases(candidates):
    aliases = {}
    for candidate in candidates:
        if candidate.seed_id not in aliases:
            aliases[candidate.seed_id] = f"seed_{len(aliases) + 1:05d}"
    return aliases


def write_candidates(candidates, output_csv, redact_source_metadata=True):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aliases = source_aliases(candidates) if redact_source_metadata else {}
    fieldnames = [
        "rank",
        "seed_id",
        "target",
        "mutation_count",
        "mutations",
        "log_likelihood",
        "epitope",
        "vh",
        "vl",
        "serialized",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            seed_id = aliases.get(candidate.seed_id, candidate.seed_id)
            target = "redacted" if redact_source_metadata else candidate.target
            writer.writerow(
                {
                    "rank": rank,
                    "seed_id": seed_id,
                    "target": target,
                    "mutation_count": len(candidate.mutations),
                    "mutations": ";".join(candidate.mutations),
                    "log_likelihood": f"{candidate.score:.8f}",
                    "epitope": candidate.epitope,
                    "vh": candidate.vh,
                    "vl": candidate.vl,
                    "serialized": candidate.serialized,
                }
            )


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_af3_inputs(candidates, output_dir, seeds, samples_per_seed, redact_source_metadata=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_seeds = list(range(1, seeds + 1))
    for rank, candidate in enumerate(candidates, start=1):
        if redact_source_metadata:
            name = f"candidate_{rank:03d}"
        else:
            name = safe_name(f"candidate_{rank:03d}_{candidate.seed_id}_{'_'.join(candidate.mutations)}")
        payload = {
            "name": name,
            "modelSeeds": model_seeds,
            "sequences": [
                {"protein": {"id": "A", "sequence": candidate.epitope}},
                {"protein": {"id": "H", "sequence": candidate.vh}},
                {"protein": {"id": "L", "sequence": candidate.vl}},
            ],
            "dialect": "alphafold3",
            "version": 1,
        }
        with (output_dir / f"{name}.json").open("w") as handle:
            json.dump(payload, handle, indent=2)

    if samples_per_seed != 5:
        print("note: AlphaFold 3 sample count is controlled by the AF3 runner, not the input JSON")


def main():
    parser = argparse.ArgumentParser(description="Likelihood-guided CDR-H optimization for AbELA/ProGen2.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="progen2-oas")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--adapter-dir", default=None, help="Directory containing adapter_model.bin from train_abela_lora.py")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--epitope-column", default=None)
    parser.add_argument("--vh-column", default=None)
    parser.add_argument("--vl-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--active-column", default=None)
    parser.add_argument("--active-only", type=str_to_bool, default=True)
    parser.add_argument("--active-values", default="AbELA-Q,active,positive,binder,true,1,yes")
    parser.add_argument("--ec50-column", default=None)
    parser.add_argument("--max-ec50", type=float, default=None)
    parser.add_argument("--cdr-index-base", type=int, default=1)
    parser.add_argument("--skip-invalid-records", type=str_to_bool, default=True)
    parser.add_argument(
        "--private-log",
        type=str_to_bool,
        default=True,
        help="Redact generated counts, scores, mutations, and target names from console logs.",
    )
    parser.add_argument(
        "--redact-source-metadata",
        type=str_to_bool,
        default=True,
        help="Anonymize seed ids and target names in candidate CSVs and AF3 JSON names.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--final-size", type=int, default=200)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--fp16", type=str_to_bool, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--af3-json-dir", default=None)
    parser.add_argument("--af3-seeds", type=int, default=10)
    parser.add_argument("--af3-samples-per-seed", type=int, default=5)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = select_device(args.device)
    if device.type != "cuda":
        args.fp16 = False

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args, device)

    records = load_abela_records(
        args.dataset,
        epitope_column=args.epitope_column,
        vh_column=args.vh_column,
        vl_column=args.vl_column,
        id_column=args.id_column,
        target_column=args.target_column,
        active_column=args.active_column,
        active_only=args.active_only,
        active_values=args.active_values,
        ec50_column=args.ec50_column,
        max_ec50=args.max_ec50,
        cdr_index_base=args.cdr_index_base,
        skip_invalid=args.skip_invalid_records,
    )
    if args.target:
        records = [record for record in records if record.target.upper() == args.target.upper()]
    if not records:
        if args.target:
            if args.private_log:
                raise ValueError("no records matched the requested target")
            raise ValueError(f"no records matched target={args.target!r}")
        raise ValueError("no valid records were loaded")

    candidates = beam_optimize(records, model, tokenizer, args, device)
    if not candidates:
        raise ValueError("optimization produced no candidates")

    write_candidates(candidates, args.output_csv, redact_source_metadata=args.redact_source_metadata)
    if args.af3_json_dir:
        write_af3_inputs(
            candidates,
            args.af3_json_dir,
            args.af3_seeds,
            args.af3_samples_per_seed,
            redact_source_metadata=args.redact_source_metadata,
        )

    if args.private_log:
        print("design_complete")
    else:
        best = candidates[0]
        print(
            f"wrote={len(candidates)} output_csv={args.output_csv} "
            f"best_score={best.score:.4f} best_mutations={';'.join(best.mutations)}"
        )


if __name__ == "__main__":
    main()
