# ProGen2 AbELA

This workflow fine-tunes ProGen2-OAS with LoRA adapters and uses the adapted
model for epitope-conditioned CDR-H design.

Serialized input format:

```text
1{epitope}GGGGSGGGGSGGGGS{VH}GGGGSGGGGSGGGGS{VL}2
```

`examples/abela_minimal.csv` is synthetic and documents the expected table
layout. Do not commit real AbELA-Q rows, sequences, labels, counts, targets, or
derived statistics.

## Expected Columns

The loader accepts CSV, TSV, JSON, or JSONL and auto-detects common column
names. For AbELA-style CSVs, use:

```text
ID, ELISA_status, H, L, HCDR1, HCDR2, HCDR3, Antigen, Antigen_sequence
```

Private logging is enabled by default and redacts dataset counts, sequence
statistics, scores, mutations, and target names.

## Smoke Test

```bash
python progen2_AbELA/train_abela_lora.py \
  --dataset examples/abela_minimal.csv \
  --checkpoint-dir <progen2-oas-checkpoint> \
  --tokenizer progen2_AbELA/tokenizer.json \
  --output-dir results/progen2_AbELA/abela_smoke \
  --active-only false \
  --private-log true \
  --device auto \
  --fp16 false \
  --epochs 1 \
  --limit-records 2
```

Expected output: a small LoRA smoke-test run under
`results/progen2_AbELA/abela_smoke/`.

Expected runtime: minutes on a single NVIDIA GPU after ProGen2-OAS weights are
available locally. Runtime is dominated by checkpoint loading.

## Fine-Tuning

```bash
python progen2_AbELA/train_abela_lora.py \
  --dataset <abela.csv> \
  --checkpoint-dir <progen2-oas-checkpoint> \
  --tokenizer progen2_AbELA/tokenizer.json \
  --output-dir results/progen2_AbELA/adapter \
  --active-only false \
  --private-log true \
  --epochs 20 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --learning-rate 1e-4
```

Adapters, split indices, and logs from real AbELA-Q runs should remain outside
public commits unless they have been explicitly cleared for release.

## CDR-H Design

```bash
python progen2_AbELA/optimize_abela_cdrh.py \
  --dataset <abela.csv> \
  --checkpoint-dir <progen2-oas-checkpoint> \
  --tokenizer progen2_AbELA/tokenizer.json \
  --adapter-dir results/progen2_AbELA/adapter/best \
  --target <target-name> \
  --output-csv private_outputs/candidates.csv \
  --af3-json-dir private_outputs/af3_inputs
```

Candidate CSVs and AF3 JSON files contain designed protein sequences and should
be kept in private output locations.

## AF3 Ranking

```bash
python progen2_AbELA/rank_af3_candidates.py \
  --input-dir private_outputs/af3_outputs \
  --output-csv private_outputs/af3_ranked.csv \
  --top-n 20
```

Original ProGen2 code and models are BSD-3 licensed; see `LICENSE.txt`.
