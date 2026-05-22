# AbBindGym

AbBindGym is the code release for antibody-antigen PLM benchmarking and
ProGen2-based AbELA design experiments.

## Layout

```text
zero_shot/       zero-shot mutation scoring and metric aggregation
supervised/      frozen-backbone PLM regression benchmarks
progen2_AbELA/   ProGen2-OAS LoRA fine-tuning and CDR-H design
examples/        synthetic smoke-test inputs
results/         public-safe aggregate result tables
tests/           lightweight tests
```

This repository does not include raw datasets, processed benchmark input tables,
controlled-access AbELA-Q records, model checkpoints, per-record predictions, or
generated design candidates. See `docs/data_availability.md`.

## Install

Use Python 3.10-3.12.

```bash
uv sync --python 3.12 --group dev
source .venv/bin/activate
```

For model-family-specific zero-shot runners:

```bash
uv sync --extra model-zoo --group dev
```

## Data

Benchmark inputs are distributed through the data archive described in the
manuscript. Raw public datasets should be obtained from their original sources
and cited according to the source terms.

The GitHub repository keeps only code, synthetic examples, tests, and
public-safe aggregate result tables.

Some zero-shot baselines require external model packages, structure files,
Foldseek, or local model caches beyond the base Python environment.

## Main Workflows

Zero-shot scoring:

```bash
DATA_ROOT=<data-archive>/zero_shot \
OUTPUT_ROOT=results/zero_shot/model_outputs \
CACHE_ROOT=results/zero_shot/logits_cache \
bash zero_shot/modelzoo/IgBert/run_IgBert_fitness.sh
```

Supervised regression:

```bash
python supervised/common/train.py \
  --data-path <supervised.csv> \
  --splits-file <splits.json> \
  --results-root results/supervised/SabDab \
  --model-name esm2_t33_650m \
  --net-name facebook/esm2_t33_650M_UR50D \
  --target pkd \
  --folds 5
```

AbELA ProGen2 smoke test:

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

Workflow-specific notes are in `zero_shot/README.md`,
`supervised/README.md`, and `progen2_AbELA/README.md`.

## Validate

```bash
python -m compileall supervised zero_shot progen2_AbELA
python -m pytest -q
```

## Citation and License

AbBindGym source code is released under the MIT License. Dataset availability
and use boundaries are summarized in `docs/data_availability.md`.

Please cite the associated manuscript and this software release.
