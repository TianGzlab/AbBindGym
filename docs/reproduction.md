# Reproducing Manuscript Results

Full manuscript reproduction requires the external AbBindGym data archive,
model checkpoints or Hugging Face model caches, and a Linux GPU workstation or
HPC node. The public GitHub repository contains code, synthetic examples, tests,
and public-safe aggregate result tables only.

Data archive: https://doi.org/10.5281/zenodo.19911284

## Quick Code Check

Run this first after installation. It does not use external data or model
weights.

```bash
python examples/run_demo.py --output examples/demo_output
python -m pytest -q
```

Expected output: `examples/demo_output/scores.csv` with five synthetic
mutation rows and `examples/demo_output/metrics.csv` with Spearman, AUC, MCC,
NDCG and AP values. A reference score table is provided in
`examples/expected_output/demo_scores.csv`.

Expected runtime: less than 1 minute on CPU.

## Zero-Shot Mutation Scoring

The zero-shot scripts expect the archived benchmark layout under
`<data-archive>/zero_shot`.

```bash
DATA_ROOT=<data-archive>/zero_shot \
OUTPUT_ROOT=results/zero_shot/model_outputs \
CACHE_ROOT=results/zero_shot/logits_cache \
PYTHON_BIN=python \
DEVICE=cuda \
bash zero_shot/modelzoo/IgBert/run_IgBert_fitness.sh
```

Aggregate model outputs:

```bash
python zero_shot/run/calc_metric.py \
  results/zero_shot/model_outputs/IgBert \
  --output-dir results/zero_shot/final_metrics \
  --label IgBert
```

Inputs: archived zero-shot mutation tables, optional structure files for
structure-aware models, and local model weights or Hugging Face caches.

Expected output: per-mutation CSV files under
`results/zero_shot/model_outputs/<model>/<dataset>/` and aggregate metrics under
`results/zero_shot/final_metrics/`.

Expected runtime: model- and dataset-dependent; small model-family runs are
typically minutes to hours on a single NVIDIA A800 80 GB GPU, while the full
multi-model benchmark should be run on a GPU cluster.

## Supervised Frozen-Backbone Regression

Use the archived supervised tables and split files. Example for SabDab:

```bash
python supervised/common/train.py \
  --data-path <data-archive>/supervised/clustered_benchmarks/SabDab/csv/SabDab_with_clusters.csv \
  --splits-file <data-archive>/supervised/clustered_benchmarks/SabDab/splits/Dunbar2014_SabDab_seqcluster_k5_seed314.json \
  --results-root results/supervised/SabDab \
  --model-name esm2_t33_650m \
  --net-name facebook/esm2_t33_650M_UR50D \
  --target pkd \
  --folds 5
```

Expected output: fold-level metrics, predictions and checkpoints under
`results/supervised/SabDab/`.

Expected runtime: hours per dataset/model combination on an NVIDIA A800 80 GB
GPU, depending on model size, sequence lengths and cache state.

## AbELA ProGen2 Workflow

The synthetic loader smoke test is public, but real AbELA-Q sequence-level
records and generated candidates are controlled-access and are not included in
this repository.

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

## Matching Manuscript Tables

Public-safe aggregate zero-shot result tables already included under
`results/zero_shot/final_metrics/` are the lightweight reference artifacts for
checking metric aggregation. Regenerating all manuscript tables requires the
external data archive and the same model checkpoints used for the submitted
analysis.
