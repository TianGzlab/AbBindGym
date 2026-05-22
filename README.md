# AbBindGym

AbBindGym is the code release for antibody-antigen protein language model
benchmarking and ProGen2-based AbELA design experiments.

It contains three independent workflows:

- `zero_shot/`: mutation-level zero-shot scoring for BindingGYM, AB-Bind,
  AbDesign and SKEMPI.
- `supervised/`: frozen PLM embedding plus regression benchmarks across
  supervised antibody-antigen datasets.
- `progen2_AbELA/`: ProGen2-OAS LoRA fine-tuning and epitope-conditioned
  antibody sequence design for AbELA-style records.

Code repository: https://github.com/TianGzlab/AbBindGym

Benchmark data archive: https://doi.org/10.5281/zenodo.19911284

For data licensing, controlled-access AbELA-Q terms and dataset-use boundaries,
see [`docs/data_availability.md`](docs/data_availability.md).

## Layout

```text
zero_shot/       zero-shot mutation scoring and metric aggregation
supervised/      frozen-backbone PLM regression benchmarks
progen2_AbELA/   ProGen2-OAS LoRA fine-tuning and CDR-H design
examples/        synthetic smoke-test inputs and a no-download demo
results/         public-safe aggregate result tables
tests/           lightweight tests
docs/            data-boundary and reproduction notes
```

This repository does not include raw datasets, processed benchmark input tables,
controlled-access AbELA-Q records, model checkpoints, per-record predictions, or
generated design candidates. See `docs/data_availability.md`.

## System Requirements

### Software

- Python 3.10-3.12; tested with Python 3.12.13.
- PyTorch >= 2.0. Use a CUDA build for GPU workflows.
- Transformers >= 4.38 and < 5.
- NumPy >= 1.26, pandas >= 2.0, scikit-learn >= 1.3, SciPy >= 1.10,
  tokenizers >= 0.15 and < 0.22.
- Optional model-zoo runners require additional packages listed under
  `[project.optional-dependencies].model-zoo` in `pyproject.toml`.
- Structure-aware runners may require Foldseek and external structure-tokenizer
  assets outside the Python environment.

### Operating Systems Tested

- Ubuntu 22.04 LTS on x86_64 Linux.

The synthetic demo and tests should also run on standard CPU-only Python
installations. Full benchmark runs were designed for Linux GPU workstations or
HPC nodes with local model caches.

### Hardware

- No-download demo and unit tests: CPU is sufficient.
- Single-model zero-shot or supervised runs: NVIDIA GPU recommended; memory
  needs depend on the selected PLM. A GPU with at least 24-40 GB VRAM is
  recommended for 650M-scale models.
- Full benchmark and ProGen2/large-model workflows: tested on NVIDIA A800
  80 GB GPUs; multi-model runs are best scheduled on a GPU cluster.

## Installation

Recommended setup with `uv`:

```bash
git clone https://github.com/TianGzlab/AbBindGym
cd AbBindGym
uv sync --python 3.12 --group dev
source .venv/bin/activate
```

Pip editable install is also supported:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e . pytest
```

For model-family-specific zero-shot runners, install the optional model-zoo
dependencies:

```bash
uv sync --extra model-zoo --group dev
# or
python -m pip install -e ".[model-zoo]" pytest
```

Typical install time: about 5-10 minutes for the base environment on a standard
Linux workstation with broadband; about 10-20 minutes with `model-zoo` extras,
excluding model checkpoint downloads.

## Demo: No External Data Required

Run the self-contained synthetic zero-shot demo:

```bash
python examples/run_demo.py --output examples/demo_output
```

Expected output: `examples/demo_output/scores.csv` with five synthetic mutation
rows and `examples/demo_output/metrics.csv` with Spearman, AUC, MCC, NDCG and
AP values. A reference score table is provided in
`examples/expected_output/demo_scores.csv`.

Expected runtime: less than 1 minute on CPU.

## Data

Benchmark inputs are distributed through the Zenodo archive:
https://doi.org/10.5281/zenodo.19911284

Raw public datasets should be obtained from their original sources and cited
according to the source terms.

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

Expected output: per-mutation CSV files under
`results/zero_shot/model_outputs/<model>/<dataset>/` and aggregate metric CSVs
under `results/zero_shot/final_metrics/`.

Expected runtime: model- and dataset-dependent; small model-family runs are
typically minutes to hours on a single NVIDIA A800 80 GB GPU. CPU execution is
not recommended for full model runners.

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

Expected output: fold-level metrics, predictions and checkpoints under
`results/supervised/SabDab/`.

Expected runtime: hours per dataset/model combination on an NVIDIA A800 80 GB
GPU, depending on model size, sequence lengths and cache state.

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

Expected output: a small LoRA smoke-test run under
`results/progen2_AbELA/abela_smoke/`.

Expected runtime: minutes on a single NVIDIA GPU after ProGen2-OAS weights are
available locally. Runtime is dominated by checkpoint loading.

Workflow-specific notes are in `zero_shot/README.md`,
`supervised/README.md`, and `progen2_AbELA/README.md`.

## Reproducing Manuscript Results

See [`docs/reproduction.md`](docs/reproduction.md) for workflow-level inputs,
commands, expected outputs and hardware requirements for reproducing manuscript
analyses from the archived data.

## Validate

```bash
python examples/run_demo.py --output examples/demo_output
python -m compileall supervised zero_shot progen2_AbELA examples
python -m pytest -q
```

## Citation and License

AbBindGym source code is released under the MIT License. Dataset availability
and use boundaries are summarized in `docs/data_availability.md`.

Please cite the associated manuscript and this software release.
