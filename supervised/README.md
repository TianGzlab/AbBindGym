# Supervised Evaluation

This workflow trains a regression head on frozen PLM embeddings from antibody
heavy chain, light chain, and antigen sequences.

The repository contains code only. Place processed benchmark tables and split
JSON files under `data/supervised/` or pass explicit paths.

## Main Entry Points

| Path | Purpose |
|---|---|
| `common/train.py` | Main frozen-backbone PLM benchmark. |
| `common/base_runner.py` | Antibody-language-model and lightweight baselines. |
| `common/evaluate_special_models.py` | Re-evaluate saved special-model checkpoints. |
| `common/greedy_balanced_kfold.py` | Shared split helper. |

Dataset-specific scripts are kept under `ABBind/`, `AbCoV/`, `AbDesign/`,
`AbELA/`, `AlphaSeq/`, `BindingGYM/`, `HER2/`, `PPB`, `SabDab`, and `SKEMPI`.
`AbELA/` is for controlled-access AbELA-Q inputs.

## Preprocessing

Available preprocessing entrypoints:

```bash
python supervised/ABBind/prepare_ABBind_split_by_cluster.py --stage all
python supervised/AbCoV/prepare_AbCoV_splits.py --stage all
python supervised/AbDesign/prepare_abdesign_elisa_splits.py
python supervised/AbELA/prepare_AbELA_split_by_cluster.py --stage all
python supervised/AlphaSeq/prepare_alphaseq_assets.py
python supervised/BindingGYM/prepare_bindinggym_assets.py
python supervised/HER2/prepare_HER2_split_by_cluster.py
python supervised/PPB/prepare_PPB_split_by_cluster.py --stage all
python supervised/SKEMPI/prepare_SKEMPI_split_by_cluster.py --stage all
```

SabDab training scripts are included, but this public release does not include a
SabDab preprocessing script. Use the processed SabDab files from the data
archive or pass externally prepared SabDab tables and splits.

## Example

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

Saved checkpoints, predictions, plots, and per-fold outputs should remain
outside public commits unless explicitly released as aggregate summaries.

Expected output: fold-level metrics, predictions and checkpoints under the
selected `--results-root`.

Expected runtime: hours per dataset/model combination on an NVIDIA A800 80 GB
GPU, depending on model size, sequence lengths and cache state. CPU execution is
only suitable for very small debugging runs.
