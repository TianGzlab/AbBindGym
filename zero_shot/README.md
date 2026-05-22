# Zero-Shot Evaluation

This directory contains mutation-level zero-shot scoring and metric aggregation.

Expected input layout:

```text
data/zero_shot/<dataset>/Binding_substitutions_DMS/
data/zero_shot/<dataset>/structures/        # optional for structure-aware models
```

Supported dataset roots are `BindingGYM`, `ABBind`, `AbDesign`, and `SKEMPI`.

Generated outputs should stay outside public commits unless they are aggregate
summary tables:

```text
results/zero_shot/model_outputs/
results/zero_shot/logits_cache/
results/zero_shot/final_metrics/
```

The public aggregate result tables may include baselines whose runners are not
bundled here.

## Launchers

Implemented launchers in this release:

- `AIDO/run_AIDO_fitness.sh`
- `IgBert/run_IgBert_fitness.sh`
- `IgT5/run_IgT5_fitness.sh`
- `MAGE/run_mage_fitness.sh`
- `VenusPLM/run_venus_fitness.sh`
- `ablang2/run_ablang2_fitness.sh`
- `ankh/run_ankh_fitness.sh`
- `antiberty/run_antiberty_fitness.sh`
- `esm/run_esm_fitness.sh`
- `esm3/run_esm3_esmc_fitness.sh`
- `progen2/run_progen2_fitness.sh`
- `progen3/run_progen3_fitness.sh`
- `prosst/run_prosst_fitness.sh`
- `proteinglm/run_proteinglm_fitness.sh`
- `protgpt2/run_protgpt2_fitness.sh`
- `saprot/run_saprot_fitness.sh`

Some launchers require external model packages, structure files, Foldseek, or
local model caches beyond the base Python environment.

Run from the repository root:

```bash
DATA_ROOT=<data-archive>/zero_shot \
OUTPUT_ROOT=results/zero_shot/model_outputs \
CACHE_ROOT=results/zero_shot/logits_cache \
bash zero_shot/modelzoo/IgBert/run_IgBert_fitness.sh
```

For AIDO, keep the structure-tokenizer codebook outside this repository and set
`AIDO_CODEBOOK_PATH`.

Aggregate generated outputs:

```bash
python zero_shot/run/calc_metric.py <model-output-dir> \
  --output-dir results/zero_shot/final_metrics \
  --label <model-label>
```
