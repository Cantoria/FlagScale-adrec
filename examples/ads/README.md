# Advertising (CTR) model training with FlagScale

Native FlagScale training for the advertising models of the FlagOS
localization project.  Each model reproduces an implementation with a
published, downloadable benchmark result; see [EVIDENCE.md](EVIDENCE.md).

| Config | Model | Data | Reference |
| --- | --- | --- | --- |
| `din` | DIN (FuxiCTR v2.0.1) | BARS AmazonElectronics_x1 | BARS test gAUC 0.8835 / AUC 0.8860 |
| `deepfm` | DeepFM (FuxiCTR v1.1.0) | BARS Criteo_x1 | BARS test AUC 0.8137 / logloss 0.4381 |
| `dcn_mix` | DCN-V2 + mixture of low-rank experts (FuxiCTR v1.1.0) | BARS Criteo_x1 | BARS DCN-V2 AUC 0.8142; paper Mix vs V2 -0.0003 |

Layout (independent of the retrieval and recommendation modules):

* `flagscale/models/ads/`: `din.py`, `deepfm.py`, `dcn_mix.py`, shared layers
  (Dice, MLPBlock, FieldEmbedding, CrossNetMix), registry;
* `flagscale/train/datasets/ads.py`, `losses/ads.py`, `metrics/ads.py`;
* `flagscale/train/train_ads.py`: FuxiCTR training procedure (embedding L2,
  gradient clipping, reduce-LR-on-plateau, early stopping, best checkpoint);
* data under `/flagos-ads/datasets`, results under `/flagos-ads/ckpts`.

## Data

```bash
python examples/ads/prepare_amazon_electronics_x1.py   # -> /flagos-ads/datasets/amazonelectronics_x1
python examples/ads/prepare_criteo_x1.py               # -> /flagos-ads/datasets/criteo_x1
```

Raw inputs are the reczoo Hugging Face datasets under
`/flagos-ads/datasets/raw/{AmazonElectronics_x1,Criteo_x1}` (md5 values in
EVIDENCE.md).

## Run

```bash
export PYTHONPATH=$PWD FLAGSCALE_HOME=$PWD WANDB_MODE=offline
# Torch-FL
FLAGSCALE_PYTHON_ENV=/flagos-search-codes/torch-fl-cuda/.venv \
python -m flagscale.run --config-path examples/ads/conf --config-name train action=test train=din \
  train.hardware.device_type=flagos train.hardware.distributed_backend=flagos
# native CUDA baseline
FLAGSCALE_PYTHON_ENV=/opt/conda \
python -m flagscale.run --config-path examples/ads/conf --config-name train action=test train=din \
  train.hardware.device_type=cuda
```

Outputs: `/flagos-ads/ckpts/<model>-<device>/checkpoint/{model.pt,config.json,metrics.json}`
and an offline W&B run under `.../checkpoint/wandb` (upload with
`wandb sync <dir>` after `wandb login`).

`python examples/ads/smoke.py --model din --device flagos` runs three steps
without FlagScale.
