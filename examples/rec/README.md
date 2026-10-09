# Recommendation model training with FlagScale

Native FlagScale training for the recommendation models of the FlagOS
localization project.  Each model follows the official implementation or the
framework whose result it is compared with; see [EVIDENCE.md](EVIDENCE.md).

| Config | Model | Data | Reference |
| --- | --- | --- | --- |
| `sasrec` | SASRec (kang205/SASRec protocol) | MovieLens-1M, 5-core, leave-one-out, 100 sampled negatives | paper HR@10 0.8245 / NDCG@10 0.5905 |
| `dssm` | DSSM two-tower (torch-rechub) | MovieLens-1M, last item per user, full-catalogue retrieval | torch-rechub Hit@100 14.74% |
| `lightgcn` | LightGCN (gusye1234/LightGCN-PyTorch) | Gowalla (official split) | paper recall@20 0.1823 / ndcg@20 0.1555 |

Layout (independent of the retrieval and advertising modules):

* `flagscale/models/rec/`: `sasrec.py`, `dssm.py`, `lightgcn.py`, shared layers, registry;
* `flagscale/train/datasets/rec.py`, `losses/rec.py`, `metrics/rec.py`;
* `flagscale/train/train_rec.py`: per-task protocol (see its docstring);
* data under `/flagos-rec/datasets`, results under `/flagos-rec/ckpts`.

## Data

```bash
python examples/rec/prepare_movielens_sasrec.py   # -> /flagos-rec/datasets/movielens_1m_sasrec
python examples/rec/prepare_movielens_dssm.py     # -> /flagos-rec/datasets/movielens_1m_dssm
python examples/rec/prepare_gowalla.py            # -> /flagos-rec/datasets/gowalla_lightgcn
```

## Run

```bash
export PYTHONPATH=$PWD FLAGSCALE_HOME=$PWD WANDB_MODE=offline
# Torch-FL (LightGCN needs index_add propagation: Torch-FL has no sparse tensors)
FLAGSCALE_PYTHON_ENV=/flagos-search-codes/torch-fl-cuda/.venv \
python -m flagscale.run --config-path examples/rec/conf --config-name train action=test train=lightgcn \
  train.hardware.device_type=flagos train.hardware.distributed_backend=flagos train.model.propagation=index_add
# native CUDA baseline
FLAGSCALE_PYTHON_ENV=/opt/conda \
python -m flagscale.run --config-path examples/rec/conf --config-name train action=test train=lightgcn \
  train.hardware.device_type=cuda
```

Outputs: `/flagos-rec/ckpts/<model>-<device>/checkpoint/{model.pt,config.json,metrics.json}`
and an offline W&B run under `.../checkpoint/wandb`.

`python examples/rec/smoke.py --model sasrec --device flagos` runs three steps
without FlagScale.
