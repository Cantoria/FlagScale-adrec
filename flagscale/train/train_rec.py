# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""Training entrypoint for the native recommendation models (examples/rec).

Launched by FlagScale's ``native`` backend: Hydra composes the YAML, torchrun
starts this module with ``--config-file``, and the device is selected through
the platform interface.  Switching between native CUDA and Torch-FL is a
launch-time override; model code never names a vendor device.

Each task reproduces the protocol of its reference implementation:

* ``sequential`` (SASRec): validation every ``eval_every`` epochs on the
  second-to-last item, best validation epoch restored for the test item,
  HR/NDCG over 1 positive + 100 sampled negatives (kang205/SASRec).
* ``two_tower`` (DSSM): fixed number of epochs, then exact full-catalogue
  retrieval for each user's last item (torch-rechub run_ml_dssm.py).
* ``graph`` (LightGCN): fixed number of epochs with the test split evaluated
  every ``eval_every`` epochs, Recall/NDCG@20 over all items with training
  items masked (gusye1234/LightGCN-PyTorch).  The final-epoch result is the
  headline number; the best evaluated epoch is reported separately.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

# torch-fl must claim PyTorch's PrivateUse1 slot before the first flagos tensor
# is created; see train_retrieval.py for the same opt-in import.
_REQUESTED_DEVICE = os.environ.get("FLAGSCALE_DEVICE", os.environ.get("FS_PLATFORM", "auto")).strip().lower()
if _REQUESTED_DEVICE in {"flagos", "privateuseone"}:
    try:
        import torch_fl  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - vendor envs
        raise RuntimeError("FLAGSCALE_DEVICE=flagos requires a matching torch-fl installation.") from exc

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from flagscale.models.rec import build_rec_model
from flagscale.train.datasets.rec import RecData, build_rec_data
from flagscale.train.losses.rec import loss_for_batch, move_batch
from flagscale.train.metrics.rec import FullRankAccumulator, sampled_topk

try:
    from flagscale.platforms import get_platform
except ImportError:  # pragma: no cover - direct standalone invocation
    get_platform = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, help="Hydra-generated or standalone YAML")
    return parser.parse_args()


def rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(message: str) -> None:
    if rank_zero():
        print(message, flush=True)


def init_distributed(hardware: DictConfig | None = None) -> tuple[int, int, torch.device, Any]:
    """Select the device through the platform interface and join torchrun."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    hardware = hardware or {}
    requested = str(
        hardware.get("device_type", os.environ.get("FS_PLATFORM", os.environ.get("FLAGSCALE_DEVICE", "auto")))
    ).lower()
    platform = None
    if get_platform is not None:
        try:
            platform = get_platform()
        except (RuntimeError, ValueError):
            platform = None
    if requested in {"privateuseone", "flagos"}:
        if platform is None or platform.name() != "flagos":
            raise RuntimeError("device_type=flagos was requested, but a Torch-FL platform is unavailable")
    elif requested == "cpu":
        platform = None
    elif requested not in {"auto", ""} and (platform is None or platform.name() != requested):
        raise RuntimeError(
            f"Requested platform {requested!r} is unavailable; the active runtime exposes "
            f"{platform.name() if platform is not None else 'cpu'!r}"
        )
    if platform is not None and (requested not in {"auto", ""} or platform.is_available()):
        platform.set_device(local_rank)
        device = platform.device(local_rank)
    else:
        platform = None
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = str(hardware.get("distributed_backend", "auto")).lower()
        if backend == "auto":
            backend = platform.dist_backend() if platform is not None else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, device, platform


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int, rank: int, platform: Any = None) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if platform is not None:
        platform.manual_seed_all(seed)


def autocast_context(device: torch.device, dtype_name: str, platform: Any = None):
    if dtype_name == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    autocast_device = platform.amp_device_type() if platform is not None else device.type
    try:
        return torch.autocast(device_type=autocast_device, dtype=dtype)
    except (RuntimeError, TypeError):
        return contextlib.nullcontext()


def all_reduce_sum(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64).to(device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


class ExperimentLogger:
    """Optional Weights & Biases run on rank zero.

    ``WANDB_MODE=offline`` writes the run under ``<output_dir>/wandb`` so it can
    be uploaded later with ``wandb sync``; ``WANDB_DISABLED=true`` or a missing
    wandb package turns logging off without affecting training.
    """

    def __init__(self, train_cfg: DictConfig, output_dir: Path, extra: dict[str, Any]) -> None:
        self.run = None
        if not rank_zero() or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
            return
        try:
            import wandb
        except ImportError:
            log("wandb is not installed; continuing without experiment logging")
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        config = OmegaConf.to_container(train_cfg, resolve=True)
        config.update(extra)
        self.run = wandb.init(
            project=str(train_cfg.get("wandb_project", "flagos-rec")),
            name=str(train_cfg.get("wandb_name", f"{train_cfg.name}-{extra.get('device_type')}")),
            group=str(train_cfg.name),
            tags=[str(train_cfg.name), str(extra.get("device_type"))],
            dir=str(output_dir),
            config=config,
        )
        log(f"W&B run {self.run.id} mode={os.environ.get('WANDB_MODE', 'online')} dir={output_dir / 'wandb'}")

    def log(self, values: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            for key, value in values.items():
                self.run.summary[key] = value

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def train_epoch(model: nn.Module, data: RecData, epoch: int, task: str, device: torch.device,
                optimizer: torch.optim.Optimizer, weight_decay: float, dtype_name: str, platform: Any,
                log_every: int) -> dict[str, float]:
    model.train()
    loader = data.train
    sampler = getattr(loader.sampler, "sampler", loader.sampler)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    total: torch.Tensor | None = None
    steps = 0
    samples = 0
    started = time.perf_counter()
    for batch in loader:
        samples += int(next(iter(batch.values())).shape[0])
        with autocast_context(device, dtype_name, platform):
            loss = loss_for_batch(model, task, batch, device, weight_decay)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        detached = loss.detach().float()
        total = detached if total is None else total + detached
        steps += 1
        if log_every > 0 and steps % log_every == 0:
            log(f"epoch={epoch} step={steps} running_loss={float(total.item()) / steps:.6f}")
    if steps == 0:
        raise RuntimeError("Training dataset is empty for this rank")
    local_loss = float(total.item())
    elapsed = time.perf_counter() - started
    loss_sum, step_sum, sample_sum = all_reduce_sum([local_loss, float(steps), float(samples)], device)
    return {"loss": loss_sum / max(step_sum, 1.0), "steps": steps, "elapsed_s": elapsed,
            "samples_per_s": sample_sum / max(elapsed, 1e-9)}


@torch.no_grad()
def evaluate_sequential(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    scores = [model.predict(move_batch(batch, device)).float().cpu().numpy() for batch in loader]
    return sampled_topk(np.concatenate(scores))


@torch.no_grad()
def evaluate_two_tower(model: nn.Module, data: RecData, device: torch.device, ks: tuple[int, ...]) -> dict[str, float]:
    model.eval()
    item_ids, item_vectors = [], []
    for batch in data.items:
        item_ids.append(batch["movie_id"].numpy())
        item_vectors.append(model.item_embedding(move_batch(batch, device)))
    item_ids = np.concatenate(item_ids)
    items = torch.cat(item_vectors)
    accumulator = FullRankAccumulator(ks)
    for batch in data.test:
        users = model.user_embedding(move_batch(batch, device))
        # Exact cosine search (both towers are L2-normalized); torch-rechub
        # uses an approximate Annoy index for the same ranking.
        top = torch.topk(users @ items.T, k=max(ks), dim=1).indices.cpu().numpy()
        truths = [np.asarray([target]) for target in batch["movie_id"].numpy()]
        accumulator.update(item_ids[top], truths)
    return accumulator.result()


@torch.no_grad()
def evaluate_graph(model: nn.Module, data: RecData, device: torch.device, ks: tuple[int, ...],
                   user_batch: int = 4096) -> dict[str, float]:
    model.eval()
    users_emb, items_emb = model.all_scores()
    test_users = data.test_items["user"].astype(np.int64)
    test_items = data.test_items["item"].astype(np.int64)
    order = np.argsort(test_users, kind="mergesort")
    test_users, test_items = test_users[order], test_items[order]
    unique_users, starts = np.unique(test_users, return_index=True)
    truths_all = np.split(test_items, starts[1:])
    sampler = data.sampler
    accumulator = FullRankAccumulator(ks)
    for begin in range(0, len(unique_users), user_batch):
        users = unique_users[begin : begin + user_batch]
        scores = users_emb[torch.as_tensor(users, device=device)] @ items_emb.T
        # Exclude each user's training items from the ranking.
        lengths = sampler.ptr[users + 1] - sampler.ptr[users]
        rows = np.repeat(np.arange(len(users)), lengths)
        cols = np.concatenate([sampler.items_sorted[sampler.ptr[u] : sampler.ptr[u + 1]] for u in users])
        scores[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)] = float("-inf")
        top = torch.topk(scores, k=max(ks), dim=1).indices.cpu().numpy()
        accumulator.update(top, truths_all[begin : begin + user_batch])
    return accumulator.result()


def fmt(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float) and k != "elapsed_s")


def train(config: DictConfig) -> None:
    train_cfg = config.train
    rank, world_size, device, platform = init_distributed(train_cfg.get("hardware", {}))
    task = str(train_cfg.task)
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed, rank, platform)
    model_cfg = train_cfg.model
    dtype_name = str(model_cfg.get("torch_dtype", "fp32"))
    data = build_rec_data(task, train_cfg.data, int(train_cfg.batch_size),
                          int(train_cfg.get("eval_batch_size", train_cfg.batch_size)),
                          pin_memory=bool(platform and platform.supports_pin_memory()), seed=seed)
    model = build_rec_model(str(model_cfg.model_type), model_cfg, data.meta, device)
    if model.task != task:
        raise ValueError(f"model {model_cfg.model_type!r} trains task {model.task!r}, config says {task!r}")
    if task == "graph":
        model.set_graph(data.graph["src"], data.graph["dst"], data.graph["weight"])
    base_model = model
    parameters = sum(p.numel() for p in model.parameters())
    device_type = platform.name() if platform is not None else "cpu"
    log(f"model={model_cfg.model_type} parameters={parameters} device={device} world_size={world_size}")
    log(f"dataset={data.meta.get('name')} train_rows={len(data.train.dataset)}")
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device_type == "cuda" else None)

    optim_cfg = train_cfg.optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(optim_cfg.lr),
        betas=tuple(float(b) for b in optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    bpr_decay = float(optim_cfg.get("bpr_weight_decay", 0.0))
    epochs = int(train_cfg.epochs)
    eval_every = max(1, int(train_cfg.get("eval_every", 1)))
    ks = tuple(int(k) for k in train_cfg.get("topk", [10]))
    select_metric = str(train_cfg.get("select_metric", "ndcg@10"))
    log_every = int(train_cfg.get("log_every", 0))
    output_dir = Path(str(train_cfg.output_dir))
    logger = ExperimentLogger(train_cfg, output_dir, {"device_type": device_type, "parameters": parameters,
                                                      "torch_version": torch.__version__,
                                                      "dataset_meta": data.meta})

    def evaluate(split: str) -> dict[str, float]:
        started = time.perf_counter()
        if task == "sequential":
            result = evaluate_sequential(base_model, data.validation if split == "validation" else data.test, device)
        elif task == "two_tower":
            result = evaluate_two_tower(base_model, data, device, ks)
        else:
            result = evaluate_graph(base_model, data, device, ks)
        result["elapsed_s"] = time.perf_counter() - started
        return result

    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"epoch": 0, "value": float("-inf"), "state": None, "metrics": None}
    last_eval: dict[str, float] | None = None
    started = time.time()
    for epoch in range(1, epochs + 1):
        if task == "graph":
            data.sampler.resample()
        stats = train_epoch(model, data, epoch, task, device, optimizer, bpr_decay, dtype_name, platform, log_every)
        record: dict[str, Any] = {"epoch": epoch, "train": stats}
        logger.log({"train/loss": stats["loss"], "train/samples_per_s": stats["samples_per_s"],
                    "train/epoch_time_s": stats["elapsed_s"]}, step=epoch)
        if rank_zero() and (epoch % eval_every == 0 or epoch == epochs):
            # Sequential selects on validation; the other references monitor
            # their test split during training and have no validation split.
            split = "validation" if task == "sequential" else "test"
            result = evaluate(split)
            record[split] = result
            last_eval = result
            logger.log({f"{split}/{k}": v for k, v in result.items()}, step=epoch)
            value = float(result[select_metric])
            if value > best["value"]:
                state = None
                if task == "sequential":
                    state = copy.deepcopy({k: v.detach().cpu() for k, v in base_model.state_dict().items()})
                best = {"epoch": epoch, "value": value, "state": state, "metrics": result}
            log(f"epoch={epoch}/{epochs} loss={stats['loss']:.6f} samples_per_s={stats['samples_per_s']:.0f} "
                f"{split}: {fmt(result)}")
        else:
            log(f"epoch={epoch}/{epochs} loss={stats['loss']:.6f} samples_per_s={stats['samples_per_s']:.0f}")
        history.append(record)
        barrier()

    if rank_zero():
        if task == "sequential":
            base_model.load_state_dict(best["state"])
            final = evaluate("test")
            headline = {"protocol": "best validation epoch restored, then test", "best_epoch": best["epoch"],
                        "validation": best["metrics"], "test": final}
        else:
            final = last_eval
            headline = {"protocol": "test after the last epoch (reference protocol)", "test": final,
                        "best_evaluated_epoch": best["epoch"], "best_evaluated_test": best["metrics"]}
        log(f"final test: {fmt(final)} elapsed={time.time() - started:.1f}s")
        base_model.save(output_dir)
        rates = [r["train"]["samples_per_s"] for r in history]
        summary = {
            "model": str(model_cfg.model_type), "task": task, "dataset": data.meta.get("name"),
            "device": str(device), "device_type": device_type, "world_size": world_size,
            "torch_version": torch.__version__, "parameters": parameters, "select_metric": select_metric,
            **headline,
            "train_samples_per_s_median": float(np.median(rates)),
            "elapsed_s": time.time() - started, "history": history,
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        logger.summary({f"final_test/{k}": v for k, v in final.items()})
        logger.summary({"train_samples_per_s_median": summary["train_samples_per_s_median"],
                        "parameters": parameters, "elapsed_s": summary["elapsed_s"]})
        log(f"saved checkpoint and metrics to {output_dir}")
    logger.finish()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config_file)
    if not config.get("train"):
        raise ValueError("Configuration must contain a train section")
    train(config)


if __name__ == "__main__":
    main()
