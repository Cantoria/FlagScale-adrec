"""Few-step smoke test for a rec model on cpu, cuda or Torch-FL.

Runs forward, backward and optimizer steps on real preprocessed batches plus
one evaluation pass, without FlagScale or torchrun:

    python examples/rec/smoke.py --model lightgcn --device flagos --steps 3
"""

from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=["sasrec", "dssm", "lightgcn"], required=True)
parser.add_argument("--device", default="cpu", help="cpu, cuda or flagos")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--propagation", default=None, help="LightGCN: sparse or index_add")
args = parser.parse_args()
os.environ["FLAGSCALE_DEVICE"] = args.device
if args.device.startswith("flagos"):
    import torch_fl  # noqa: F401  # must be imported before torch

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from flagscale.models.rec import build_rec_model  # noqa: E402
from flagscale.train import train_rec  # noqa: E402
from flagscale.train.datasets.rec import build_rec_data  # noqa: E402
from flagscale.train.losses.rec import loss_for_batch  # noqa: E402

CONF = os.path.join(os.path.dirname(__file__), "conf", "train")


def main() -> None:
    cfg = OmegaConf.load(os.path.join(CONF, f"{args.model}.yaml"))
    if args.propagation:
        cfg.model.propagation = args.propagation
    device = torch.device(f"{args.device}:0" if args.device in {"cuda", "flagos"} else args.device)
    data = build_rec_data(cfg.task, cfg.data, int(cfg.batch_size), int(cfg.get("eval_batch_size", 1024)),
                          pin_memory=False, seed=0)
    model = build_rec_model(args.model, cfg.model, data.meta, device)
    if cfg.task == "graph":
        model.set_graph(data.graph["src"], data.graph["dst"], data.graph["weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    decay = float(cfg.optimizer.get("bpr_weight_decay", 0.0))
    print(f"{args.model} parameters={sum(p.numel() for p in model.parameters())} device={device}", flush=True)
    model.train()
    for step, batch in enumerate(data.train, start=1):
        loss = loss_for_batch(model, cfg.task, batch, device, decay)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={float(loss.detach().cpu()):.6f}", flush=True)
        if step >= args.steps:
            break
    ks = tuple(int(k) for k in cfg.get("topk", [10]))
    if cfg.task == "sequential":
        result = train_rec.evaluate_sequential(model, data.validation, device)
    elif cfg.task == "two_tower":
        result = train_rec.evaluate_two_tower(model, data, device, ks)
    else:
        result = train_rec.evaluate_graph(model, data, device, ks)
    print("eval", {k: round(v, 4) for k, v in result.items()})
    print("SMOKE OK")


if __name__ == "__main__":
    main()
