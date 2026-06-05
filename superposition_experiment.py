#!/usr/bin/env python3
"""
Superposition-biased coder experiment.

This script creates a synthetic hierarchical classification task and compares
plain vs skip-connected models using three diagnostics from the paper:

1. End-to-end performance.
2. Base drift: how much the base representation moves during training.
3. Extendibility gap: freeze a learned base and retrain only refinement/head.
   If frozen-base performance is close to full end-to-end performance, the base
   is extendible/non-rewriting.

Run:
    python superposition_experiment_v2.py --epochs 60 --freeze-epochs 60 --device cpu

Optional:
    python superposition_experiment.py --epochs 80 --aux-weight 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Synthetic hierarchical source
# -----------------------------

@dataclass
class DataConfig:
    n_train: int = 12000
    n_test: int = 4000
    input_dim: int = 32
    n_coarse: int = 4
    fine_per_coarse: int = 4
    coarse_scale: float = 4.0
    fine_scale: float = 3.0
    noise_std: float = 0.6
    nonrefinable: bool = False


def make_hierarchical_data(cfg: DataConfig, seed: int = 0) -> Tuple[TensorDataset, TensorDataset]:
    """
    Creates labels with hierarchy:
        coarse label c in {0,...,n_coarse-1}
        fine label f in {0,...,fine_per_coarse-1}
        class y = c * fine_per_coarse + f

    Refinable mode:
        x = coarse_center[c] + fine_offset[c,f] + noise
        Coarse structure is strong; fine class is a residual refinement.

    Nonrefinable mode:
        fine offsets are not organized around coarse centers; this is a harder
        condition where the coarse base is less naturally extendible.
    """
    gen = torch.Generator().manual_seed(seed)
    n_classes = cfg.n_coarse * cfg.fine_per_coarse

    # Coarse centers are orthogonal-ish random directions.
    coarse = torch.randn(cfg.n_coarse, cfg.input_dim, generator=gen)
    coarse = cfg.coarse_scale * F.normalize(coarse, dim=1)

    if cfg.nonrefinable:
        # Fine classes get large independent centers, weakening the coarse-to-fine structure.
        class_centers = torch.randn(n_classes, cfg.input_dim, generator=gen)
        class_centers = cfg.coarse_scale * F.normalize(class_centers, dim=1)
    else:
        fine = torch.randn(cfg.n_coarse, cfg.fine_per_coarse, cfg.input_dim, generator=gen)
        # Make fine offsets live mostly orthogonal to the corresponding coarse center.
        fine = fine - (fine * coarse[:, None, :]).sum(-1, keepdim=True) * coarse[:, None, :] / (
            coarse[:, None, :].pow(2).sum(-1, keepdim=True) + 1e-8
        )
        fine = cfg.fine_scale * F.normalize(fine, dim=2)
        class_centers = (coarse[:, None, :] + fine).reshape(n_classes, cfg.input_dim)

    def sample(n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = torch.randint(0, n_classes, (n,), generator=gen)
        c = y // cfg.fine_per_coarse
        x = class_centers[y] + cfg.noise_std * torch.randn(n, cfg.input_dim, generator=gen)
        # Standardize features for stable training.
        return x.float(), y.long(), c.long()

    x_tr, y_tr, c_tr = sample(cfg.n_train)
    x_te, y_te, c_te = sample(cfg.n_test)

    mean = x_tr.mean(dim=0, keepdim=True)
    std = x_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    train = TensorDataset(x_tr, y_tr, c_tr)
    test = TensorDataset(x_te, y_te, c_te)
    return train, test


# -----------------------------
# Models
# -----------------------------

class ResidualBlock(nn.Module):
    def __init__(self, width: int, hidden_mult: int = 2, residual_scale: float = 0.2):
        super().__init__()
        hidden = hidden_mult * width
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, width),
        )
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.net(x)


def zero_init_residual_branches(module: nn.Module) -> None:
    """
    Make residual blocks start as exact identity maps by zero-initializing the
    final linear layer in each residual branch. This is important for the
    superposition experiment: after freezing a base, a reset residual refinement
    should start from "preserve the base" rather than from a random rewrite.
    """
    for m in module.modules():
        if isinstance(m, ResidualBlock):
            last = m.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)


class PlainBlock(nn.Module):
    def __init__(self, width: int, hidden_mult: int = 2):
        super().__init__()
        hidden = hidden_mult * width
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, width),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseRefineNet(nn.Module):
    """
    Base + refinement model.

    - base: creates candidate base representation b
    - refinement: either plain transformation or residual refinement stack
    - head: predicts fine class
    - aux_head: predicts coarse class from the base representation

    use_skip=True makes the refinement path superposition-like:
        h_{l+1} = h_l + F_l(h_l)
    """
    def __init__(
        self,
        input_dim: int,
        width: int,
        n_classes: int,
        n_coarse: int,
        depth: int = 4,
        use_skip: bool = True,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.n_classes = n_classes
        self.n_coarse = n_coarse
        self.depth = depth
        self.use_skip = use_skip

        self.base = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )

        blocks = []
        for _ in range(depth):
            if use_skip:
                blocks.append(ResidualBlock(width, residual_scale=residual_scale))
                blocks.append(nn.ReLU())
            else:
                blocks.append(PlainBlock(width))
        self.refinement = nn.Sequential(*blocks)
        if self.use_skip:
            zero_init_residual_branches(self.refinement)

        self.head = nn.Linear(width, n_classes)
        self.aux_head = nn.Linear(width, n_coarse)

    def forward(self, x: torch.Tensor, return_reps: bool = False):
        b = self.base(x)
        z = self.refinement(b)
        logits = self.head(z)
        aux_logits = self.aux_head(b)
        if return_reps:
            return logits, aux_logits, b, z
        return logits

    def reset_refinement_and_heads(self) -> None:
        """
        Keep base fixed but reset refinement/head/aux_head, for the frozen-base
        extendibility diagnostic.
        """
        for module in [self.refinement, self.head, self.aux_head]:
            for m in module.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        if self.use_skip:
            zero_init_residual_branches(self.refinement)

    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


# -----------------------------
# Metrics and training
# -----------------------------

@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-3
    aux_weight: float = 0.1
    width: int = 128
    depth: int = 4
    residual_scale: float = 0.2
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "runs_superposition"


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    n = 0
    total_loss = 0.0
    total_aux_loss = 0.0
    correct = 0
    correct_coarse = 0

    for x, y, c in loader:
        x, y, c = x.to(device), y.to(device), c.to(device)
        logits, aux_logits, _, _ = model(x, return_reps=True)
        loss = F.cross_entropy(logits, y, reduction="sum")
        aux_loss = F.cross_entropy(aux_logits, c, reduction="sum")
        total_loss += float(loss.item())
        total_aux_loss += float(aux_loss.item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        correct_coarse += int((aux_logits.argmax(dim=1) == c).sum().item())
        n += y.numel()

    return {
        "loss": total_loss / n,
        "aux_loss": total_aux_loss / n,
        "acc": correct / n,
        "coarse_acc": correct_coarse / n,
    }


@torch.no_grad()
def base_repr(model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    x = x.to(device)
    _, _, b, _ = model(x, return_reps=True)
    return b.detach().cpu()


def representation_drift(b0: torch.Tensor, b1: torch.Tensor) -> float:
    return float(torch.mean((b1 - b0) ** 2).item())


def train_model(
    model: BaseRefineNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    probe_x: torch.Tensor,
    run_name: str,
) -> Tuple[BaseRefineNet, Dict[str, float], list]:
    device = cfg.device
    model.to(device)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    b_initial = base_repr(model, probe_x, device)
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y, c in train_loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            logits, aux_logits, _, _ = model(x, return_reps=True)
            loss = F.cross_entropy(logits, y)
            if cfg.aux_weight > 0:
                loss = loss + cfg.aux_weight * F.cross_entropy(aux_logits, c)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        b_now = base_repr(model, probe_x, device)
        drift = representation_drift(b_initial, b_now)

        row = {
            "run": run_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_coarse_acc": train_metrics["coarse_acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_coarse_acc": test_metrics["coarse_acc"],
            "base_drift_from_init": drift,
        }
        history.append(row)

        if epoch == 1 or epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs:
            print(
                f"[{run_name}] epoch {epoch:03d} "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_acc={test_metrics['acc']:.3f} "
                f"coarse_acc={test_metrics['coarse_acc']:.3f} "
                f"drift={drift:.4f}"
            )

    final_metrics = evaluate(model, test_loader, device)
    return model, final_metrics, history


def frozen_base_extendibility_experiment(
    trained_model: BaseRefineNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    probe_x: torch.Tensor,
    run_name: str,
) -> Tuple[BaseRefineNet, Dict[str, float], list]:
    """
    Freeze the learned base, reset refinement/head, train only refinement/head.

    Extendibility gap:
        frozen_test_loss - full_end_to_end_test_loss

    If this gap is small, the base behaves like an extendible/non-rewriting base.
    """
    model = copy.deepcopy(trained_model)
    model.reset_refinement_and_heads()
    model.freeze_base()
    return train_model(
        model,
        train_loader,
        test_loader,
        cfg,
        probe_x,
        run_name=run_name,
    )



def plot_results(history_rows: list, summary_rows: list, out_dir: Path) -> None:
    """
    Save plots that illustrate superposition-like behavior.

    Key idea:
        If frozen-base refinement nearly matches full end-to-end training,
        the learned base is extendible/non-rewriting.

    Plots saved:
        plots/test_loss_curves.png
        plots/test_accuracy_curves.png
        plots/base_drift_full_training.png
        plots/extendibility_gap_loss.png
        plots/full_vs_frozen_accuracy.png
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for row in history_rows:
        runs.setdefault(row["run"], []).append(row)

    # 1. Test loss curves: full training vs frozen-base refinement.
    plt.figure()
    for run_name, rows in sorted(runs.items()):
        epochs = [int(r["epoch"]) for r in rows]
        test_loss = [float(r["test_loss"]) for r in rows]
        plt.plot(epochs, test_loss, label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.title("Full training vs frozen-base refinement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "test_loss_curves.png", dpi=200)
    plt.close()

    # 2. Test accuracy curves.
    plt.figure()
    for run_name, rows in sorted(runs.items()):
        epochs = [int(r["epoch"]) for r in rows]
        test_acc = [float(r["test_acc"]) for r in rows]
        plt.plot(epochs, test_acc, label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy of full and frozen-base training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "test_accuracy_curves.png", dpi=200)
    plt.close()

    # 3. Base drift during full training only.
    plt.figure()
    has_full_run = False
    for run_name, rows in sorted(runs.items()):
        if not run_name.endswith("_full"):
            continue
        has_full_run = True
        epochs = [int(r["epoch"]) for r in rows]
        drift = [float(r["base_drift_from_init"]) for r in rows]
        plt.plot(epochs, drift, label=run_name.replace("_full", ""))
    if has_full_run:
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared base drift")
        plt.title("Base drift during end-to-end training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "base_drift_full_training.png", dpi=200)
    plt.close()

    # 4. Extendibility gap by model.
    if summary_rows:
        labels = [row["model"] for row in summary_rows]
        gaps = [float(row["extendibility_gap_loss"]) for row in summary_rows]
        plt.figure()
        plt.bar(labels, gaps)
        plt.xlabel("Model")
        plt.ylabel("Frozen-base loss - full loss")
        plt.title("Extendibility gap: smaller means more superposition-like")
        plt.tight_layout()
        plt.savefig(plot_dir / "extendibility_gap_loss.png", dpi=200)
        plt.close()

    # 5. Full vs frozen final accuracy by model.
    if summary_rows:
        labels = [row["model"] for row in summary_rows]
        x = list(range(len(labels)))
        width = 0.35
        full_acc = [float(row["full_test_acc"]) for row in summary_rows]
        frozen_acc = [float(row["frozen_test_acc"]) for row in summary_rows]

        plt.figure()
        plt.bar([i - width / 2 for i in x], full_acc, width, label="Full end-to-end")
        plt.bar([i + width / 2 for i in x], frozen_acc, width, label="Frozen base + refinement")
        plt.xticks(x, labels)
        plt.xlabel("Model")
        plt.ylabel("Test accuracy")
        plt.title("Does freezing the base preserve performance?")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "full_vs_frozen_accuracy.png", dpi=200)
        plt.close()

    print(f"Wrote plots to {plot_dir}")


def write_csv(rows: Iterable[Dict[str, float]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Main experiment
# -----------------------------

def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.")
        device = "cpu"

    data_cfg = DataConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        n_coarse=args.n_coarse,
        fine_per_coarse=args.fine_per_coarse,
        coarse_scale=args.coarse_scale,
        fine_scale=args.fine_scale,
        noise_std=args.noise_std,
        nonrefinable=args.nonrefinable,
    )
    train_ds, test_ds = make_hierarchical_data(data_cfg, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Fixed probe batch used for representation drift.
    probe_x = train_ds.tensors[0][: min(512, len(train_ds))]

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aux_weight=args.aux_weight,
        width=args.width,
        depth=args.depth,
        residual_scale=args.residual_scale,
        seed=args.seed,
        device=device,
        output_dir=args.output_dir,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_classes = args.n_coarse * args.fine_per_coarse
    all_history = []
    summary_rows = []

    for use_skip in [False, True]:
        model_name = "skip" if use_skip else "plain"
        print(f"\n=== Training {model_name} model end-to-end ===")

        model = BaseRefineNet(
            input_dim=args.input_dim,
            width=args.width,
            n_classes=n_classes,
            n_coarse=args.n_coarse,
            depth=args.depth,
            use_skip=use_skip,
            residual_scale=args.residual_scale,
        )

        trained, full_metrics, hist = train_model(
            model,
            train_loader,
            test_loader,
            cfg,
            probe_x,
            run_name=f"{model_name}_full",
        )
        all_history.extend(hist)

        print(f"\n=== Frozen-base extendibility diagnostic for {model_name} ===")
        frozen_cfg = copy.deepcopy(cfg)
        frozen_cfg.epochs = args.freeze_epochs

        frozen_model, frozen_metrics, frozen_hist = frozen_base_extendibility_experiment(
            trained,
            train_loader,
            test_loader,
            frozen_cfg,
            probe_x,
            run_name=f"{model_name}_frozen_base",
        )
        all_history.extend(frozen_hist)

        extendibility_gap_loss = frozen_metrics["loss"] - full_metrics["loss"]
        extendibility_gap_acc = full_metrics["acc"] - frozen_metrics["acc"]

        summary = {
            "model": model_name,
            "nonrefinable_data": int(args.nonrefinable),
            "full_test_loss": full_metrics["loss"],
            "full_test_acc": full_metrics["acc"],
            "full_base_coarse_acc": full_metrics["coarse_acc"],
            "frozen_test_loss": frozen_metrics["loss"],
            "frozen_test_acc": frozen_metrics["acc"],
            "frozen_base_coarse_acc": frozen_metrics["coarse_acc"],
            "extendibility_gap_loss": extendibility_gap_loss,
            "extendibility_gap_acc": extendibility_gap_acc,
            "aux_weight": args.aux_weight,
            "width": args.width,
            "depth": args.depth,
            "seed": args.seed,
        }
        summary_rows.append(summary)

        print(
            f"\n[{model_name}] extendibility_gap_loss={extendibility_gap_loss:.4f} "
            f"extendibility_gap_acc={extendibility_gap_acc:.4f}"
        )

    write_csv(all_history, out_dir / "training_history.csv")
    write_csv(summary_rows, out_dir / "summary.csv")

    if not args.no_plots:
        plot_results(all_history, summary_rows, out_dir)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(row)

    print(f"\nWrote:\n  {out_dir / 'training_history.csv'}\n  {out_dir / 'summary.csv'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Superposition-biased deep-network experiment")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--freeze-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--aux-weight", type=float, default=0.1, help="Optional coarse auxiliary loss weight")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--residual-scale", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output-dir", type=str, default="runs_superposition")
    p.add_argument("--no-plots", action="store_true", help="Do not generate summary plots.")

    # Data.
    p.add_argument("--n-train", type=int, default=12000)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--input-dim", type=int, default=32)
    p.add_argument("--n-coarse", type=int, default=4)
    p.add_argument("--fine-per-coarse", type=int, default=4)
    p.add_argument("--coarse-scale", type=float, default=4.0)
    p.add_argument("--fine-scale", type=float, default=3.0)
    p.add_argument("--noise-std", type=float, default=0.6)
    p.add_argument(
        "--nonrefinable",
        action="store_true",
        help="Use a less hierarchical source as a negative-control condition.",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_experiment(args)
