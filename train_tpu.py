#!/usr/bin/env python3
"""
OASIS TPU Trainer — XLA-native, static-graph implementation.

Key design principle: The entire forward → backward → compress → step sequence
must be ONE contiguous XLA graph. This means:
  1. No Python loops over parameters inside the hot path.
  2. No .item() calls before xm.optimizer_step().
  3. Compression is done via pre-registered static projection matrices (V).
  4. All compression math is pure torch tensor ops (matmul only in freeze mode).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch_xla.core.xla_model as xm

sys.path.insert(0, str(Path(__file__).parent))

from oasis.model import ResNet18CIFAR
from oasis.logger import OASISLogger


# ── XLA-Native Gradient Compressor ───────────────────────────────────────────

class FrozenBasisCompressor:
    """
    Gradient compressor for TPU/XLA that uses only static-shape matrix
    multiplications. All dynamic logic (SVD, QR, rank selection) has been
    moved OUTSIDE the training graph.

    During training, compression is exactly:
        G_hat = (G @ V) @ V.T
    where V is a fixed, preloaded orthonormal basis matrix for each layer.

    This produces a static XLA graph: two matmuls, zero control flow.
    """

    def __init__(
        self,
        rank_table: Dict[str, int],
        bases: Dict[str, torch.Tensor],
        min_numel: int = 1024,
        skip_classifier: bool = True,
    ):
        self.rank_table = rank_table
        self.bases = bases          # name → V  [n, r]  on XLA device
        self.min_numel = min_numel
        self.skip_classifier = skip_classifier
        self.last_stats: Dict[str, dict] = {}

    @torch.no_grad()
    def compress_all(self, model: nn.Module):
        """
        Compress every compressible gradient in-place.
        Called AFTER loss.backward(), BEFORE xm.optimizer_step().

        All operations are pure XLA tensor ops — no Python control flow
        that depends on tensor values, no .item() calls.
        """
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            G = param.grad

            # Skip small / 1D / classifier tensors
            is_classifier = self.skip_classifier and ("fc." in name or "classifier" in name)
            if G.dim() < 2 or G.numel() < self.min_numel or is_classifier:
                continue

            if name not in self.bases:
                continue

            V = self.bases[name]        # [n, r]  static, on XLA device
            G_2d = G.reshape(G.shape[0], -1)   # [m, n]

            # G_hat = (G @ V) @ V.T  — two static-shape matmuls, nothing else
            G_hat = (G_2d @ V) @ V.t()
            param.grad = G_hat.reshape(G.shape)


def build_frozen_compressor(rank_table_path: str, bases_path: str, device) -> FrozenBasisCompressor:
    """Load rank table + bases from disk and build the compressor."""
    with open(rank_table_path) as f:
        rank_table = json.load(f)

    # Always load to CPU first — XLA cannot restore tagged xla:0 storages
    raw_bases = torch.load(bases_path, map_location="cpu", weights_only=True)

    # Move to XLA device one tensor at a time
    bases_on_device = {}
    for name, V in raw_bases.items():
        r = rank_table.get(name)
        if r is not None:
            # Trim to calibrated rank if the saved basis is wider
            bases_on_device[name] = V[:, :r].to(device)

    return FrozenBasisCompressor(rank_table, bases_on_device)


# ── Data Loading ──────────────────────────────────────────────────────────────

def build_loaders(args):
    is_cifar100 = args.dataset == "cifar100"
    mean = (0.5071, 0.4867, 0.4408) if is_cifar100 else (0.4914, 0.4822, 0.4465)
    std  = (0.2675, 0.2565, 0.2761) if is_cifar100 else (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_class = datasets.CIFAR100 if is_cifar100 else datasets.CIFAR10
    train_ds = ds_class(args.data_dir, train=True,  download=xm.is_master_ordinal(), transform=train_tf)
    val_ds   = ds_class(args.data_dir, train=False, download=xm.is_master_ordinal(), transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=False,  # pin_memory=True causes errors on XLA
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
        pin_memory=False,
    )
    return train_loader, val_loader


# ── Training & Eval Loops ─────────────────────────────────────────────────────

def train_one_epoch(epoch, model, loader, criterion, optimizer, scheduler, compressor, logger, device, args):
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches    = 0
    train_compute_time = 0.0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        _t = time.perf_counter()

        # ── Forward ──────────────────────────────────────────────────────────
        logits = model(images)
        loss   = criterion(logits, labels)

        # ── Backward ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # ── Compress (pure XLA matmuls, no Python control flow on values) ────
        if compressor is not None:
            compressor.compress_all(model)

        # ── Optimizer step ────────────────────────────────────────────────────
        # CRITICAL: Do NOT use xm.optimizer_step() on a TPU Pod.
        # It calls xm.reduce_gradients() internally which does an all-reduce
        # across all workers. If workers desync by even one step, deadlock.
        # Instead: plain optimizer.step() + xm.mark_step() for local execution.
        optimizer.step()
        xm.mark_step()

        train_compute_time += time.perf_counter() - _t

        n_batches += 1

        # ── Logging (only sync to host at log intervals to minimize stalls) ──
        if step == 1:
            # Force a host sync on step 1 to confirm compilation finished
            loss_val = loss.item()
            print(f"    [TPU] Step 1 done (loss={loss_val:.4f}) — graph cached, training is now fast!")
            running_loss += loss_val
            running_acc  += (logits.argmax(1) == labels).float().mean().item()
        elif step % args.log_every == 0 or step == len(loader):
            running_loss += loss.item()
            running_acc  += (logits.argmax(1) == labels).float().mean().item()
            if xm.is_master_ordinal():
                logger.step_log(epoch, step, running_loss / n_batches,
                                running_acc / n_batches, optimizer.param_groups[0]["lr"])
        # Steps between log intervals: no .item() calls, pure XLA execution

    scheduler.step()
    avg_loss = running_loss / max(n_batches, 1)
    avg_acc  = running_acc  / max(n_batches, 1)
    return avg_loss, avg_acc, train_compute_time


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = n = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        acc_t  = (logits.argmax(1) == labels).float().mean()
        # Local-only graph flush, no cross-worker sync
        xm.mark_step()
        total_loss += loss.item()
        total_acc  += acc_t.item()
        n += 1
    return total_loss / n, total_acc / n


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OASIS TPU Trainer (XLA-native frozen-basis compression)")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=5e-4)
    p.add_argument("--log-every",     type=int,   default=50)
    p.add_argument("--data-dir",      type=str,   default="./data")
    p.add_argument("--seed",          type=int,   default=42)
    # IMPORTANT: Keep num-workers=0 on TPU. DataLoader multiprocessing workers
    # deadlock against the XLA runtime thread after step 1.
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--dataset",       type=str,   default="cifar10", choices=["cifar10", "cifar100"])
    # Compression
    p.add_argument("--rank-table",    type=str,   default=None, help="JSON rank table from calibrate_ranks.py")
    p.add_argument("--bases-file",    type=str,   default=None, help=".pt bases file from calibrate_ranks.py")
    p.add_argument("--no-compress",   action="store_true", help="Disable compression (baseline run)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = xm.xla_device()

    # Build compressor (or None for baseline)
    compressor = None
    mode_str   = "Baseline (no compression)"
    if not args.no_compress:
        if args.rank_table is None or args.bases_file is None:
            raise ValueError("--rank-table and --bases-file are required unless --no-compress is set.\n"
                             "Run: python3 calibrate_ranks.py --dataset cifar100 --out ranks_cifar100.json --out-bases bases_cifar100.pt")
        compressor = build_frozen_compressor(args.rank_table, args.bases_file, device)
        n_compressed = len(compressor.bases)
        mode_str = f"OASIS-Calibrated (frozen bases, {n_compressed} layers)"

    train_loader, val_loader = build_loaders(args)
    num_classes = 100 if args.dataset == "cifar100" else 10
    model       = ResNet18CIFAR(num_classes=num_classes).to(device)
    optimizer   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)

    logger = OASISLogger(args.epochs, len(train_loader), args.log_every)
    if xm.is_master_ordinal():
        logger.print_header({
            "Dataset":    args.dataset.upper(),
            "Mode":       mode_str,
            "Batch size": args.batch_size,
            "Epochs":     args.epochs,
            "Seed":       args.seed,
            "Device":     "TPU (XLA)",
        })

    history, best_val_acc, best_epoch = [], 0.0, 0

    for epoch in range(1, args.epochs + 1):
        if xm.is_master_ordinal():
            logger.epoch_start(epoch)

        train_loss, train_acc, compute_time = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, scheduler,
            compressor, logger, device, args
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if xm.is_master_ordinal():
            logger.epoch_end(epoch, train_loss, train_acc, val_loss, val_acc,
                             train_compute_time=compute_time)
            history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                            "val_loss": val_loss, "val_acc": val_acc})
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch

    if xm.is_master_ordinal():
        logger.print_final_summary(best_val_acc, best_epoch, history)


if __name__ == "__main__":
    main()
