#!/usr/bin/env python3
"""
OASIS TPU Pod Trainer — mirrors the proven tdcf TPU pattern exactly.

Key design decisions copied from working tdcf/train_tpu_tiny_imagenet.py:
  1. Use pl.MpDeviceLoader (NOT plain DataLoader.to(device))
  2. Accumulate metrics as on-device XLA tensors — never .item() inside step loop
  3. Use xm.mesh_reduce after the epoch to sync metrics across chips
  4. xmp.spawn(..., start_method="spawn")
  5. num_workers=0 forced for TPU stability
  6. DistributedSampler for data sharding across all chips
"""
import argparse
import json
import os
import sys
import traceback
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

sys.path.insert(0, str(Path(__file__).parent))
from oasis.model import ResNet18CIFAR


# ── XLA-Native Frozen Basis Compressor ───────────────────────────────────────

class FrozenBasisCompressor:
    """
    Gradient compressor for TPU/XLA.
    Per layer: G_hat = (G @ V) @ V.T  — two static-shape matmuls, nothing else.
    """
    def __init__(self, bases: Dict[str, torch.Tensor], min_numel: int = 1024):
        self.bases     = bases
        self.min_numel = min_numel

    @torch.no_grad()
    def compress_all(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            G = param.grad
            if G.dim() < 2 or G.numel() < self.min_numel:
                continue
            if "fc." in name or "classifier" in name:
                continue
            if name not in self.bases:
                continue
            V    = self.bases[name]
            G2d  = G.reshape(G.shape[0], -1)
            param.grad = ((G2d @ V) @ V.t()).reshape(G.shape)


def load_compressor(rank_table_path, bases_path, device) -> FrozenBasisCompressor:
    with open(rank_table_path) as f:
        rank_table = json.load(f)
    raw = torch.load(bases_path, map_location="cpu", weights_only=True)
    bases = {name: V[:, :rank_table[name]].to(device)
             for name, V in raw.items() if name in rank_table}
    return FrozenBasisCompressor(bases)


# ── Data Loading ──────────────────────────────────────────────────────────────

def build_loaders(args, world_size, rank):
    is_c100 = args.dataset == "cifar100"
    mean = (0.5071, 0.4867, 0.4408) if is_c100 else (0.4914, 0.4822, 0.4465)
    std  = (0.2675, 0.2565, 0.2761) if is_c100 else (0.2470, 0.2435, 0.2616)

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

    ds_class  = datasets.CIFAR100 if is_c100 else datasets.CIFAR10
    train_ds  = ds_class(args.data_dir, train=True,  download=(rank == 0), transform=train_tf)
    val_ds    = ds_class(args.data_dir, train=False, download=(rank == 0), transform=val_tf)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    # num_workers=0 is MANDATORY on TPU — multiprocessing workers deadlock XLA
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=0,
                              drop_last=True, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              sampler=val_sampler,   num_workers=0,
                              drop_last=False, pin_memory=False)
    return train_loader, val_loader


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(epoch, model, loader, device, criterion, optimizer, compressor, args):
    model.train()

    # On-device accumulators — never .item() inside the step loop (mirrors tdcf pattern)
    loss_sum = torch.zeros((), device=device)
    correct  = torch.zeros((), device=device)
    n_seen   = torch.zeros((), device=device)

    # pl.MpDeviceLoader is the XLA-native prefetcher — this is what the working
    # tdcf trainer uses instead of plain DataLoader + .to(device)
    para_loader = pl.MpDeviceLoader(loader, device)

    for step, (images, labels) in enumerate(para_loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        if compressor is not None:
            compressor.compress_all(model)

        xm.optimizer_step(optimizer)

        # Accumulate on-device (no host sync)
        batch_n   = labels.new_tensor(labels.size(0), dtype=torch.float32)
        loss_sum += loss.detach() * batch_n
        correct  += (logits.argmax(1) == labels).sum().float()
        n_seen   += batch_n

        if step == 1 and xm.is_master_ordinal():
            print(f"    [TPU] Epoch {epoch} Step 1 complete — graph compiled and cached.", flush=True)

    # Aggregate across all chips via mesh_reduce (matches tdcf pattern exactly)
    n_total    = xm.mesh_reduce("train_n",    n_seen,   sum)
    corr_total = xm.mesh_reduce("train_corr", correct,  sum)
    loss_total = xm.mesh_reduce("train_loss", loss_sum, sum)

    return (loss_total / n_total).item(), (corr_total / n_total).item()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()

    loss_sum = torch.zeros((), device=device)
    correct  = torch.zeros((), device=device)
    n_seen   = torch.zeros((), device=device)

    para_loader = pl.MpDeviceLoader(loader, device)
    for images, labels in para_loader:
        logits = model(images)
        loss   = criterion(logits, labels)
        batch_n   = labels.new_tensor(labels.size(0), dtype=torch.float32)
        loss_sum += loss * batch_n
        correct  += (logits.argmax(1) == labels).sum().float()
        n_seen   += batch_n

    n_total    = xm.mesh_reduce("val_n",    n_seen,   sum)
    corr_total = xm.mesh_reduce("val_corr", correct,  sum)
    loss_total = xm.mesh_reduce("val_loss", loss_sum, sum)

    return (loss_total / n_total).item(), (corr_total / n_total).item()


# ── Per-process Entry (called by xmp.spawn) ───────────────────────────────────

def train_fn(index, args):
    device     = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank       = xm.get_ordinal()
    is_master  = xm.is_master_ordinal()

    torch.manual_seed(args.seed)

    if is_master:
        print("=" * 60)
        print(f"  OASIS — TPU Pod  |  {args.dataset.upper()}")
        print(f"  World size: {world_size} chips")
        print(f"  Per-chip batch: {args.batch_size}  |  "
              f"Effective batch: {args.batch_size * world_size}")
        mode = "OASIS-Calibrated (frozen bases)" if not args.no_compress else "Baseline"
        print(f"  Mode: {mode}")
        print("=" * 60, flush=True)

    train_loader, val_loader = build_loaders(args, world_size, rank)

    num_classes = 100 if args.dataset == "cifar100" else 10
    model     = ResNet18CIFAR(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    compressor = None
    if not args.no_compress:
        compressor = load_compressor(args.rank_table, args.bases_file, device)
        if is_master:
            print(f"  Compressor: {len(compressor.bases)} layers with frozen bases", flush=True)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(
            epoch, model, train_loader, device, criterion, optimizer, compressor, args
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        if is_master:
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Tr Loss={tr_loss:.4f} Acc={tr_acc*100:.2f}% | "
                f"Val Loss={val_loss:.4f} Acc={val_acc*100:.2f}% | "
                f"LR={optimizer.param_groups[0]['lr']:.2e} | "
                f"Time={elapsed:.1f}s",
                flush=True,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  ★ New best val acc: {val_acc*100:.2f}%", flush=True)

    if is_master:
        print(f"\n[OASIS] Done. Best val acc: {best_val_acc*100:.2f}%", flush=True)


def train_fn_entry(index, args):
    try:
        train_fn(index, args)
    except Exception:
        print(f"[rank {index}] Unhandled exception:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


# ── Argument Parsing & Entry Point ───────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OASIS TPU Pod Trainer")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=128,
                   help="Per-chip batch size")
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=5e-4)
    p.add_argument("--data-dir",      type=str,   default="./data")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--dataset",       type=str,   default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--nprocs",        type=int,   default=8,
                   help="TPU chips per host (8 for v4)")
    # Compression
    p.add_argument("--rank-table",    type=str,   default=None)
    p.add_argument("--bases-file",    type=str,   default=None)
    p.add_argument("--no-compress",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.no_compress and (args.rank_table is None or args.bases_file is None):
        raise ValueError(
            "--rank-table and --bases-file are required.\n"
            "Run: python3 calibrate_ranks.py --dataset cifar100 "
            "--out ranks_cifar100.json --out-bases bases_cifar100.pt"
        )

    xmp.spawn(train_fn_entry, args=(args,), nprocs=args.nprocs, start_method="spawn")
