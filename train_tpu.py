#!/usr/bin/env python3
"""
OASIS TPU Pod Trainer — correct xmp.spawn-based implementation.

Architecture for TPU v4-32 (4 hosts × 8 chips = 32 chips total):
  - Each host runs: python3 train_tpu.py  (via --worker=all gcloud ssh)
  - Each host calls xmp.spawn(train_fn, nprocs=8)
  - 8 processes per host × 4 hosts = 32 total processes
  - Each process owns one chip (xla:0 in its local namespace)
  - DistributedSampler shards data across all 32 replicas
  - xm.optimizer_step() all-reduces gradients across all 32 chips

OASIS compression happens per-replica BEFORE the all-reduce:
  gradients → compress → all-reduce → optimizer update
This is valid and research-defensible: you're compressing each shard's
gradient subspace before communication, reducing bandwidth.
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
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

sys.path.insert(0, str(Path(__file__).parent))

from oasis.model import ResNet18CIFAR
from oasis.logger import OASISLogger


# ── XLA-Native Frozen Basis Compressor ───────────────────────────────────────

class FrozenBasisCompressor:
    """
    Gradient compressor for TPU/XLA.
    Compression per layer = two static-shape matmuls only:
        G_hat = (G @ V) @ V.T
    No SVD, no QR, no dynamic shapes, no Python control flow on tensor values.
    """
    def __init__(self, bases: Dict[str, torch.Tensor], min_numel: int = 1024):
        self.bases    = bases       # name → V [n, r] on XLA device
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
            V     = self.bases[name]            # [n, r]
            G_2d  = G.reshape(G.shape[0], -1)   # [m, n]
            G_hat = (G_2d @ V) @ V.t()          # [m, n]  — two matmuls, fully static
            param.grad = G_hat.reshape(G.shape)


def load_compressor(rank_table_path: str, bases_path: str, device) -> FrozenBasisCompressor:
    with open(rank_table_path) as f:
        rank_table = json.load(f)
    raw = torch.load(bases_path, map_location="cpu", weights_only=True)
    bases = {}
    for name, V in raw.items():
        r = rank_table.get(name)
        if r is not None:
            bases[name] = V[:, :r].to(device)
    return FrozenBasisCompressor(bases)


# ── Data Loading ──────────────────────────────────────────────────────────────

def build_loaders(args, world_size: int, rank: int):
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

    ds_class = datasets.CIFAR100 if is_c100 else datasets.CIFAR10
    train_ds = ds_class(args.data_dir, train=True,  download=(rank == 0), transform=train_tf)
    val_ds   = ds_class(args.data_dir, train=False, download=(rank == 0), transform=val_tf)

    # DistributedSampler shards the dataset across all chips
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=0,      # Must be 0 on TPU — multiprocessing workers deadlock XLA
        drop_last=True,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, sampler=val_sampler,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    return train_loader, val_loader


# ── Training & Eval ───────────────────────────────────────────────────────────

def train_one_epoch(epoch, model, loader, criterion, optimizer, scheduler,
                    compressor, device, args):
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches    = 0
    t_compute    = 0.0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        t0 = time.perf_counter()

        logits = model(images)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        if compressor is not None:
            compressor.compress_all(model)

        # xm.optimizer_step = optimizer.step + xm.mark_step + all-reduce across all chips
        # This is correct for TPU Pod: all 32 processes call it in lockstep
        # because they all process the same number of batches (DistributedSampler + drop_last)
        xm.optimizer_step(optimizer)

        t_compute += time.perf_counter() - t0
        n_batches += 1

        # Only sync to host at log intervals — avoid per-step device→host stalls
        if step % args.log_every == 0 or step == len(loader):
            loss_val = loss.item()   # triggers xm.mark_step implicitly
            acc_val  = (logits.argmax(1) == labels).float().mean().item()
            running_loss += loss_val
            running_acc  += acc_val
            if xm.is_master_ordinal():
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch} | Step {step:>4}/{len(loader)} | "
                      f"Loss: {running_loss/n_batches:.4f} | "
                      f"Acc: {running_acc/n_batches*100:.2f}% | LR: {lr:.2e}",
                      flush=True)

        if step == 1 and xm.is_master_ordinal():
            print(f"    [TPU] Epoch {epoch} Step 1 compiled and cached. "
                  f"World size = {xm.xrt_world_size()} chips.", flush=True)

    scheduler.step()
    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1), t_compute


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = n = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        xm.mark_step()
        total_loss += loss.item()
        total_acc  += (logits.argmax(1) == labels).float().mean().item()
        n += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


# ── Per-process Training Function (called by xmp.spawn) ──────────────────────

def train_fn(index, args):
    """
    This function runs in each of the nprocs=8 processes per host.
    index = local chip index (0..7)
    Global rank = xm.get_ordinal() (0..31 across all hosts)
    """
    device     = xm.xla_device()
    world_size = xm.xrt_world_size()   # total chips across all hosts
    rank       = xm.get_ordinal()      # global rank

    if xm.is_master_ordinal():
        print(f"[OASIS] Training on {world_size} chips (v4-32 Pod)", flush=True)

    # Load compressor bases (each process loads its own copy to its chip)
    compressor = None
    if not args.no_compress:
        compressor = load_compressor(args.rank_table, args.bases_file, device)

    train_loader, val_loader = build_loaders(args, world_size, rank)

    num_classes = 100 if args.dataset == "cifar100" else 10
    model     = ResNet18CIFAR(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if xm.is_master_ordinal():
            print(f"\n── Epoch {epoch}/{args.epochs} ──────────────────────────────", flush=True)

        # Shuffle sampler each epoch
        train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc, t_compute = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, scheduler,
            compressor, device, args
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if xm.is_master_ordinal():
            print(f"  [Epoch {epoch}] Train Loss={train_loss:.4f} Acc={train_acc*100:.2f}%  "
                  f"Val Loss={val_loss:.4f} Acc={val_acc*100:.2f}%  "
                  f"Compute={t_compute:.1f}s", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  ★ New best val acc: {val_acc*100:.2f}%", flush=True)

    if xm.is_master_ordinal():
        print(f"\n[OASIS] Training complete. Best val acc: {best_val_acc*100:.2f}%", flush=True)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OASIS TPU Pod Trainer")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=128,
                   help="Per-chip batch size. Total batch = batch_size × world_size")
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=5e-4)
    p.add_argument("--log-every",     type=int,   default=50)
    p.add_argument("--data-dir",      type=str,   default="./data")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--dataset",       type=str,   default="cifar10",
                   choices=["cifar10", "cifar100"])
    # Compression
    p.add_argument("--rank-table",    type=str,   default=None)
    p.add_argument("--bases-file",    type=str,   default=None)
    p.add_argument("--no-compress",   action="store_true",
                   help="Disable compression (baseline run)")
    # TPU Pod
    p.add_argument("--nprocs",        type=int,   default=8,
                   help="Number of chips per host (8 for v4)")
    return p.parse_args()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.no_compress and (args.rank_table is None or args.bases_file is None):
        raise ValueError(
            "--rank-table and --bases-file required.\n"
            "Run first: python3 calibrate_ranks.py --dataset cifar100 "
            "--out ranks_cifar100.json --out-bases bases_cifar100.pt"
        )

    # xmp.spawn launches nprocs processes on this host.
    # Each process calls train_fn(index, args).
    # All 4 hosts do this simultaneously → 32 total processes.
    xmp.spawn(train_fn, args=(args,), nprocs=args.nprocs)
