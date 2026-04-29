#!/usr/bin/env python3
"""
train_baseline.py

Vanilla AdamW baseline for OASIS comparison.

Identical model / optimizer / scheduler / data pipeline to train_cifar10.py
but with ZERO gradient compression. Use results to compare:

  - Val accuracy  (does OASIS match baseline?)
  - Memory usage  (does OASIS save memory?)
  - Training time (what is the OASIS overhead?)

Run:
    source ~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/activate
    python train_baseline.py [same flags as train_cifar10.py]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))

from oasis.model import ResNet18CIFAR, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers (same palette as OASISLogger)
# ─────────────────────────────────────────────────────────────────────────────

_R = "\033[0m"
_BOLD    = "\033[1m"
_GREEN   = "\033[32m"
_CYAN    = "\033[36m"
_YELLOW  = "\033[33m"
_MAGENTA = "\033[35m"
_DIM     = "\033[2m"
_BLUE    = "\033[34m"


def _fmt_bytes(n):
    if n < 1024**2:   return f"{n/1024:.1f} KB"
    elif n < 1024**3: return f"{n/1024**2:.2f} MB"
    else:             return f"{n/1024**3:.3f} GB"


def _bar(frac, width=15, fill="█", empty="░"):
    filled = int(round(frac * width))
    return fill * filled + empty * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OASIS Baseline (vanilla AdamW, no compression)")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--log-every",    type=int,   default=50)
    p.add_argument("--data-dir",     type=str,   default="./data")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--dataset",      type=str,   default="cifar10", choices=["cifar10", "cifar100"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data  (identical to OASIS run)
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def build_loaders(data_dir, batch_size, num_workers, dataset="cifar10"):
    is_cifar100 = dataset == "cifar100"
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
    train_ds = ds_class(data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = ds_class(data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = n = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        total_acc  += accuracy(logits, labels)
        n += 1
    return total_loss / n, total_acc / n


# ─────────────────────────────────────────────────────────────────────────────
# Header / epoch printers
# ─────────────────────────────────────────────────────────────────────────────

def print_header(config):
    print()
    print(f"{_BOLD}{_CYAN}{'═'*70}{_R}")
    print(f"{_BOLD}{_CYAN}  BASELINE  │  Vanilla AdamW — No Gradient Compression{_R}")
    print(f"{_CYAN}{'═'*70}{_R}")
    for k, v in config.items():
        print(f"  {_YELLOW}{k:<30}{_R} {v}")
    print(f"{_CYAN}{'─'*70}{_R}\n")


def print_epoch_summary(epoch, total_epochs, train_loss, train_acc,
                         val_loss, val_acc, epoch_time, device,
                         train_compute_time=None):
    print(f"\n{_BOLD}{_BLUE}│{_R}  ── Epoch {epoch}/{total_epochs} Summary ─────────────────────────────────────")
    print(f"{_BOLD}{_BLUE}│{_R}  {'Train':>10}  Loss: {_GREEN}{train_loss:.4f}{_R}  Acc: {_GREEN}{train_acc*100:.2f}%{_R}")
    print(f"{_BOLD}{_BLUE}│{_R}  {'Val':>10}  Loss: {_CYAN}{val_loss:.4f}{_R}  Acc: {_CYAN}{val_acc*100:.2f}%{_R}")
    print(f"{_BOLD}{_BLUE}│{_R}  {'Epoch time':>10}  {epoch_time:.1f}s  \033[2m(wall-clock)\033[0m")
    if train_compute_time is not None:
        print(f"{_BOLD}{_BLUE}│{_R}  {'Compute time':>10}  \033[33m{train_compute_time:.1f}s\033[0m  "
              f"\033[2m(fwd+bwd+optim, CUDA sync'd)\033[0m")

    # Memory
    print(f"{_BOLD}{_BLUE}│{_R}  ── Memory Report ──────────────────────────────────────")
    if device.type == "cuda":
        alloc    = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak     = torch.cuda.max_memory_allocated(device)
        print(f"{_BOLD}{_BLUE}│{_R}  {_YELLOW}CUDA allocated      :{_R} {_fmt_bytes(alloc)}")
        print(f"{_BOLD}{_BLUE}│{_R}  {_YELLOW}CUDA reserved       :{_R} {_fmt_bytes(reserved)}")
        print(f"{_BOLD}{_BLUE}│{_R}  {_YELLOW}CUDA peak alloc     :{_R} {_fmt_bytes(peak)}")
        torch.cuda.reset_peak_memory_stats(device)
    else:
        try:
            import psutil, os
            rss = psutil.Process(os.getpid()).memory_info().rss
            print(f"{_BOLD}{_BLUE}│{_R}  {_YELLOW}Process RSS         :{_R} {_fmt_bytes(rss)}")
        except ImportError:
            pass

    print(f"{_BOLD}{_BLUE}└{'─'*65}┘{_R}")


def print_final(best_val_acc, best_epoch, history, total_time):
    print()
    print(f"{_BOLD}{_CYAN}{'═'*70}{_R}")
    print(f"{_BOLD}{_CYAN}  BASELINE Training Complete{_R}")
    print(f"{_CYAN}{'─'*70}{_R}")
    print(f"  {_YELLOW}Best Val Acc      :{_R} {_BOLD}{_GREEN}{best_val_acc*100:.2f}%{_R}  (epoch {best_epoch})")
    print(f"  {_YELLOW}Total wall time  :{_R} {total_time:.1f}s")
    print()
    print(f"  {_DIM}{'Ep':>4} {'TrLoss':>9} {'TrAcc':>8} {'VaLoss':>9} {'VaAcc':>8}{_R}")
    for h in history:
        marker = f" {_GREEN}←{_R}" if h["epoch"] == best_epoch else ""
        print(
            f"  {h['epoch']:>4} "
            f"{h['train_loss']:>9.4f} "
            f"{h['train_acc']*100:>7.2f}% "
            f"{h['val_loss']:>9.4f} "
            f"{h['val_acc']*100:>7.2f}%"
            f"{marker}"
        )
    print(f"{_CYAN}{'═'*70}{_R}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )

    print(f"{_DIM}Loading {args.dataset.upper()} from {args.data_dir} …{_R}")
    train_loader, val_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers, args.dataset)

    num_classes = 100 if args.dataset == "cifar100" else 10
    model     = ResNet18CIFAR(num_classes=num_classes).to(device)
    n_params  = count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print_header({
        "Device"        : str(device),
        "Model"         : f"ResNet-18 (CIFAR-10), {n_params:,} parameters",
        "Dataset"       : "CIFAR-10  (50 000 train / 10 000 val)",
        "Batch size"    : args.batch_size,
        "Optimizer"     : f"AdamW  lr={args.lr}  wd={args.weight_decay}",
        "Scheduler"     : f"CosineAnnealingLR  T_max={args.epochs}",
        "Epochs"        : args.epochs,
        "Compression"   : "NONE  (vanilla baseline)",
        "Seed"          : args.seed,
    })

    history      = []
    best_val_acc = 0.0
    best_epoch   = 0
    train_start  = time.time()
    total_epochs = args.epochs

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        model.train()

        print(f"\n{_BOLD}{_BLUE}┌─── Epoch {epoch}/{total_epochs}"
              f"  (run time: {time.time()-train_start:.1f}s) ───────────────────┐{_R}")

        running_loss = running_acc = n_batches = 0

        train_compute_time = 0.0
        use_cuda = device.type == "cuda"
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Pure compute ─────────────────────────────────────────────────
            if use_cuda: torch.cuda.synchronize()
            _t0 = time.perf_counter()

            logits = model(images)
            loss   = criterion(logits, labels)
            acc    = accuracy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_cuda: torch.cuda.synchronize()
            train_compute_time += time.perf_counter() - _t0
            # ─────────────────────────────────────────────────────────────────

            running_loss += loss.item()
            running_acc  += acc
            n_batches    += 1

            if step % args.log_every == 0 or step == len(train_loader):
                avg_loss = running_loss / n_batches
                avg_acc  = running_acc  / n_batches
                lr       = optimizer.param_groups[0]["lr"]
                progress = step / len(train_loader)
                bar      = _bar(progress)
                print(
                    f"  {_DIM}Step {step:>4}/{len(train_loader)}{_R} "
                    f"[{_GREEN}{bar}{_R}]  "
                    f"Loss: {_BOLD}{avg_loss:.4f}{_R}  "
                    f"Acc: {_BOLD}{avg_acc*100:.2f}%{_R}  "
                    f"LR: {_DIM}{lr:.2e}{_R}"
                )

        scheduler.step()

        train_loss = running_loss / n_batches
        train_acc  = running_acc  / n_batches
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        print_epoch_summary(
            epoch, total_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            epoch_time, device,
            train_compute_time=train_compute_time,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), "baseline_best.pth")

    print_final(best_val_acc, best_epoch, history, time.time() - train_start)


if __name__ == "__main__":
    main()
