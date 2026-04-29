#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))

from oasis.compressor import OASISCompressor
from oasis.logger import OASISLogger
from oasis.model import ResNet18CIFAR, count_parameters

def parse_args():
    p = argparse.ArgumentParser(description="OASIS CIFAR Smoke Test")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch-size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight-decay",    type=float, default=5e-4)
    p.add_argument("--tau",             type=float, default=0.98)
    p.add_argument("--r-max-fraction",  type=float, default=0.5)
    p.add_argument("--n-power-iter",    type=int,   default=2)
    p.add_argument("--no-error-feedback", action="store_true")
    p.add_argument("--compress-warmup", type=int,   default=1)
    p.add_argument("--log-every",       type=int,   default=50)
    p.add_argument("--data-dir",        type=str,   default="./data")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--device",          type=str,   default="auto")
    
    # Ablation / mode flags
    p.add_argument("--mode",              type=str,   default="exact",  choices=["exact", "track"],
                   help="exact=randomized SVD, track=OASIS-Track (online warm-started)")
    p.add_argument("--r-max",             type=int,   default=72,       help="Max rank for OASIS-Track")
    p.add_argument("--energy-tau",        type=float, default=0.97,     help="Core energy threshold for OASIS-Track rank selection")
    p.add_argument("--min-numel",         type=int,   default=1024,     help="Skip compression for tensors smaller than this")
    p.add_argument("--update-freq",       type=int,   default=1,        help="Fixed-K refresh interval (exact mode only)")
    p.add_argument("--adaptive-refresh",  action="store_true",           help="Drift-triggered refresh (exact mode only)")
    p.add_argument("--tau-drift",         type=float, default=0.95,     help="Drift trigger threshold")
    p.add_argument("--criterion",         type=str,   default="optimizer_aware",
                   choices=["optimizer_aware", "energy", "fixed"])
    p.add_argument("--fixed-rank",        type=int,   default=None)
    p.add_argument("--dataset",           type=str,   default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--skip-classifier",   action="store_true")
    return p.parse_args()


def build_loaders(args):
    is_cifar100 = args.dataset == "cifar100"
    mean = (0.5071, 0.4867, 0.4408) if is_cifar100 else (0.4914, 0.4822, 0.4465)
    std = (0.2675, 0.2565, 0.2761) if is_cifar100 else (0.2470, 0.2435, 0.2616)

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
    train_ds = ds_class(args.data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = ds_class(args.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

def train_one_epoch(epoch, model, loader, criterion, optimizer, scheduler, compressor, logger, device, args, global_step):
    model.train()
    running_loss = running_acc = n_batches = 0
    use_cuda = device.type == "cuda"

    train_compute_time = 0.0   # forward + backward + compress + optimizer (GPU sync'd)
    log_overhead_time  = 0.0   # printing / metric formatting only

    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # ── Pure compute: forward + backward + compress + optimizer ───────────
        if use_cuda: torch.cuda.synchronize()
        _t_compute = time.perf_counter()

        logits = model(images)
        loss   = criterion(logits, labels)
        acc    = accuracy(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        compression_stats = None
        if global_step[0] >= args.compress_warmup:
            compute_metrics = (step % args.log_every == 0) or (step == len(loader))
            compression_stats = compressor.compress_model(
                model, optimizer, step=global_step[0], compute_metrics=compute_metrics
            )

        optimizer.step()

        if use_cuda: torch.cuda.synchronize()
        train_compute_time += time.perf_counter() - _t_compute
        # ─────────────────────────────────────────────────────────────────────

        global_step[0] += 1
        running_loss += loss.item()
        running_acc  += acc
        n_batches    += 1

        # ── Logging (time tracked separately) ─────────────────────────────────
        if step % args.log_every == 0 or step == len(loader):
            avg_loss = running_loss / n_batches
            avg_acc  = running_acc  / n_batches
            show_stats = compression_stats if step % args.log_every == 0 else None
            logger.step_log(epoch, step, avg_loss, avg_acc, optimizer.param_groups[0]["lr"], show_stats)
            log_overhead_time += logger._print_overhead  # logger tracks its own print time
        # ─────────────────────────────────────────────────────────────────────

    scheduler.step()
    return running_loss / n_batches, running_acc / n_batches, train_compute_time


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

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    train_loader, val_loader = build_loaders(args)
    num_classes = 100 if args.dataset == "cifar100" else 10
    model = ResNet18CIFAR(num_classes=num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    compressor = OASISCompressor(
        mode=args.mode,
        tau=args.tau,             tau_drift=args.tau_drift,
        r_max=args.r_max,         energy_tau=args.energy_tau,
        r_max_fraction=args.r_max_fraction, n_power_iter=args.n_power_iter,
        use_error_feedback=not args.no_error_feedback, skip_1d=True,
        min_numel=args.min_numel,
        update_freq=args.update_freq, adaptive_refresh=args.adaptive_refresh,
        criterion=args.criterion, fixed_rank=args.fixed_rank,
        skip_classifier=args.skip_classifier,
    )

    logger = OASISLogger(args.epochs, len(train_loader), args.log_every)
    if args.mode == "track":
        mode_str = f"OASIS-Track  r_max={args.r_max}  energy_tau={args.energy_tau}"
    elif args.adaptive_refresh:
        mode_str = f"OASIS-Fast  tau_drift={args.tau_drift}"
    else:
        mode_str = f"OASIS-Exact  K={args.update_freq}"

    logger.print_header({
        "Dataset"           : args.dataset.upper(),
        "Mode"              : mode_str,
        "Criterion"         : args.criterion,
        "Tau (target)"      : args.tau,
        "Batch size"        : args.batch_size,
        "Epochs"            : args.epochs,
        "Seed"              : args.seed,
    })

    history, best_val_acc, best_epoch, global_step = [], 0.0, 0, [0]
    for epoch in range(1, args.epochs + 1):
        logger.epoch_start(epoch)
        train_loss, train_acc, train_compute_time = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, scheduler,
            compressor, logger, device, args, global_step
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.epoch_end(epoch, train_loss, train_acc, val_loss, val_acc,
                         compressor.last_stats, optimizer, model,
                         train_compute_time=train_compute_time)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch

    logger.print_final_summary(best_val_acc, best_epoch, history)

if __name__ == "__main__":
    main()
