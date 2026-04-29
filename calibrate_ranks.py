#!/usr/bin/env python3
"""
OASIS Rank Calibration Script
Runs a short warmup phase (e.g., 200 batches) on GPU/CPU to estimate the
intrinsic subspace rank for each compressible layer using the energy criterion.
Saves the median rank per layer to a JSON file to be loaded by the static-graph
TPU training script (OASIS-Calibrated Track).
"""
import argparse
import json
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from pathlib import Path

from oasis.compressor import OASISCompressor
from oasis.model import ResNet18CIFAR
from train_cifar10 import build_loaders

def main():
    p = argparse.ArgumentParser(description="OASIS Rank Calibration")
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--energy-tau", type=float, default=0.97)
    p.add_argument("--r-max", type=int, default=72)
    p.add_argument("--steps", type=int, default=200, help="Number of calibration batches")
    p.add_argument("--out", type=str, default="ranks_cifar100.json")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting rank calibration on {device} for {args.steps} steps...")

    train_loader, _ = build_loaders(args)
    num_classes = 100 if args.dataset == "cifar100" else 10
    model = ResNet18CIFAR(num_classes=num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    compressor = OASISCompressor(
        mode="track",
        r_max=args.r_max,
        energy_tau=args.energy_tau,
        skip_1d=True,
        skip_classifier=True
    )

    model.train()
    rank_history = defaultdict(list)

    for step, (images, labels) in enumerate(train_loader, start=1):
        if step > args.steps:
            break
            
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        
        # We set compute_metrics=True so that rank_selected is populated in stats
        compressor.compress_model(model, optimizer, step=step, compute_metrics=True)
        optimizer.step()

        for name, stats in compressor.last_stats.items():
            if "rank_selected" in stats:
                try:
                    rank_history[name].append(int(stats["rank_selected"]))
                except (ValueError, TypeError):
                    pass
            
        if step % 50 == 0:
            print(f"  Calibration step {step}/{args.steps}...")

    # Compute median rank per layer
    final_ranks = {}
    print("\nCalibration complete. Assigned ranks:")
    for name, ranks in rank_history.items():
        median_rank = int(statistics.median(ranks))
        final_ranks[name] = median_rank
        print(f"  {name}: {median_rank}")

    with open(args.out, 'w') as f:
        json.dump(final_ranks, f, indent=2)
    
    print(f"\nSaved rank table to {args.out}")
    print(f"You can now run TPU training with: --mode calibrated-track --rank-table {args.out}")

if __name__ == "__main__":
    main()
