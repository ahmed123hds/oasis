#!/usr/bin/env python3
"""
parse_results.py — Post-ablation result parser for OASIS.

Usage (run after run_ablations.sh completes):
    python3 parse_results.py

Reads all log_*.txt files in the current directory, extracts key metrics,
groups multi-seed runs, computes mean±std, and prints a final paper table.
"""

import re
import glob
import math
from pathlib import Path
from collections import defaultdict

# ─── Regex patterns ───────────────────────────────────────────────────────────

RE_VAL_ACC       = re.compile(r"Val\s+.*?Acc:\s+([\d.]+)%")
RE_BEST_VAL      = re.compile(r"Best Val Acc\s*[:\|]\s*([\d.]+)%")
RE_EPOCH_TIME    = re.compile(r"Epoch time\s+([\d.]+)s")
RE_COMPUTE_TIME  = re.compile(r"Compute time\s+([\d.]+)s")
RE_GRAD_RATIO    = re.compile(r"Overall grad ratio\s*:\s*([\d.]+)×")
RE_AVG_RANK      = re.compile(r"Avg rank selected\s*:\s*([\d.]+)")
RE_BYTES_SAVED   = re.compile(r"Grad bytes saved\s*:.*?\(run total:\s*([\d.]+)\s*MB\)")
RE_COSINE        = re.compile(r"Avg cosine sim\s*:\s*([\d.]+)")

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _mean_std(vals):
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def _fmt(mean, std, fmt=".2f", unit=""):
    if mean is None:
        return "—"
    if std == 0 or std is None:
        return f"{mean:{fmt}}{unit}"
    return f"{mean:{fmt}}±{std:{fmt}}{unit}"


def parse_log(path: str) -> dict:
    """Extract metrics from a single training log file."""
    text = Path(path).read_text(errors="replace")
    text = ANSI_ESCAPE.sub('', text)

    val_accs      = [float(x) for x in RE_VAL_ACC.findall(text)]
    best_val_line = RE_BEST_VAL.search(text)
    best_val      = float(best_val_line.group(1)) if best_val_line else (max(val_accs) if val_accs else None)

    epoch_times   = [float(x) for x in RE_EPOCH_TIME.findall(text)]
    compute_times = [float(x) for x in RE_COMPUTE_TIME.findall(text)]
    grad_ratios   = [float(x) for x in RE_GRAD_RATIO.findall(text)]
    avg_ranks     = [float(x) for x in RE_AVG_RANK.findall(text)]
    cosines       = [float(x) for x in RE_COSINE.findall(text)]
    bytes_saved   = RE_BYTES_SAVED.findall(text)

    total_wall    = sum(epoch_times) if epoch_times else None
    total_compute = sum(compute_times) if compute_times else None
    avg_ratio     = (sum(grad_ratios) / len(grad_ratios)) if grad_ratios else None
    avg_rank      = (sum(avg_ranks) / len(avg_ranks)) if avg_ranks else None
    avg_cos       = (sum(cosines) / len(cosines)) if cosines else None
    run_total_mb  = float(bytes_saved[-1]) if bytes_saved else None

    return {
        "best_val_acc":   best_val,
        "total_wall_s":   total_wall,
        "total_compute_s": total_compute,
        "avg_grad_ratio": avg_ratio,
        "avg_rank":       avg_rank,
        "avg_cosine":     avg_cos,
        "bytes_saved_mb": run_total_mb,
    }


# ─── Log file → experiment group mapping ─────────────────────────────────────

GROUPS = {
    # label: [list of glob patterns], description
    "Baseline":               (["log_baseline.txt", "log_baseline_seed*.txt"],          "No compression"),
    "OASIS-Track (Adaptive)": (["log_track_adaptive.txt", "log_track_adaptive_seed*.txt"], "energy_tau=0.97, r_max=72"),
    "OASIS-Track r=20":       (["log_track_fixed_r20.txt"],                              "Fixed rank=20"),
    "OASIS-Track r=42":       (["log_track_fixed_r42.txt"],                              "Fixed rank=42 (~mean)"),
    "OASIS-Track r=72":       (["log_track_fixed_r72.txt"],                              "Fixed rank=72 (max)"),
    "Exact K=1 (Opt-Aware)":  (["log_exact_optaware.txt"],                               "Full SVD, optimizer-aware"),
    "Exact K=1 (Energy)":     (["log_exact_energy.txt"],                                 "Full SVD, energy criterion"),
    "Baseline CIFAR-100":     (["log_baseline_cifar100.txt", "log_baseline_cifar100_seed*.txt"], "CIFAR-100, no compression"),
    "OASIS-Track CIFAR-100":  (["log_track_cifar100.txt", "log_track_cifar100_seed*.txt"],       "CIFAR-100, adaptive track"),
}


def collect_group(patterns):
    """Collect all parsed metrics for a group of log file patterns."""
    results = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            m = parse_log(f)
            if m["best_val_acc"] is not None:
                results.append(m)
    return results


def aggregate(runs: list) -> dict:
    """Mean ± std across multiple seeds/runs."""
    if not runs:
        return {}
    keys = runs[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        out[k] = _mean_std(vals)
    return out


# ─── Print table ──────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
CYAN  = "\033[96m"
RESET = "\033[0m"
DIM   = "\033[2m"
SEP   = "─" * 120


def print_table(groups_data: dict, baseline_wall: float):
    hdr = (
        f"{'Method':<30} {'Config':<28} {'Seeds':>5} "
        f"{'Best Val Acc':>16} {'Wall Time (s)':>15} "
        f"{'Compute Time (s)':>18} {'Overhead':>9} "
        f"{'Grad Ratio':>11} {'Avg Rank':>9} "
        f"{'Cosine':>8} {'Saved (MB)':>11}"
    )
    print()
    print(f"{BOLD}{CYAN}{SEP}{RESET}")
    print(f"{BOLD}{CYAN}  OASIS Ablation Results — Final Table{RESET}")
    print(f"{BOLD}{CYAN}{SEP}{RESET}")
    print(f"{BOLD}{hdr}{RESET}")
    print(SEP)

    for label, (agg, desc, n) in groups_data.items():
        if not agg:
            print(f"  {label:<30}  {DIM}(no log found){RESET}")
            continue

        acc_m, acc_s   = agg.get("best_val_acc", (None, None))
        wall_m, wall_s = agg.get("total_wall_s", (None, None))
        comp_m, comp_s = agg.get("total_compute_s", (None, None))
        rat_m, _       = agg.get("avg_grad_ratio", (None, None))
        rnk_m, _       = agg.get("avg_rank", (None, None))
        cos_m, _       = agg.get("avg_cosine", (None, None))
        byt_m, byt_s   = agg.get("bytes_saved_mb", (None, None))

        overhead = f"{wall_m / baseline_wall:.2f}×" if (wall_m and baseline_wall) else "—"

        print(
            f"  {label:<30} {desc:<28} {n:>5} "
            f"  {_fmt(acc_m, acc_s, '.2f', '%'):>16}"
            f"  {_fmt(wall_m, wall_s, '.1f'):>15}"
            f"  {_fmt(comp_m, comp_s, '.1f'):>18}"
            f"  {overhead:>9}"
            f"  {_fmt(rat_m, None, '.2f', '×'):>11}"
            f"  {_fmt(rnk_m, None, '.1f'):>9}"
            f"  {_fmt(cos_m, None, '.4f'):>8}"
            f"  {_fmt(byt_m, byt_s, '.1f', ' MB'):>11}"
        )

    print(f"{BOLD}{CYAN}{SEP}{RESET}")
    print()
    print(f"{DIM}Notes:{RESET}")
    print(f"  {DIM}• Wall Time  = total run wall-clock (logger prints excluded){RESET}")
    print(f"  {DIM}• Compute Time = fwd+bwd+compress+optim (CUDA sync'd){RESET}")
    print(f"  {DIM}• Overhead   = Wall Time / Baseline Wall Time{RESET}")
    print(f"  {DIM}• Grad Ratio = avg compressed gradient ratio across compressed layers{RESET}")
    print(f"  {DIM}• Cosine     = avg preconditioned cosine similarity (1.0 = lossless){RESET}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    groups_data = {}
    baseline_wall = None

    for label, (patterns, desc) in GROUPS.items():
        runs = collect_group(patterns)
        agg  = aggregate(runs)
        groups_data[label] = (agg, desc, len(runs))
        if label == "Baseline" and agg:
            m, _ = agg.get("total_wall_s", (None, None))
            baseline_wall = m

    if baseline_wall is None:
        print("⚠  Baseline log not found — overhead column will show '—'")

    print_table(groups_data, baseline_wall)


if __name__ == "__main__":
    main()
