"""
oasis/logger.py

Verbose logging for OASIS training.

Formats per-epoch summaries with:
  - Training loss / accuracy
  - Validation loss / accuracy
  - Per-layer compression stats (rank, cosine-sim, ratio, memory saved)
  - Aggregate memory savings
  - Running wall-clock time
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA= "\033[35m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_WHITE  = "\033[37m"
_DIM    = "\033[2m"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024**2:
        return f"{n/1024:.1f} KB"
    elif n < 1024**3:
        return f"{n/1024**2:.2f} MB"
    else:
        return f"{n/1024**3:.3f} GB"


def _bar(frac: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    filled = int(round(frac * width))
    return fill * filled + empty * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Logger class
# ─────────────────────────────────────────────────────────────────────────────

class OASISLogger:
    """
    Handles all console output for the OASIS smoke-test training run.
    """

    def __init__(self, total_epochs: int, steps_per_epoch: int, log_every_n_steps: int = 50):
        self.total_epochs      = total_epochs
        self.steps_per_epoch   = steps_per_epoch
        self.log_every_n_steps = log_every_n_steps

        self._epoch_start_time = 0.0
        self._train_start_time = time.time()
        self._print_overhead   = 0.0   # accumulated logger print time, excluded from epoch time

        # Running totals across the whole run
        self._total_bytes_saved_run = 0

    # ── top-level banners ─────────────────────────────────────────────────────

    def print_header(self, config: dict):
        print()
        print(f"{_BOLD}{_CYAN}{'═'*70}{_RESET}")
        print(f"{_BOLD}{_CYAN}  OASIS  │  Optimizer-Aware Subspace Importance Selection{_RESET}")
        print(f"{_CYAN}{'═'*70}{_RESET}")
        for k, v in config.items():
            print(f"  {_YELLOW}{k:<30}{_RESET} {v}")
        print(f"{_CYAN}{'─'*70}{_RESET}")
        print()

    def epoch_start(self, epoch: int):
        self._epoch_start_time = time.time()
        self._print_overhead   = 0.0          # reset per-epoch logger overhead
        elapsed_total = time.time() - self._train_start_time
        print(f"\n{_BOLD}{_BLUE}┌─── Epoch {epoch}/{self.total_epochs}"
              f"  (run time: {elapsed_total:.1f}s) ───────────────────┐{_RESET}")

    def step_log(
        self,
        epoch: int,
        step: int,
        loss: float,
        acc: float,
        lr: float,
        compression_stats: Optional[Dict[str, dict]] = None,
    ):
        """Print a training-step line. Time is tracked and excluded from epoch time."""
        _t0 = time.time()
        progress = step / self.steps_per_epoch
        bar = _bar(progress, width=15)
        print(
            f"  {_DIM}Step {step:>4}/{self.steps_per_epoch}{_RESET} "
            f"[{_GREEN}{bar}{_RESET}]  "
            f"Loss: {_BOLD}{loss:.4f}{_RESET}  "
            f"Acc: {_BOLD}{acc*100:.2f}%{_RESET}  "
            f"LR: {_DIM}{lr:.2e}{_RESET}"
        )

        if compression_stats:
            self._print_compression_table(compression_stats, indent=4)

        self._print_overhead += time.time() - _t0

    def epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        compression_stats: Optional[Dict[str, dict]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model: Optional[torch.nn.Module] = None,
        train_compute_time: Optional[float] = None,  # GPU-sync'd compute-only time
    ):
        # Wall-clock time (excludes logger print overhead)
        elapsed = time.time() - self._epoch_start_time - self._print_overhead
        print(f"\n{_BOLD}{_BLUE}│{_RESET}  ── Epoch {epoch} Summary ─────────────────────────────────────")
        print(f"{_BOLD}{_BLUE}│{_RESET}  {'Train':>10}  Loss: {_GREEN}{train_loss:.4f}{_RESET}  "
              f"Acc: {_GREEN}{train_acc*100:.2f}%{_RESET}")
        print(f"{_BOLD}{_BLUE}│{_RESET}  {'Val':>10}  Loss: {_CYAN}{val_loss:.4f}{_RESET}  "
              f"Acc: {_CYAN}{val_acc*100:.2f}%{_RESET}")
        print(f"{_BOLD}{_BLUE}│{_RESET}  {'Epoch time':>10}  {elapsed:.1f}s  {_DIM}(wall-clock, logger prints excluded){_RESET}")
        if train_compute_time is not None:
            print(f"{_BOLD}{_BLUE}│{_RESET}  {'Compute time':>10}  {_YELLOW}{train_compute_time:.1f}s{_RESET}  "
                  f"{_DIM}(fwd+bwd+compress+optim, CUDA sync'd){_RESET}")

        if compression_stats:
            self._print_compression_summary(compression_stats)

        if model is not None:
            self._print_memory_report(model)

        print(f"{_BOLD}{_BLUE}└{'─'*65}┘{_RESET}")

    # ── compression tables ────────────────────────────────────────────────────

    def _print_compression_table(
        self,
        stats: Dict[str, dict],
        indent: int = 0,
        max_layers: int = 6,
    ):
        """Print a concise per-layer compression table."""
        pad = " " * indent
        hdr = (
            f"{pad}{_DIM}{'Layer':<38} {'Shape':<20} {'Rank':>6} {'MaxR':>6} "
            f"{'cos':>7} {'drift':>7} {'Ratio':>7} {'Saved':>10} {'R?':>3}{_RESET}"
        )
        print(hdr)
        print(f"{pad}{_DIM}{'─'*105}{_RESET}")

        items = list(stats.items())
        if len(items) > max_layers:
            items = items[:max_layers//2] + [("...", None)] + items[-(max_layers//2):]

        for name, s in items:
            if s is None:
                print(f"{pad}  {'  ... (layers omitted for brevity) ...'}")
                continue

            rank     = s.get("rank_selected", "?")
            max_rank = s.get("max_rank", "?")
            cos      = s.get("cosine_sim", 0.0)
            drift    = s.get("drift_cos",  cos)  # falls back to cos when no caching
            ratio    = s.get("compression_ratio", 1.0)
            saved    = s.get("bytes_saved", 0)
            shape    = str(s.get("shape", ""))
            refreshed = s.get("refreshed", True)

            cos_color   = _GREEN   if cos   >= 0.98 else (_YELLOW if cos   >= 0.90 else _RED)
            drift_color = _GREEN   if drift >= 0.95 else (_YELLOW if drift >= 0.85 else _RED)
            ratio_color = _MAGENTA if ratio >  2.0  else _WHITE
            ref_marker  = f"{_CYAN}SVD{_RESET}" if refreshed else f"{_DIM}  ·{_RESET}"

            short_name = name if len(name) <= 37 else "…" + name[-36:]

            print(
                f"{pad}  {short_name:<38} {shape:<20} "
                f"{_CYAN}{str(rank):>6}{_RESET} {str(max_rank):>6} "
                f"{cos_color}{cos:>7.4f}{_RESET} "
                f"{drift_color}{drift:>7.4f}{_RESET} "
                f"{ratio_color}{ratio:>7.2f}×{_RESET} "
                f"{_GREEN}{_fmt_bytes(saved):>10}{_RESET} "
                f"{ref_marker}"
            )

    def _print_compression_summary(self, stats: Dict[str, dict]):
        """Print aggregate compression stats at epoch end."""
        # ── Partition into compressed vs skipped ────────────────────────────
        compressed_stats = {k: s for k, s in stats.items()
                            if isinstance(s, dict) and isinstance(s.get("rank_selected"), int)}
        skipped_stats    = {k: s for k, s in stats.items()
                            if isinstance(s, dict) and isinstance(s.get("rank_selected"), str)}

        n_compressed = len(compressed_stats)
        n_skipped    = len(skipped_stats)

        # Skip sub-categories
        n_skip_1d       = sum(1 for s in skipped_stats.values() if "1-D"     in str(s.get("rank_selected", "")))
        n_skip_small    = sum(1 for s in skipped_stats.values() if "params"   in str(s.get("rank_selected", "")))
        n_skip_other    = n_skipped - n_skip_1d - n_skip_small

        # Bytes / ratio (only compressed layers contribute real savings)
        total_orig  = sum(s["original_numel"]  for s in compressed_stats.values())
        total_comp  = sum(s["compressed_numel"] for s in compressed_stats.values())
        total_saved = sum(s["bytes_saved"]      for s in compressed_stats.values())
        self._total_bytes_saved_run += total_saved

        ranks   = [s["rank_selected"] for s in compressed_stats.values()]
        avg_rank = sum(ranks) / len(ranks) if ranks else 0

        cosines  = [s["cosine_sim"] for s in compressed_stats.values() if "cosine_sim" in s]
        avg_cos  = sum(cosines) / len(cosines) if cosines else 0

        drift_vals = [s["drift_cos"] for s in compressed_stats.values()
                      if "drift_cos" in s and not s.get("refreshed", True)]
        avg_drift  = sum(drift_vals) / len(drift_vals) if drift_vals else None

        n_refreshed  = sum(1 for s in compressed_stats.values() if s.get("refreshed", False))
        refresh_pct  = 100 * n_refreshed / n_compressed if n_compressed else 0
        overall_ratio = total_orig / max(total_comp, 1)

        # ── Print ────────────────────────────────────────────────────────────
        b = f"{_BOLD}{_BLUE}│{_RESET}"
        print(f"{b}  ── Compression Report ─────────────────────────────────")
        print(f"{b}  {_YELLOW}Tensors seen        :{_RESET} {n_compressed + n_skipped}  "
              f"{_DIM}(compressed: {n_compressed}  skipped: {n_skipped}){_RESET}")
        if n_skipped:
            parts = []
            if n_skip_1d:    parts.append(f"1-D/BN/bias: {n_skip_1d}")
            if n_skip_small: parts.append(f"small (<min_numel): {n_skip_small}")
            if n_skip_other: parts.append(f"other: {n_skip_other}")
            print(f"{b}  {_DIM}  └─ skip reasons    :  {',  '.join(parts)}{_RESET}")
        print(f"{b}  {_YELLOW}Avg rank selected   :{_RESET} {avg_rank:.1f}")
        print(f"{b}  {_YELLOW}Avg cosine sim      :{_RESET} {_GREEN}{avg_cos:.4f}{_RESET}")
        if avg_drift is not None:
            dc = _GREEN if avg_drift >= 0.95 else (_YELLOW if avg_drift >= 0.85 else _RED)
            print(f"{b}  {_YELLOW}Avg drift sim       :{_RESET} {dc}{avg_drift:.4f}{_RESET}  "
                  f"{_DIM}(cached steps only){_RESET}")
        # Label is "Core SVD updates" for track mode, else "SVD refresh rate"
        if refresh_pct == 100.0:
            svd_label = "Core SVD updates    "
            svd_val   = f"{n_refreshed}/{n_compressed} compressed tensors"
            svd_note  = f"  {_DIM}(tiny r×r core, no full SVD){_RESET}"
        else:
            svd_label = "SVD refresh rate    "
            svd_val   = f"{n_refreshed}/{n_compressed} ({refresh_pct:.0f}%)"
            svd_note  = ""
        
        print(f"{b}  {_YELLOW}{svd_label}:{_RESET} {_CYAN}{svd_val}{_RESET}{svd_note}")
        print(f"{b}  {_YELLOW}Overall grad ratio  :{_RESET} {_MAGENTA}{overall_ratio:.2f}×{_RESET}")
        print(f"{b}  {_YELLOW}Grad bytes saved    :{_RESET} {_GREEN}{_fmt_bytes(total_saved)}{_RESET}"
              f"  (run total: {_fmt_bytes(self._total_bytes_saved_run)})")

    def _print_memory_report(self, model: torch.nn.Module):
        """Print GPU/CPU memory stats."""
        device = next(model.parameters()).device

        print(f"{_BOLD}{_BLUE}│{_RESET}  ── Memory Report ──────────────────────────────────────")

        if device.type == "cuda":
            alloc   = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            max_alloc = torch.cuda.max_memory_allocated(device)
            print(f"{_BOLD}{_BLUE}│{_RESET}  {_YELLOW}CUDA allocated      :{_RESET} {_fmt_bytes(alloc)}")
            print(f"{_BOLD}{_BLUE}│{_RESET}  {_YELLOW}CUDA reserved       :{_RESET} {_fmt_bytes(reserved)}")
            print(f"{_BOLD}{_BLUE}│{_RESET}  {_YELLOW}CUDA peak alloc     :{_RESET} {_fmt_bytes(max_alloc)}")
            torch.cuda.reset_peak_memory_stats(device)
        else:
            try:
                import psutil, os
                proc = psutil.Process(os.getpid())
                rss = proc.memory_info().rss
                print(f"{_BOLD}{_BLUE}│{_RESET}  {_YELLOW}Process RSS         :{_RESET} {_fmt_bytes(rss)}")
            except ImportError:
                print(f"{_BOLD}{_BLUE}│{_RESET}  {_DIM}(psutil not available for CPU memory stats){_RESET}")

    # ── final summary ─────────────────────────────────────────────────────────

    def print_final_summary(
        self,
        best_val_acc: float,
        best_epoch: int,
        history: List[dict],
    ):
        total_time = time.time() - self._train_start_time
        print()
        print(f"{_BOLD}{_CYAN}{'═'*70}{_RESET}")
        print(f"{_BOLD}{_CYAN}  OASIS Training Complete{_RESET}")
        print(f"{_CYAN}{'─'*70}{_RESET}")
        print(f"  {_YELLOW}Best Val Acc        :{_RESET} {_BOLD}{_GREEN}{best_val_acc*100:.2f}%{_RESET}  (epoch {best_epoch})")
        print(f"  {_YELLOW}Total wall time     :{_RESET} {total_time:.1f}s")
        print(f"  {_YELLOW}Total grad bytes saved:{_RESET} {_BOLD}{_GREEN}{_fmt_bytes(self._total_bytes_saved_run)}{_RESET}")
        print()

        # Epoch table
        print(f"  {_DIM}{'Ep':>4} {'TrLoss':>9} {'TrAcc':>8} {'VaLoss':>9} {'VaAcc':>8}{_RESET}")
        for h in history:
            marker = " ←" if h["epoch"] == best_epoch else ""
            print(
                f"  {h['epoch']:>4} "
                f"{h['train_loss']:>9.4f} "
                f"{h['train_acc']*100:>7.2f}% "
                f"{h['val_loss']:>9.4f} "
                f"{h['val_acc']*100:>7.2f}%"
                f"{_GREEN}{marker}{_RESET}"
            )

        print(f"{_CYAN}{'═'*70}{_RESET}")
        print()
