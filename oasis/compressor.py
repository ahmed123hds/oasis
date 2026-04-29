"""
oasis/compressor.py — OASIS full compressor with 4 modes:
  exact   : randomized SVD every step (original OASIS)
  k-step  : fixed K-step caching (OASIS-K)
  fast    : drift-triggered adaptive refresh (OASIS-Fast)
  track   : online warm-started subspace tracking (OASIS-Track) ← new
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


# ── Math helpers ──────────────────────────────────────────────────────────────

def _randomized_svd(matrix, rank, n_power_iter=2, n_oversampling=10):
    m, n = matrix.shape
    r_s  = min(rank + n_oversampling, min(m, n))
    Omega = torch.randn(n, r_s, device=matrix.device, dtype=matrix.dtype)
    Y = matrix @ Omega
    for _ in range(n_power_iter):
        Y = matrix @ (matrix.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    B    = Q.T @ matrix
    Uh, S, Vt = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uh
    return U[:, :rank], S[:rank], Vt[:rank, :]


def _project(H, U, Vt):
    """Two-sided rank-r projection: U(U^T H Vt^T)Vt"""
    return U @ (U.T @ H @ Vt.T) @ Vt


def _precond_cosine(G, G_approx, D):
    a = (D * G).flatten() if D is not None else G.flatten()
    b = (D * G_approx).flatten() if D is not None else G_approx.flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _energy_frac(G, G_approx):
    return (torch.linalg.norm(G_approx, ord='fro') /
            (torch.linalg.norm(G, ord='fro') + 1e-8)).item()


# ── OASIS-Track core ──────────────────────────────────────────────────────────

def _track_compress(H_2d, G_2d, V_old, r_max, energy_tau, r_min, fixed_rank=None):
    """
    OASIS-Track: one warm-started power iteration on H, tiny core SVD.

    H_2d       : preconditioned gradient (D⊙G) or raw G  [m, n]
    G_2d       : raw gradient                             [m, n]
    V_old      : cached right basis                       [n, r_max]
    fixed_rank : if set, bypass energy threshold and always use this rank

    Returns: G_hat [m,n], rank r, new V [n, r_max]
    """
    m, n = H_2d.shape

    # Level 1: one warm power step on H (O(mn·r_max))
    U = torch.linalg.qr(H_2d @ V_old,    mode="reduced").Q   # [m, r_max]
    V = torch.linalg.qr(H_2d.T @ U,      mode="reduced").Q   # [n, r_max]

    # Level 2: tiny core SVD  (O(r_max^3) only!)
    B = U.T @ H_2d @ V                                        # [r_max, r_max]
    P, S, Qt = torch.linalg.svd(B, full_matrices=False)

    # Level 3: rank selection — adaptive (energy threshold) or fixed
    if fixed_rank is not None:
        r = max(r_min, min(fixed_rank, r_max))
    else:
        s2  = S ** 2
        cum = torch.cumsum(s2, 0) / s2.sum().clamp(min=1e-10)
        r   = int((cum < energy_tau).sum().item()) + 1
        r   = max(r_min, min(r, r_max))

    # Reconstruct in raw-gradient space via optimizer-aware basis
    A   = U  @ P[:, :r]          # [m, r]
    C   = Qt[:r, :] @ V.T        # [r, n]
    G_hat = A @ (A.T @ G_2d @ C.T) @ C

    return G_hat, r, V.detach()


# ── Compressor class ──────────────────────────────────────────────────────────

class OASISCompressor:
    """
    mode="exact"  → randomized SVD every step
    mode="track"  → OASIS-Track (recommended for speed)
    update_freq=K → fixed-K caching (used only when mode="exact")
    adaptive_refresh=True → drift-triggered refresh (mode="exact" only)
    """

    def __init__(
        self,
        mode: str = "exact",           # "exact" | "track"
        tau: float = 0.98,             # SVD cosine target  (exact mode)
        tau_drift: float = 0.95,       # drift trigger       (fast mode)
        r_max: int = 72,               # max rank            (track mode)
        energy_tau: float = 0.97,      # core energy thresh  (track mode)
        r_max_fraction: float = 0.5,   # max rank fraction   (exact mode)
        r_min: int = 1,
        n_power_iter: int = 2,
        use_error_feedback: bool = True,
        skip_1d: bool = True,
        min_numel: int = 1024,         # Skip tensors smaller than this
        update_freq: int = 1,          # K for exact+K mode
        adaptive_refresh: bool = False,
        criterion: str = "optimizer_aware",
        fixed_rank: Optional[int] = None,
        skip_classifier: bool = True,  # Skip compressing the final layer
    ):
        self.mode             = mode
        self.tau              = tau
        self.tau_drift        = tau_drift
        self.r_max            = r_max
        self.energy_tau       = energy_tau
        self.r_max_fraction   = r_max_fraction
        self.r_min            = r_min
        self.n_power_iter     = n_power_iter
        self.use_error_feedback = use_error_feedback
        self.skip_1d          = skip_1d
        self.min_numel        = min_numel
        self.update_freq      = update_freq
        self.adaptive_refresh = adaptive_refresh
        self.criterion        = criterion
        self.fixed_rank       = fixed_rank
        self.skip_classifier  = skip_classifier

        # Per-layer state
        self._error_buffers : Dict[int, torch.Tensor] = {}
        self._cached_bases  : Dict[int, Tuple] = {}
        self._cached_ranks  : Dict[int, int]   = {}
        self._track_state   : Dict[int, torch.Tensor] = {}  # pid -> V
        self._refresh_counts: Dict[int, int]   = {}

        self.last_stats: Dict[str, dict] = {}

    # ── main entry ─────────────────────────────────────────────────────────────

    def compress_gradient(self, name, param, step, v2=None, eps=1e-8, compute_metrics=False):
        G = param.grad
        if G is None:
            return None, {}

        original_shape = G.shape

        is_classifier = self.skip_classifier and ("fc." in name or "classifier" in name)

        if (G.dim() < 2 and self.skip_1d) or G.numel() < self.min_numel or is_classifier:
            if is_classifier:
                reason = "classifier"
            elif G.dim() < 2:
                reason = "1-D"
            else:
                reason = f"<{self.min_numel} params"

            # Only update last_stats when metrics are requested (avoids stomping on
            # cached stats with empty dicts on non-log steps)
            if compute_metrics:
                stats = {"name": name, "shape": tuple(original_shape),
                         "rank_selected": f"skipped ({reason})", "max_rank": "–",
                         "cosine_sim": 1.0, "drift_cos": 1.0, "refreshed": False,
                         "compression_ratio": 1.0, "original_numel": G.numel(),
                         "compressed_numel": G.numel(), "bytes_saved": 0}
                self.last_stats[name] = stats
            return G, self.last_stats.get(name, {})

        G_2d = G.reshape(G.shape[0], -1) if G.dim() > 2 else G
        m, n  = G_2d.shape
        pid   = id(param)

        # Error feedback
        if self.use_error_feedback:
            if pid not in self._error_buffers:
                self._error_buffers[pid] = torch.zeros_like(G_2d)
            H_ef = G_2d + self._error_buffers[pid]
        else:
            H_ef = G_2d

        # Preconditioner
        if v2 is not None and self.criterion == "optimizer_aware":
            v2_2d = v2.reshape(m, n) if v2.dim() > 2 else v2
            D = 1.0 / (v2_2d.sqrt() + eps)
            H_2d = D * H_ef   # optimizer-aware tracking signal
        else:
            D    = None
            H_2d = H_ef

        # ── Dispatch ──────────────────────────────────────────────────────────
        if self.mode == "track":
            compressed, best_rank, refreshed, drift_cos = \
                self._compress_track(H_2d, H_ef, pid, m, n)
            
            best_metric = _precond_cosine(H_ef, compressed, D) if compute_metrics else 1.0
        else:
            compressed, best_rank, refreshed, drift_cos, best_metric = \
                self._compress_exact(H_ef, D, pid, m, n, step, compute_metrics=compute_metrics)

        # Error feedback update
        if self.use_error_feedback:
            self._error_buffers[pid] = (H_ef - compressed).detach()

        param.grad = compressed.reshape(original_shape)

        if not compute_metrics:
            return param.grad, {}

        orig_numel       = G.numel()
        compressed_numel = best_rank * (m + 1 + n)
        bytes_saved      = max(0, orig_numel - compressed_numel) * G.element_size()

        stats = {
            "name": name, "shape": tuple(original_shape),
            "rank_selected": best_rank,
            "max_rank": min(int(self.r_max_fraction * min(m,n)), min(m,n)) if self.mode == "exact" else self.r_max,
            "cosine_sim": round(best_metric, 6),
            "drift_cos":  round(drift_cos, 6),
            "refreshed":  refreshed,
            "compression_ratio": round(orig_numel / max(compressed_numel, 1), 3),
            "original_numel": orig_numel,
            "compressed_numel": compressed_numel,
            "bytes_saved": bytes_saved,
            "svd_calls": self._refresh_counts.get(pid, 0),
        }
        self.last_stats[name] = stats
        return param.grad, stats

    # ── OASIS-Track ───────────────────────────────────────────────────────────

    def _compress_track(self, H_2d, G_2d, pid, m, n):
        r_max = min(self.r_max, min(m, n))

        # Initialize V with orthonormal random matrix
        if pid not in self._track_state:
            V0 = torch.linalg.qr(
                torch.randn(n, r_max, device=H_2d.device, dtype=H_2d.dtype)
            ).Q
            self._track_state[pid] = V0

        V_old = self._track_state[pid]
        self._refresh_counts[pid] = self._refresh_counts.get(pid, 0) + 1

        compressed, r, V_new = _track_compress(
            H_2d, G_2d, V_old, r_max, self.energy_tau, self.r_min,
            fixed_rank=self.fixed_rank   # None → adaptive; int → fixed-rank ablation
        )
        self._track_state[pid] = V_new

        return compressed, r, True, 1.0   # always "refreshes" (continuous tracking)

    # ── OASIS-Exact (original) ────────────────────────────────────────────────

    def _compress_exact(self, H_ef, D, pid, m, n, step, compute_metrics=False):
        max_rank = max(self.r_min, int(self.r_max_fraction * min(m, n)))
        max_rank = min(max_rank, min(m, n))

        def _metric(approx):
            if not compute_metrics: return 1.0
            if self.criterion == "optimizer_aware":
                return _precond_cosine(H_ef, approx, D)
            elif self.criterion == "energy":
                return _energy_frac(H_ef, approx)
            return 1.0

        def _svd_approx(r):
            U, S, Vt = _randomized_svd(H_ef, r, self.n_power_iter)
            return (U * S.unsqueeze(0)) @ Vt, U, Vt

        need_refresh = True
        drift_cos    = 0.0
        compressed   = None

        if pid in self._cached_bases:
            U_c, Vt_c = self._cached_bases[pid]
            use_cache  = False

            if self.adaptive_refresh:
                proj      = _project(H_ef, U_c, Vt_c)
                drift_cos = _metric(proj)
                if drift_cos >= self.tau_drift:
                    compressed, best_rank = proj, self._cached_ranks[pid]
                    best_metric, need_refresh = drift_cos, False
                    use_cache = True
            elif step % self.update_freq != 0:
                proj      = _project(H_ef, U_c, Vt_c)
                drift_cos = _metric(proj)
                compressed, best_rank = proj, self._cached_ranks[pid]
                best_metric, need_refresh = drift_cos, False
                use_cache = True

        if need_refresh:
            self._refresh_counts[pid] = self._refresh_counts.get(pid, 0) + 1

            if self.criterion == "fixed" and self.fixed_rank:
                best_rank = min(self.fixed_rank, min(m, n))
                compressed, bU, bVt = _svd_approx(best_rank)
                best_metric = _metric(compressed)
            else:
                lo, hi = self.r_min, max_rank
                fa, fU, fVt = _svd_approx(max_rank)
                fm = _metric(fa)

                if fm < self.tau:
                    best_rank, best_metric, compressed, bU, bVt = max_rank, fm, fa, fU, fVt
                else:
                    best_rank, best_metric, compressed, bU, bVt = max_rank, fm, fa, fU, fVt
                    while lo < hi:
                        mid = (lo + hi) // 2
                        a, U, Vt = _svd_approx(mid)
                        met = _metric(a)
                        if met >= self.tau:
                            best_rank, best_metric, compressed, bU, bVt = mid, met, a, U, Vt
                            hi = mid
                        else:
                            lo = mid + 1

            self._cached_bases[pid] = (bU.detach(), bVt.detach())
            self._cached_ranks[pid] = best_rank
            drift_cos = best_metric

        return compressed, best_rank, need_refresh, drift_cos, best_metric

    # ── Model-level ───────────────────────────────────────────────────────────

    def compress_model(self, model, optimizer, step, eps=1e-8, compute_metrics=False):
        p2v2 = {}
        if self.criterion == "optimizer_aware":
            for g in optimizer.param_groups:
                for p in g["params"]:
                    st = optimizer.state.get(p, {})
                    if "exp_avg_sq" in st:
                        p2v2[id(p)] = st["exp_avg_sq"]

        all_stats = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            v2 = p2v2.get(id(param))
            _, stats = self.compress_gradient(name, param, step=step, v2=v2, eps=eps, compute_metrics=compute_metrics)
            if stats:
                all_stats[name] = stats
        return all_stats

    def total_svd_calls(self):
        return sum(self._refresh_counts.values())

    def reset_refresh_counts(self):
        self._refresh_counts.clear()

    def reset_error_buffers(self):
        self._error_buffers.clear()
