"""
oasis/hooks.py

PyTorch hook-based integration for OASIS.

Provides OASISHookManager which attaches gradient hooks to a model so that
compression happens automatically before the optimizer step, without requiring
manual call sites in the training loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .compressor import OASISCompressor


class OASISHookManager:
    """
    Manages backward hooks that compress gradients automatically.

    Usage::

        hook_manager = OASISHookManager(model, compressor, optimizer)
        hook_manager.register()

        # ... normal training loop ...
        loss.backward()          # hooks fire here, gradients compressed in-place
        optimizer.step()
        hook_manager.zero()      # also calls optimizer.zero_grad()

    """

    def __init__(
        self,
        model: nn.Module,
        compressor: OASISCompressor,
        optimizer: torch.optim.Optimizer,
        compress_after_n_steps: int = 1,
    ):
        self.model       = model
        self.compressor  = compressor
        self.optimizer   = optimizer
        self.compress_after_n_steps = compress_after_n_steps

        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._step    = 0

    # ── hook lifecycle ────────────────────────────────────────────────────────

    def register(self):
        """Attach gradient hooks to all 2-D+ parameters."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Capture `name` in closure
            def make_hook(n, p):
                def _hook(grad):
                    if self._step >= self.compress_after_n_steps:
                        # Pull second-moment estimate if available
                        state = self.optimizer.state.get(p, {})
                        v2 = state.get("exp_avg_sq", None)
                        _, _ = self.compressor.compress_gradient(n, p, v2=v2)
                        return p.grad   # return compressed grad
                    return grad
                return _hook

            handle = param.register_hook(make_hook(name, param))
            self._handles.append(handle)

    def remove(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def step(self):
        """Call after optimizer.step() to increment internal counter."""
        self._step += 1

    def zero(self):
        """Zero gradients and advance step counter."""
        self.optimizer.zero_grad()
        self.step()

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def last_stats(self) -> Dict[str, dict]:
        return self.compressor.last_stats
