# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Online mean / variance accumulator for per-parameter gradient tensors,
following Welford's algorithm.

For a sequence of samples x_1, x_2, ... x_N (each a ``dict[str, Tensor]``):

    n = 0;  mean[k] = 0;  M2[k] = 0
    on new sample x:
        n     += 1
        delta  = x[k] - mean[k]
        mean[k] += delta / n
        delta2 = x[k] - mean[k]
        M2[k]  += delta * delta2

    population variance = M2 / n
    sample variance     = M2 / (n - 1)

Only the running ``mean`` and ``M2`` need to survive between updates, so
each incoming gradient file can be deleted as soon as it has been folded in.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch


class WelfordAccumulator:
    """In-memory accumulator with disk-checkpointed state."""

    def __init__(self, state_dir: str, dtype: torch.dtype = torch.float32):
        self.state_dir = state_dir
        self.dtype = dtype
        os.makedirs(state_dir, exist_ok=True)
        self._mean_path = os.path.join(state_dir, "welford_mean.pt")
        self._M2_path = os.path.join(state_dir, "welford_M2.pt")
        self._meta_path = os.path.join(state_dir, "welford_meta.json")

        self.n: int = 0
        self.mean: Optional[dict[str, torch.Tensor]] = None
        self.M2: Optional[dict[str, torch.Tensor]] = None
        self._try_resume()

    def _try_resume(self) -> None:
        if not os.path.exists(self._meta_path):
            return
        with open(self._meta_path) as f:
            self.n = int(json.load(f)["n"])
        if self.n > 0:
            self.mean = torch.load(self._mean_path, map_location="cpu")
            self.M2 = torch.load(self._M2_path, map_location="cpu")

    def update(self, sample: dict[str, torch.Tensor]) -> None:
        """Fold one new sample into the running mean / M2."""
        if self.mean is None:
            self.mean = {k: v.detach().to(self.dtype).clone() for k, v in sample.items()}
            self.M2 = {k: torch.zeros_like(v, dtype=self.dtype) for k, v in sample.items()}
            self.n = 1
        else:
            self.n += 1
            inv_n = 1.0 / self.n
            for k, v in sample.items():
                x = v.detach().to(self.dtype)
                delta = x - self.mean[k]
                self.mean[k].add_(delta, alpha=inv_n)
                delta2 = x - self.mean[k]
                self.M2[k].add_(delta * delta2)
        self._checkpoint()

    def _checkpoint(self) -> None:
        tmp_mean = self._mean_path + ".tmp"
        tmp_M2 = self._M2_path + ".tmp"
        torch.save(self.mean, tmp_mean)
        os.replace(tmp_mean, self._mean_path)
        torch.save(self.M2, tmp_M2)
        os.replace(tmp_M2, self._M2_path)
        with open(self._meta_path, "w") as f:
            json.dump({"n": self.n}, f)

    def finalize(
        self,
        out_dir: Optional[str] = None,
        sample_variance: bool = True,
        save_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return ``(mean, variance)`` and optionally write them out.

        Sample variance uses Bessel's correction (divide by n-1); set
        ``sample_variance=False`` for population variance.
        """
        if self.n == 0:
            raise RuntimeError("WelfordAccumulator has no samples")
        denom = max(self.n - 1, 1) if sample_variance else self.n
        var = {k: (self.M2[k] / denom) for k in self.M2}

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                {k: v.to(save_dtype) for k, v in self.mean.items()},
                os.path.join(out_dir, "grad_mean.pt"),
            )
            torch.save(
                {k: v.to(save_dtype) for k, v in var.items()},
                os.path.join(out_dir, "grad_var.pt"),
            )
            with open(os.path.join(out_dir, "grad_stats_meta.json"), "w") as f:
                json.dump({"n": self.n, "sample_variance": sample_variance}, f, indent=2)
        return self.mean, var
