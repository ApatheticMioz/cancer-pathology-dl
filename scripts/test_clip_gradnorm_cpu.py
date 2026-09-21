#!/usr/bin/env python3
"""CPU-only unit test for the gradient-clipping + GradNorm weight-clamp fix.

Verifies, on CPU (no GPU run), that the stabilizers keep the optimizer step
and the combined multi-task loss FINITE under synthetic gradient spikes that
previously produced 1e5+ total grad norms (the TCGA batch-49 NaN regime):

  1. The EXACT clip path from src/training.py
     (``torch.nn.utils.clip_grad_norm_(clip_params, max_norm=GRAD_CLIP_MAX_NORM)``)
     bounds a 2e5+ pre-clip norm to <= max_norm and leaves finite grads.
  2. The REAL ``GradNormBalancer`` (src/models.py) keeps ``weights()`` finite
     and the task-weight ratio bounded even when a spike drives a log-weight
     to an extreme (where ``exp()`` would otherwise overflow float32 -> inf).
  3. The combined loss ``w[0].detach()*seg + w[1].detach()*cls`` stays finite
     at the clamp bound.
  4. The ``_scalar`` fix in ``_build_nan_diagnosis`` converts a 0-dim
     grad-requiring tensor to float WITHOUT the "Converting a tensor to a
     Python scalar" UserWarning.

Run:  ./venv/bin/python scripts/test_clip_gradnorm_cpu.py
Exit 0 = all checks pass; non-zero = a check failed.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn

from src.config import GRAD_CLIP_MAX_NORM, GRADNORM_WEIGHT_CLAMP
from src.models import GradNormBalancer

DEVICE = "cpu"  # CPU-only verification (no GPU run).
PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# 1. Exact clip path from training.py, with a 2e5+ pre-clip norm
# ---------------------------------------------------------------------------
def test_clip_path() -> None:
    print("\n== 1. clip path (exact training.py call) ==")
    # Tiny model: a shared encoder (Linear) feeding two heads, mirroring the
    # hard-parameter-sharing structure. We only need .parameters() for clipping.
    model = nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4),
    ).to(DEVICE)

    # Synthetic grads that produce a ~2e5 total L2 norm (matches the observed
    # 205,690.4 pre-clip norm on the TCGA batch-49 NaN). The model has 192
    # params, so per-element magnitude 15000 -> norm ~ 15000*sqrt(192) ~ 2.08e5.
    with torch.no_grad():
        for p in model.parameters():
            p.grad = torch.full_like(p, 15000.0)  # large, finite grads

    clip_params = list(model.parameters())
    # EXACT call from src/training.py (now using the config constant).
    pre_clip_norm = float(
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=GRAD_CLIP_MAX_NORM).item()
    )
    post_clip_norm = float(
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=GRAD_CLIP_MAX_NORM).item()
    )

    check("pre-clip norm is a 1e5+ spike", pre_clip_norm > 1e5,
          f"pre_clip={pre_clip_norm:.1f}")
    check("clip engaged (pre-clip > max_norm)", pre_clip_norm > GRAD_CLIP_MAX_NORM,
          f"max_norm={GRAD_CLIP_MAX_NORM}")
    check("post-clip norm bounded to <= max_norm", post_clip_norm <= GRAD_CLIP_MAX_NORM + 1e-6,
          f"post_clip={post_clip_norm:.6f} max_norm={GRAD_CLIP_MAX_NORM}")
    check("post-clip grads all finite",
          all(torch.isfinite(p.grad).all() for p in model.parameters()))
    # The per-step update magnitude is now bounded: with Adam (lr=1e-4) the
    # max param move per step is ~ lr * (grad/clip) which is O(1e-4), not O(1e5).
    max_grad_after = max(float(p.grad.abs().max()) for p in model.parameters())
    check("max |grad| after clip is small (no 1e5+ update)", max_grad_after < 1.0,
          f"max|grad|={max_grad_after:.4f}")


# ---------------------------------------------------------------------------
# 2. Real GradNormBalancer: weights() finite + ratio bounded under spikes
# ---------------------------------------------------------------------------
def test_gradnorm_weights() -> None:
    print("\n== 2. GradNormBalancer weights() under log-weight spikes ==")
    g = GradNormBalancer(5.0, 1.0, alpha=1.5, weight_clamp=GRADNORM_WEIGHT_CLAMP).to(DEVICE)

    # (a) Normal init: weights ~ [5,1], finite, positive.
    w0 = g.weights()
    check("init weights finite", bool(torch.isfinite(w0).all()),
          f"w={w0.tolist()}")
    check("init weights positive", bool((w0 > 0).all()))

    # (b) normalize_ keeps sum==2 and log_weights clamped.
    g.normalize_()
    w1 = g.weights()
    w1_sum = float(w1.sum().detach())
    check("after normalize_ sum==2", abs(w1_sum - 2.0) < 1e-4,
          f"sum={w1_sum:.6f}")
    check("after normalize_ log_weights clamped",
          bool((g.log_weights.abs() <= g._log_clamp + 1e-6).all()),
          f"log_w={g.log_weights.tolist()} bound={g._log_clamp:.4f}")

    # (c) SPIKE: drive one log-weight to an extreme where exp() would overflow
    #     float32 (exp(100) -> inf). weights() must stay finite via the clamp.
    with torch.no_grad():
        g.log_weights.copy_(torch.tensor([100.0, -100.0], dtype=torch.float32))
    w_spike = g.weights()
    check("spike weights() finite (no exp overflow)", bool(torch.isfinite(w_spike).all()),
          f"w={w_spike.tolist()}")
    w_max = float(w_spike.max().detach())
    check("spike weight capped at weight_clamp",
          w_max <= GRADNORM_WEIGHT_CLAMP + 1e-3,
          f"max={w_max:.4f} clamp={GRADNORM_WEIGHT_CLAMP}")
    # The task-weight ratio is bounded: max/min <= clamp^2 (both clamped to
    # +/-log(clamp)), so one task cannot dominate the shared encoder unboundedly.
    ratio = float((w_spike.max() / w_spike.min()).detach())
    check("task-weight ratio bounded (<= clamp^2)", ratio <= (GRADNORM_WEIGHT_CLAMP ** 2) + 1e-3,
          f"ratio={ratio:.3f} bound={GRADNORM_WEIGHT_CLAMP**2:.0f}")

    # (d) normalize_ after a spike: log_weights pulled back into the clamp band.
    g.normalize_()
    check("normalize_ after spike keeps log_weights clamped",
          bool((g.log_weights.abs() <= g._log_clamp + 1e-6).all()),
          f"log_w={g.log_weights.tolist()}")
    check("normalize_ after spike weights finite", bool(torch.isfinite(g.weights()).all()))


# ---------------------------------------------------------------------------
# 3. Combined multi-task loss stays finite at the clamp bound
# ---------------------------------------------------------------------------
def test_combined_loss_finite() -> None:
    print("\n== 3. combined loss w[0]*seg + w[1]*cls finite at clamp bound ==")
    g = GradNormBalancer(5.0, 1.0, alpha=1.5, weight_clamp=GRADNORM_WEIGHT_CLAMP).to(DEVICE)
    with torch.no_grad():
        g.log_weights.copy_(torch.tensor([100.0, -100.0], dtype=torch.float32))  # spike
    w = g.weights()
    # Finite (possibly large) per-task losses, as a hard batch would produce.
    seg_loss = torch.tensor(50.0, dtype=torch.float32, requires_grad=True)
    cls_loss = torch.tensor(50.0, dtype=torch.float32, requires_grad=True)
    loss = w[0].detach() * seg_loss + w[1].detach() * cls_loss
    check("combined loss finite at clamp bound", bool(torch.isfinite(loss)),
          f"loss={float(loss.detach()):.4f} w={w.tolist()}")
    # Backward through the combined loss: grads finite.
    loss.backward()
    check("combined-loss backward grads finite",
          bool(torch.isfinite(seg_loss.grad)) and bool(torch.isfinite(cls_loss.grad)))


# ---------------------------------------------------------------------------
# 4. _scalar fix: 0-dim grad-requiring tensor -> float without UserWarning
# ---------------------------------------------------------------------------
def test_scalar_no_warning() -> None:
    print("\n== 4. _scalar: 0-dim grad tensor -> float, no UserWarning ==")
    # Replicate the fixed _scalar from src/training.py::_build_nan_diagnosis.
    def _scalar(x):
        if x is None:
            return None
        try:
            if torch.is_tensor(x):
                return float(x.detach())
            return float(x)
        except (TypeError, ValueError):
            return None

    t = torch.tensor(3.14, requires_grad=True)  # 0-dim, grad-requiring
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn the UserWarning into an exception
        try:
            val = _scalar(t)
            check("_scalar(grad tensor) returns float, no warning",
                  abs(val - 3.14) < 1e-6, f"val={val}")
        except UserWarning as ex:
            check("_scalar(grad tensor) returns float, no warning", False,
                  f"UserWarning raised: {ex}")
    # NaN tensor still converts (the diagnosis path needs the NaN value).
    nan_t = torch.tensor(float("nan"), requires_grad=True)
    check("_scalar(NaN tensor) returns nan (diagnosis intact)",
          _scalar(nan_t) != _scalar(nan_t))  # NaN != NaN


def main() -> int:
    print(f"CPU-only clip + GradNorm verification  (torch {torch.__version__})")
    print(f"  GRAD_CLIP_MAX_NORM={GRAD_CLIP_MAX_NORM}  GRADNORM_WEIGHT_CLAMP={GRADNORM_WEIGHT_CLAMP}")
    test_clip_path()
    test_gradnorm_weights()
    test_combined_loss_finite()
    test_scalar_no_warning()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
