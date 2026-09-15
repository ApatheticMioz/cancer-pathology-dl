"""Figure 8 — Qualitative GT-vs-Prediction Validation Tiles.

A 4-row (one per dataset) x 3-column (Image | Ground truth | Prediction)
grid of representative validation tiles, produced by **forward-pass
inference only** on the canonical v1 naked VGG16 baselines:

* Run 01 — TCGA-LGG   (binary seg, sigmoid > 0.5)
* Run 03 — PANDA      (6-class seg, argmax)
* Run 05 — SIIM-ACR   (binary seg, sigmoid > 0.5)
* Run 13 — PanNuke    (6-class seg, argmax)

Design contract
---------------
* The validation split is **identical to training**: the bundle is loaded
  with :func:`src.data.load_dataset_bundle` and split with
  :func:`src.data.make_group_split` at ``seed=RANDOM_SEED`` (42), and the
  val transform comes from :func:`src.data.build_transforms`. PANDA uses the
  ``preprocessed_macenko/`` images (the bundle default, ``skip_macenko=False``);
  SIIM and PanNuke use their bundle defaults.
* Checkpoint paths are **resolved from each run's**
  ``checkpoints/summary_<NN>_*.json`` ``checkpoint`` field — never guessed.
* Tile selection is deterministic and stated in the caption: the first
  validation tile (index order) with a non-empty ground-truth mask. SIIM is
  the exception — it advances to the first lesion-bearing tile whose
  prediction is *all-empty*, so the empty-mask degeneracy of Sec.~2.2 is the
  visible face of the figure. The chosen index is recorded and printed.
* Overlays: the GT panel paints the mask (fill + outline) in a
  colorblind-safe Okabe-Ito color over the image; the Prediction panel paints
  the predicted mask in the same palette and shows the **per-tile Dice**
  (binary Dice, or macro Dice over the classes present in the GT) in the
  panel title. **No result value is hard-coded** — every number (Dice, tile
  index) is computed at runtime from the forward pass (I4).

Output: ``paper/fig8_qualitative_overlays.pdf`` + ``.png`` (via
:func:`paper.figures.style.save_figure`).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import style
from .loaders import REPO_ROOT

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Okabe-Ito) — the same palette family that
# :data:`paper.figures.style.DATASET_COLORS` is drawn from. Extended to the
# full 8-color Okabe-Ito set so multi-class masks (up to 6 seg classes) get
# distinct, colorblind-safe colors.
# ---------------------------------------------------------------------------
_OKABE_ITO = [
    "#E69F00",  # 0  orange
    "#56B4E9",  # 1  sky blue
    "#009E73",  # 2  bluish green
    "#F0E442",  # 3  yellow
    "#0072B2",  # 4  blue
    "#D55E00",  # 5  vermillion
    "#CC79A7",  # 6  reddish purple
    "#999999",  # 7  grey
]

#: Overlay alpha for the mask fill (0 = invisible, 1 = opaque).
_OVERLAY_ALPHA = 0.45


def _hex_to_rgb255(hexstr: str) -> np.ndarray:
    """Convert a ``#RRGGBB`` string to an ``(3,)`` float array in [0, 255]."""
    h = hexstr.lstrip("#")
    return np.array(
        [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Run specification (canonical v1 naked VGG16 baselines).
# ``ckpt`` is resolved from the run's summary JSON (never hard-coded).
# ---------------------------------------------------------------------------
_RUNS = [
    {"dataset": "tcga",    "run": "01", "label": "TCGA-LGG", "binary": True},
    {"dataset": "panda",   "run": "03", "label": "PANDA",    "binary": False},
    {"dataset": "siim",    "run": "05", "label": "SIIM-ACR", "binary": True},
    {"dataset": "pannuke", "run": "13", "label": "PanNuke",  "binary": False},
]


def _resolve_checkpoint(run: dict) -> Path:
    """Resolve a run's checkpoint path from its ``summary_<NN>_*.json``.

    Reads the ``checkpoint`` field out of the run's summary JSON (the
    ``runs.<key>.checkpoint`` entry) — the path is never guessed.
    """
    pattern = str(REPO_ROOT / "checkpoints" / f"summary_{run['run']}_*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No summary JSON matching {pattern}")
    summary = json.loads(Path(matches[0]).read_text())
    for _key, run_info in summary.get("runs", {}).items():
        if "checkpoint" in run_info:
            return Path(run_info["checkpoint"])
    raise KeyError(f"No 'checkpoint' field in {matches[0]}")


# ---------------------------------------------------------------------------
# Inference (forward-pass only). Cached so build() and verify() share one run.
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _infer(run: dict) -> dict:
    """Run forward-pass inference for one dataset and return a tile record.

    Returns a dict with the display image, GT mask, predicted mask, the
    per-tile Dice, the chosen validation index, and the run metadata.
    """
    if run["dataset"] in _CACHE:
        return _CACHE[run["dataset"]]

    from src.config import DATASET_ROOTS, DATASET_META, RANDOM_SEED
    from src.data import (
        load_dataset_bundle,
        make_group_split,
        build_transforms,
        MultiTaskDataset,
    )
    from src.models import MultiTaskUNet

    ds = run["dataset"]
    meta = DATASET_META[ds]
    img_size = meta["img_size"]
    seg_classes = meta["seg_classes"]
    binary = run["binary"]

    # Exact training val split: bundle + group split at the locked seed.
    # PANDA uses the bundle default (preprocessed_macenko/); SIIM/PanNuke use
    # their bundle defaults. skip_macenko=False for all four.
    bundle = load_dataset_bundle(ds, DATASET_ROOTS[ds], skip_macenko=False)
    labels, groups = bundle["labels"], bundle["groups"]
    _tr_idx, val_idx = make_group_split(labels, groups, seed=RANDOM_SEED, test_size=0.2)

    _, val_tf = build_transforms(img_size)
    val_ds = MultiTaskDataset(
        bundle["images"][val_idx], bundle["masks"][val_idx], bundle["labels"][val_idx],
        seg_classes=seg_classes,
        binary_positive_min=int(meta.get("binary_positive_min", 1)),
        crop_to_mask_bbox=False,
        transform=val_tf,
        cache_size=0,
    )

    # Load the canonical checkpoint (resolved from the summary JSON).
    ckpt = _resolve_checkpoint(run)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskUNet(
        encoder_name="vgg16",
        num_classes=meta["num_classes"],
        seg_classes=seg_classes,
        skip_connections=True,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    def _forward(pos: int):
        img, mask, _label = val_ds[pos]
        with torch.no_grad():
            seg_out, _cls_out = model(img.unsqueeze(0).to(device))
        # Normalize both to 2D (H, W): binary seg_out is (1,1,H,W), the mask
        # is (1,H,W) (binary) or (H,W) (multi-class); .squeeze() collapses the
        # leading size-1 dims to a consistent (H, W).
        if binary:
            pred = (torch.sigmoid(seg_out) > 0.5).squeeze().cpu().numpy().astype(np.float32)
        else:
            pred = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy().astype(np.int64)
        gt = mask.squeeze().cpu().numpy()
        return img, gt, pred

    def _dice(gt: np.ndarray, pred: np.ndarray) -> float:
        if binary:
            g = (gt > 0).astype(np.float32)
            p = (pred > 0).astype(np.float32)
            inter = float((p * g).sum())
            union = float(p.sum() + g.sum())
            return (2.0 * inter / (union + 1e-8)) if union > 0 else 1.0
        # Macro Dice over the foreground classes present in the GT.
        scores = []
        for c in range(1, seg_classes):
            if int((gt == c).sum()) == 0:
                continue
            p = (pred == c).astype(np.float32)
            t = (gt == c).astype(np.float32)
            inter = float((p * t).sum())
            union = float(p.sum() + t.sum())
            scores.append((2.0 * inter / (union + 1e-8)) if union > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    # Deterministic tile selection (index order).
    chosen = None
    for pos in range(len(val_ds)):
        _img, gt, pred = _forward(pos)
        gt_nonempty = bool(gt.sum() > 0)
        if not gt_nonempty:
            continue
        if binary and ds == "siim":
            # SIIM: advance to the first lesion-bearing tile whose prediction
            # is all-empty (the visible face of the empty-mask degeneracy).
            if float(pred.sum()) == 0.0:
                chosen = pos
                break
        else:
            chosen = pos
            break

    if chosen is None:
        # Fallback: first non-empty GT tile (should not happen for SIIM here).
        for pos in range(len(val_ds)):
            _img, gt, _pred = _forward(pos)
            if gt.sum() > 0:
                chosen = pos
                break
    if chosen is None:
        raise RuntimeError(f"{ds}: no non-empty GT tile found in the val split")

    img, gt, pred = _forward(chosen)
    dice = _dice(gt, pred)

    # Display image: un-normalized RGB resized to the model input size.
    disp_path = str(bundle["images"][val_idx[chosen]])
    img_rgb = np.array(Image.open(disp_path).convert("RGB").resize((img_size, img_size)))

    rec = {
        "dataset": ds,
        "run": run["run"],
        "label": run["label"],
        "encoder": "VGG16",
        "binary": binary,
        "seg_classes": seg_classes,
        "img_size": img_size,
        "img_rgb": img_rgb,
        "gt": gt,
        "pred": pred,
        "dice": dice,
        "tile_idx": int(chosen),
        "n_val": int(len(val_ds)),
        "ckpt": str(ckpt),
    }
    _CACHE[ds] = rec
    return rec


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
def _overlay(img_rgb: np.ndarray, mask: np.ndarray, seg_classes: int) -> np.ndarray:
    """Paint a mask over an RGB image using the Okabe-Ito palette with boundary contours.

    Binary masks use the first palette color; multi-class masks use one
    colorblind-safe color per class. Returns a new RGB array with semi-transparent
    fill and sharp solid boundary contours.
    """
    from scipy.ndimage import binary_dilation

    out = img_rgb.astype(np.float64).copy()
    if seg_classes == 1:
        classes_to_draw = [(1, mask > 0, _hex_to_rgb255(_OKABE_ITO[0]))]
    else:
        classes_to_draw = [
            (c, mask == c, _hex_to_rgb255(_OKABE_ITO[c % len(_OKABE_ITO)]))
            for c in range(1, seg_classes)
        ]

    for _c, m, col in classes_to_draw:
        if not m.any():
            continue
        # Boundary delineation (1-pixel contour)
        contour = binary_dilation(m, iterations=1) ^ m
        # Semi-transparent mask fill
        out[m] = out[m] * (1.0 - _OVERLAY_ALPHA) + col * _OVERLAY_ALPHA
        # High-contrast solid contour for clinical boundary scrutiny
        out[contour] = col

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------
def _build_fig():
    """Build the Figure 8 grid and return ``(fig, records)``."""
    import matplotlib.pyplot as plt

    records = [_infer(run) for run in _RUNS]

    n_rows = len(records)
    fig = plt.figure(figsize=(5.4, 7.2))
    gs = fig.add_gridspec(
        n_rows, 4,
        left=0.02, right=0.98, top=0.93, bottom=0.02,
        wspace=0.12, hspace=0.22,
        width_ratios=[0.95, 1.0, 1.0, 1.0],
    )

    for r, rec in enumerate(records):
        # --- Row label (dataset + run + encoder) -------------------------
        ax_lab = fig.add_subplot(gs[r, 0])
        ax_lab.axis("off")
        ax_lab.text(
            0.5, 0.5,
            f"{rec['label']}\nRun {rec['run']} · {rec['encoder']}",
            ha="center", va="center", fontsize=8, fontweight="bold",
        )

        # --- Image panel --------------------------------------------------
        ax_img = fig.add_subplot(gs[r, 1])
        ax_img.imshow(rec["img_rgb"])
        ax_img.set_title("Image", fontsize=8, fontweight="bold")
        ax_img.axis("off")

        # --- Ground-truth panel ------------------------------------------
        ax_gt = fig.add_subplot(gs[r, 2])
        ax_gt.imshow(_overlay(rec["img_rgb"], rec["gt"], rec["seg_classes"]))
        ax_gt.set_title("Ground truth", fontsize=8, fontweight="bold")
        ax_gt.axis("off")

        # --- Prediction panel (with per-tile Dice) -----------------------
        ax_pred = fig.add_subplot(gs[r, 3])
        ax_pred.imshow(_overlay(rec["img_rgb"], rec["pred"], rec["seg_classes"]))
        dice_txt = f"Prediction\nDice = {rec['dice']:.2f}"
        if rec["binary"] and rec["dataset"] == "siim" and float(rec["pred"].sum()) == 0.0:
            dice_txt += "\n(all-empty)"
        ax_pred.set_title(dice_txt, fontsize=8, fontweight="bold")
        ax_pred.axis("off")

    return fig, records


def build() -> list:
    """Render Figure 8 and save PDF + PNG. Returns the written paths."""
    fig, records = _build_fig()
    out_dir = REPO_ROOT / "paper"
    paths = style.save_figure(fig, out_dir, "fig8_qualitative_overlays")
    # Print the computed (never hard-coded) per-tile values.
    for rec in records:
        print(
            f"    [fig8] {rec['label']:10s} run {rec['run']} {rec['encoder']}: "
            f"val idx {rec['tile_idx']} (of {rec['n_val']}), "
            f"{'binary' if rec['binary'] else 'macro'} Dice = {rec['dice']:.4f}, "
            f"ckpt = {Path(rec['ckpt']).name}"
        )
    return paths


def verify() -> dict:
    """Programmatically verify the Figure 8 layout and computed values.

    Checks that the grid is 4 rows x 3 image columns, that every panel has a
    title, that the SIIM row demonstrates the all-empty prediction on a
    lesion-bearing tile (Dice == 0), and that the figure fits the target
    print size. Raises ``AssertionError`` on any violation.
    """
    fig, records = _build_fig()
    fig.canvas.draw()

    n_rows = len(records)
    assert n_rows == 4, f"expected 4 dataset rows, got {n_rows}"

    # Count image axes (those with an image) and confirm titles exist.
    n_image_axes = 0
    for ax in fig.axes:
        if ax.get_images():
            n_image_axes += 1
            assert ax.get_title(), "an image panel is missing a title"
    # 4 rows x 3 image panels = 12 image axes.
    assert n_image_axes == 12, f"expected 12 image panels, got {n_image_axes}"

    # SIIM must be the all-empty-prediction, lesion-bearing case.
    siim = next(r for r in records if r["dataset"] == "siim")
    assert siim["gt"].sum() > 0, "SIIM chosen tile must be lesion-bearing"
    assert float(siim["pred"].sum()) == 0.0, "SIIM chosen tile must have an all-empty prediction"
    assert siim["dice"] == 0.0, "SIIM all-empty prediction must yield Dice 0"

    # Figure must fit the target print size (~textwidth x <=19cm).
    w_in, h_in = fig.get_size_inches()
    w_cm, h_cm = w_in * 2.54, h_in * 2.54
    assert h_cm <= 19.5, f"figure height {h_cm:.2f}cm exceeds the 19cm budget"

    return {
        "fig8_n_rows": n_rows,
        "fig8_n_image_panels": n_image_axes,
        "fig8_size_inches": (round(w_in, 3), round(h_in, 3)),
        "fig8_size_cm": (round(w_cm, 2), round(h_cm, 2)),
        "fig8_tile_indices": {r["label"]: r["tile_idx"] for r in records},
        "fig8_dice": {r["label"]: round(r["dice"], 4) for r in records},
        "fig8_siim_all_empty": bool(float(siim["pred"].sum()) == 0.0),
    }
