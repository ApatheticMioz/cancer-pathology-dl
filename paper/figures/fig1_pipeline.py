"""Figure 1: Publication-grade Architectural Formulation and Factorial Audit Framework.

Clean, professional, vector-quality research diagram following top IEEE/CV standards.
Eliminates equation clutter and verbose text while highlighting the modular flow,
multi-modal clinical cohorts, decoupled task heads, lateral skip connections,
and optimization/evaluation protocols.
Strict non-overlapping coordinate boundaries verified by visual inspection.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from . import style
from .fig8_qualitative import _infer, _RUNS, _overlay

# Apply locked matplotlib styling
style.apply()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_fig():
    """Build the publication-grade Figure 1 with zero label overlap."""
    # IEEE single column width: 3.35 inches wide, 4.90 inches tall
    fig, ax = plt.subplots(figsize=(3.35, 4.90), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Load 4 real thumbnails via cached inference runner or direct fallback
    try:
        recs = [_infer(r) for r in _RUNS]
    except Exception:
        import glob
        from PIL import Image
        img_tcga = np.array(Image.open(sorted(glob.glob(str(REPO_ROOT / 'data/TCGA/*/*_14.tif')))[0]).convert('RGB').resize((128, 128)))
        img_panda = np.array(Image.open(sorted(glob.glob(str(REPO_ROOT / 'data/PANDA/preprocessed_macenko_fixed/images/*.png')))[0]).convert('RGB').resize((128, 128)))
        img_siim = np.array(Image.open(sorted(glob.glob(str(REPO_ROOT / 'data/SIIM/preprocessed/images/*.png')))[0]).convert('RGB').resize((128, 128)))
        img_pannuke = np.array(Image.open(sorted(glob.glob(str(REPO_ROOT / 'data/PanNuke/preprocessed/images/*.png')))[0]).convert('RGB').resize((128, 128)))
        mask_path = sorted(glob.glob(str(REPO_ROOT / 'data/PANDA/train_label_masks/train_label_masks/*.png')))[0]
        mask_raw = np.array(Image.open(mask_path).resize((128, 128), Image.NEAREST))
        gt_panda = mask_raw[..., 0] if mask_raw.ndim == 3 else mask_raw
        recs = [
            {"dataset": "tcga", "img_rgb": img_tcga, "gt": None, "seg_classes": 1},
            {"dataset": "panda", "img_rgb": img_panda, "gt": gt_panda, "seg_classes": 6},
            {"dataset": "siim", "img_rgb": img_siim, "gt": None, "seg_classes": 1},
            {"dataset": "pannuke", "img_rgb": img_pannuke, "gt": None, "seg_classes": 5},
        ]

    # Ensure PANDA tile displays true glandular H&E biopsy tissue rather than empty white slide
    panda_raw_path = REPO_ROOT / "data" / "PANDA" / "train_images" / "train_images" / "3dab3238ef15a3c5b3d43e0b777073a5.png"
    panda_mask_path = REPO_ROOT / "data" / "PANDA" / "train_label_masks" / "train_label_masks" / "3dab3238ef15a3c5b3d43e0b777073a5.png"
    if panda_raw_path.exists():
        from PIL import Image
        p_raw = np.array(Image.open(panda_raw_path).convert("RGB"))
        h, w = p_raw.shape[:2]
        ch, cw = h // 2, w // 2
        crop_sz = 160
        r1, r2 = max(0, ch - crop_sz // 2), min(h, ch + crop_sz // 2)
        c1, c2 = max(0, cw - crop_sz // 2), min(w, cw + crop_sz // 2)
        recs[1]["img_rgb"] = np.array(Image.fromarray(p_raw[r1:r2, c1:c2]).resize((128, 128)))
        if panda_mask_path.exists():
            p_m = np.array(Image.open(panda_mask_path))
            if p_m.ndim == 3:
                p_m = p_m[..., 0]
            recs[1]["gt"] = np.array(Image.fromarray(p_m[r1:r2, c1:c2]).resize((128, 128), Image.NEAREST))

    # Helper: draw rounded container card
    def card(x, y, w, h, bg, border, title="", title_col="#0F172A", title_size=4.9):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.4",
            facecolor=bg, edgecolor=border, linewidth=0.9, zorder=1
        )
        ax.add_patch(rect)
        if title:
            ax.text(
                x + 2.5, y + h - 1.2, title,
                fontsize=title_size, fontweight="bold", color=title_col,
                va="top", ha="left", zorder=3
            )

    # =========================================================================
    # STAGE 1: CLINICAL COHORTS & PREPROCESSING (y: 77.5 to 99.0, h: 21.5)
    # =========================================================================
    card(2.0, 77.5, 96.0, 21.5, "#F8FAFC", "#64748B",
         "Stage 1: Multi-Modal Clinical Cohorts & Preprocessing", "#1E293B", title_size=4.9)

    # 4 miniature tiles
    tile_w, tile_h = 17.0, 10.4
    tile_y = 85.0
    x_starts = [5.0, 28.0, 51.0, 74.0]
    tile_labels = ["TCGA (MRI)", "PANDA (Biopsy)", "SIIM (X-Ray)", "PanNuke (Tissue)"]

    for rec, x_pos, lbl in zip(recs, x_starts, tile_labels):
        ax_ins = ax.inset_axes([x_pos / 100, tile_y / 100, tile_w / 100, tile_h / 100],
                               transform=ax.transAxes)
        ax_ins.imshow(rec["img_rgb"])
        ax_ins.axis("off")
        for spine in ax_ins.spines.values():
            spine.set_visible(True)
            spine.set_color("#475569")
            spine.set_linewidth(0.6)

        ax.text(
            x_pos + tile_w / 2, tile_y - 0.7, lbl,
            fontsize=4.4, fontweight="bold", ha="center", va="top", color="#334155"
        )

    # Preprocessing pill badge below tiles
    badge1 = patches.FancyBboxPatch(
        (5.0, 78.4), 90.0, 3.2,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=0.5, zorder=2
    )
    ax.add_patch(badge1)
    ax.text(50.0, 80.0, r"Patient-Isolated Partitioning   $\vert$   Stain Normalization (Macenko)",
            fontsize=4.4, fontweight="bold", ha="center", va="center", color="#1E293B", zorder=3)

    # Connector Arrow: Stage 1 -> Stage 2 (Gap from 77.5 down to 73.0: 4.5 units)
    ax.annotate(
        "", xy=(50.0, 73.4), xytext=(50.0, 77.2),
        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.2, mutation_scale=8),
        zorder=5
    )
    ax.text(53.0, 75.3, r"Input Image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$",
            fontsize=4.6, fontweight="bold", color="#1E293B", ha="left", va="center")

    # =========================================================================
    # STAGE 2: SHARED BACKBONE ENCODER (y: 54.0 to 73.0, h: 19.0)
    # Contracting Conv Hierarchy (Row 1) -> Shared Bottleneck Representation (Row 2)
    # =========================================================================
    card(2.0, 54.0, 96.0, 19.0, "#EFF6FF", "#2563EB",
         "Stage 2: Shared Backbone Encoder (VGG16 / MobileNetV2)", "#1E3A8A", title_size=4.9)

    # Row 1: Contracting Convolutional Blocks (y: 64.0 to 68.8, h: 4.8)
    b_xs = [5.0, 18.0, 31.0, 44.0]
    b_ws = 11.0
    b_hs = 4.8
    b_ys = 64.0
    b_lbls = ["Block 1", "Block 2", "Block 3", "Block 4"]
    b_dims = [r"$128^2$", r"$64^2$", r"$32^2$", r"$16^2$"]
    b_cols = ["#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6"]
    t_cols = ["#1E3A8A", "#1E3A8A", "#1E3A8A", "#FFFFFF"]

    for bx, bl, bd, bc, tc in zip(b_xs, b_lbls, b_dims, b_cols, t_cols):
        r_ = patches.FancyBboxPatch(
            (bx, b_ys), b_ws, b_hs,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=bc, edgecolor="#1E40AF", linewidth=0.5, zorder=2
        )
        ax.add_patch(r_)
        ax.text(bx + b_ws / 2, b_ys + b_hs / 2 + 0.6, bl,
                fontsize=4.2, fontweight="bold", ha="center", va="center", color=tc, zorder=3)
        ax.text(bx + b_ws / 2, b_ys + b_hs / 2 - 0.8, bd,
                fontsize=3.8, ha="center", va="center", color=tc, zorder=3)

    # Inter-block horizontal arrows
    for i in range(3):
        ax.annotate(
            "", xy=(b_xs[i + 1], b_ys + b_hs / 2),
            xytext=(b_xs[i] + b_ws, b_ys + b_hs / 2),
            arrowprops=dict(arrowstyle="-|>", color="#1E40AF", lw=0.7, mutation_scale=4.5),
            zorder=4
        )

    # Direct downward connection from Block 4 into Bottleneck
    ax.annotate(
        "", xy=(49.5, 60.5), xytext=(49.5, 64.0),
        arrowprops=dict(arrowstyle="-|>", color="#1E40AF", lw=1.1, mutation_scale=6),
        zorder=4
    )

    # Lateral Skip Feature Tap Bus in Row 1: Extends cleanly from Block 4 to x=82.0
    ax.annotate(
        "", xy=(82.0, 66.4), xytext=(55.0, 66.4),
        arrowprops=dict(arrowstyle="-", color="#0D9488", lw=1.1, linestyle="--"),
        zorder=3
    )
    ax.text(68.5, 68.2, r"Lateral Skips $\{\mathbf{s}_l\}_{l=1}^4$",
            fontsize=4.2, fontweight="bold", color="#0D9488", ha="center", va="center", zorder=3)

    # Row 2: Deep Shared Bottleneck Representation (y: 54.0 to 60.6, h: 6.6)
    # Ends at x=71.0, leaving x=71.0 to 98.0 clear for Lateral Skips Channel
    bottle_w = 58.0
    bottle_x = 13.0
    bottle_y = 54.0
    bottle_h = 6.6
    r_bot = patches.FancyBboxPatch(
        (bottle_x, bottle_y), bottle_w, bottle_h,
        boxstyle="round,pad=0.2,rounding_size=0.7",
        facecolor="#1D4ED8", edgecolor="#1E3A8A", linewidth=0.6, zorder=2
    )
    ax.add_patch(r_bot)
    ax.text(bottle_x + bottle_w / 2, bottle_y + bottle_h / 2 + 1.2,
            r"Shared Bottleneck Representation $\mathbf{z} \in \mathbb{R}^{h \times w \times d}$",
            fontsize=4.2, fontweight="bold", ha="center", va="center", color="#FFFFFF", zorder=3)
    ax.text(bottle_x + bottle_w / 2, bottle_y + bottle_h / 2 - 1.1,
            r"Lowest-Resolution Deep Feature Encoding ($8 \times 8$ Latent Grid)",
            fontsize=3.6, ha="center", va="center", color="#BFDBFE", zorder=3)

    # =========================================================================
    # CONNECTORS: STAGE 2 -> STAGE 3 (Gap: 54.0 down to 48.0, h: 6.0)
    # 3 Clean Dedicated Non-Crossing Channels:
    # Channel 1 (x: 25.0): Bottleneck z -> Task 1 (Classification Head)
    # Channel 2 (x: 60.0): Bottleneck z -> Task 2 (Segmentation Decoder)
    # Channel 3 (x: 82.0): Lateral Skips {s_l} -> Task 2 (Segmentation Decoder)
    # =========================================================================
    # Channel 1 -> Task 1 (Classification): Takes Bottleneck z
    ax.annotate(
        "", xy=(25.0, 48.3), xytext=(25.0, 55.4),
        arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.2, mutation_scale=8),
        zorder=5
    )
    ax.text(26.8, 51.5, r"Bottleneck $\mathbf{z}$", fontsize=4.3, fontweight="bold",
            color="#92400E", ha="left", va="center")

    # Channel 2 -> Task 2 (Segmentation Decoder): Takes Bottleneck z
    ax.annotate(
        "", xy=(60.0, 48.3), xytext=(60.0, 55.4),
        arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.2, mutation_scale=8),
        zorder=5
    )
    ax.text(61.8, 51.5, r"Bottleneck $\mathbf{z}$", fontsize=4.3, fontweight="bold",
            color="#065F46", ha="left", va="center")

    # Channel 3 -> Task 2 (Segmentation Decoder): Lateral Skips route straight down
    ax.annotate(
        "", xy=(82.0, 48.3), xytext=(82.0, 66.4),
        arrowprops=dict(arrowstyle="-|>", color="#0D9488", lw=1.1, linestyle="--", mutation_scale=7),
        zorder=5
    )
    ax.text(83.8, 51.5, r"Skips $\{\mathbf{s}_l\}$", fontsize=4.1, fontweight="bold",
            color="#0D9488", ha="left", va="center")

    # =========================================================================
    # STAGE 3: DECOUPLED MULTI-TASK HEADS (y: 23.5 to 48.0, h: 24.5)
    # Clean Card Boundaries with Inset Diagnostics & Loss Tags
    # =========================================================================
    # Task 1: Classification Head (Left Card, x: 2.0 to 48.5)
    card(2.0, 23.5, 46.5, 24.5, "#FFFBEB", "#D97706",
         "Stage 3: Task 1 (Classification)", "#92400E", title_size=4.3)

    ax.text(4.5, 45.4, r"$\text{GAP} \to \text{MLP} \to \hat{\mathbf{y}}_{cls}$",
            fontsize=4.3, fontweight="bold", color="#78350F", va="top")

    # Miniature probability bar chart (y from 31.5 to 41.0 in figure coords)
    ax_bar = ax.inset_axes([0.055, 0.315, 0.38, 0.095], transform=ax.transAxes)
    classes = ["G0", "G1", "G2", "G3", "G4", "G5"]
    probs = [0.08, 0.12, 0.62, 0.10, 0.05, 0.03]
    x_indices = np.arange(len(probs))
    ax_bar.bar(x_indices, probs, color="#F59E0B", edgecolor="#B45309", linewidth=0.5, width=0.65)
    ax_bar.set_ylim(0, 0.8)
    ax_bar.set_yticks([])
    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels(classes, fontsize=4.1)
    ax_bar.tick_params(axis="x", pad=1, length=1)
    for sp in ax_bar.spines.values():
        sp.set_visible(False)
    ax_bar.spines["bottom"].set_visible(True)
    ax_bar.spines["bottom"].set_color("#B45309")
    ax_bar.spines["bottom"].set_linewidth(0.5)

    # Classification Loss Tag (y: 24.6 to 27.6)
    loss_box1 = patches.FancyBboxPatch(
        (4.5, 24.6), 41.5, 3.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#FEF3C7", edgecolor="#D97706", linewidth=0.5, zorder=2
    )
    ax.add_patch(loss_box1)
    ax.text(25.2, 26.1, r"Loss $\mathcal{L}_{cls}$ (Categorical CE)",
            fontsize=4.2, fontweight="bold", ha="center", va="center", color="#92400E", zorder=3)

    # Task 2: Segmentation Decoder (Right Card, x: 51.5 to 98.0)
    card(51.5, 23.5, 46.5, 24.5, "#ECFDF5", "#059669",
         "Stage 3: Task 2 (Segmentation)", "#065F46", title_size=4.3)

    ax.text(54.0, 45.4, r"$\text{Dec}_\phi(\mathbf{z}, \{\mathbf{s}_l\}) \to \mathbf{l}_{seg}$",
            fontsize=4.3, fontweight="bold", color="#064E3B", va="top")

    # Miniature segmentation overlay thumbnail (y from 31.5 to 41.0 in figure coords)
    ax_seg = ax.inset_axes([0.555, 0.315, 0.38, 0.095], transform=ax.transAxes)
    over = _overlay(recs[1]["img_rgb"], recs[1]["gt"], recs[1]["seg_classes"])
    ax_seg.imshow(over)
    ax_seg.axis("off")
    for sp in ax_seg.spines.values():
        sp.set_visible(True)
        sp.set_color("#059669")
        sp.set_linewidth(0.5)

    # Segmentation Loss Tag (y: 24.6 to 27.6)
    loss_box2 = patches.FancyBboxPatch(
        (54.0, 24.6), 41.5, 3.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#D1FAE5", edgecolor="#059669", linewidth=0.5, zorder=2
    )
    ax.add_patch(loss_box2)
    ax.text(74.7, 26.1, r"Loss $\mathcal{L}_{seg}$ (Mask CE)",
            fontsize=4.2, fontweight="bold", ha="center", va="center", color="#065F46", zorder=3)

    # Converging connectors into Stage 4 (Gap from 23.5 down to 18.5: 5.0 units)
    ax.annotate(
        "", xy=(25.0, 18.8), xytext=(25.0, 23.2),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, mutation_scale=8),
        zorder=5
    )
    ax.annotate(
        "", xy=(74.0, 18.8), xytext=(74.0, 23.2),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, mutation_scale=8),
        zorder=5
    )

    # =========================================================================
    # STAGE 4: JOINT OPTIMIZATION & EVALUATION PROTOCOL (y: 1.5 to 18.5, h: 17.0)
    # Zero Formula Clutter - Clean Functional Modules per IEEE Standards
    # =========================================================================
    card(2.0, 1.5, 96.0, 17.0, "#F5F3FF", "#7C3AED",
         "Stage 4: Multi-Task Optimization & Evaluation Protocols", "#5B21B6", title_size=4.9)

    # Module 1: Multi-Task Loss Balancing (y: 9.8 to 15.2, h: 5.4)
    g_strip = patches.FancyBboxPatch(
        (4.5, 9.8), 91.0, 5.4,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor="#EDE9FE", edgecolor="#8B5CF6", linewidth=0.5, zorder=2
    )
    ax.add_patch(g_strip)
    ax.text(50.0, 13.4, "Multi-Task Optimization Dynamics",
            fontsize=4.5, fontweight="bold", color="#4C1D95", ha="center", va="center", zorder=3)
    ax.text(50.0, 11.4, r"Static Loss Weighting ($\lambda_{cls}{:}\lambda_{seg}$)   $\vert$   Adaptive GradNorm Balancing",
            fontsize=4.1, color="#5B21B6", ha="center", va="center", zorder=3)

    # Module 2: Evaluation Integrity Protocol (y: 2.8 to 8.2, h: 5.4)
    a_strip = patches.FancyBboxPatch(
        (4.5, 2.8), 91.0, 5.4,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor="#FFFFFF", edgecolor="#A78BFA", linewidth=0.5, zorder=2
    )
    ax.add_patch(a_strip)
    ax.text(50.0, 6.4, "Clinical Evaluation Protocol",
            fontsize=4.5, fontweight="bold", color="#4C1D95", ha="center", va="center", zorder=3)
    ax.text(50.0, 4.4, r"Disaggregated Foreground Dice ($d_{fg}$)   $\vert$   Empty-Credit Score Floor Guard ($\rho$)",
            fontsize=4.1, color="#6D28D9", ha="center", va="center", zorder=3)

    return fig, ax


def build() -> list:
    """Render Figure 1 and save PDF + PNG. Returns the written paths."""
    fig, _ = _build_fig()
    out_dir = REPO_ROOT / "paper"
    pdf_path = out_dir / "fig1_pipeline.pdf"
    png_path = out_dir / "fig1_pipeline.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=style.SAVE_DPI)
    plt.close(fig)
    return [pdf_path, png_path]


def verify() -> dict:
    """Verify Figure 1 assets and dimensions."""
    fig, ax = _build_fig()
    fig.canvas.draw()
    w_in, h_in = fig.get_size_inches()
    plt.close(fig)
    return {
        "fig1_pipeline_w_in": float(w_in),
        "fig1_pipeline_h_in": float(h_in),
    }


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
