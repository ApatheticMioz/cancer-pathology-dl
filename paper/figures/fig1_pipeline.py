"""Figure 1: Publication-grade Architectural Formulation and Factorial Audit Framework.

Clean, professional, vector-quality research diagram following top IEEE/CV standards.
Eliminates defensive experimental scorecards while highlighting the modular flow,
multi-modal clinical cohorts, decoupled task heads, and optimization dynamics.
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

    # Load 4 real thumbnails via cached inference runner
    recs = [_infer(r) for r in _RUNS]

    # Helper: draw rounded container card
    def card(x, y, w, h, bg, border, title="", title_col="#0F172A", title_size=5.4):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.6",
            facecolor=bg, edgecolor=border, linewidth=0.9, zorder=1
        )
        ax.add_patch(rect)
        if title:
            ax.text(
                x + 2.5, y + h - 1.0, title,
                fontsize=title_size, fontweight="bold", color=title_col,
                va="top", ha="left", zorder=3
            )

    # =========================================================================
    # STAGE 1: CLINICAL COHORTS & PREPROCESSING (y: 77.0 to 99.0, h: 22.0)
    # =========================================================================
    card(2.0, 77.0, 96.0, 22.0, "#F8FAFC", "#64748B",
         "Stage 1: Multi-Modal Clinical Cohorts & Preprocessing", "#1E293B", title_size=5.3)

    # 4 miniature tiles
    tile_w, tile_h = 17.0, 10.6
    tile_y = 84.4
    x_starts = [5.0, 28.0, 51.0, 74.0]
    tile_labels = ["TCGA (MRI)", "PANDA (Biopsy)", "SIIM (X-Ray)", "PanNuke (Tissue)"]

    for rec, x_pos, lbl in zip(recs, x_starts, tile_labels):
        # Inset image
        ax_ins = ax.inset_axes([x_pos / 100, tile_y / 100, tile_w / 100, tile_h / 100],
                               transform=ax.transAxes)
        ax_ins.imshow(rec["img_rgb"])
        ax_ins.axis("off")
        for spine in ax_ins.spines.values():
            spine.set_visible(True)
            spine.set_color("#475569")
            spine.set_linewidth(0.6)

        # Label placed cleanly BELOW tile
        ax.text(
            x_pos + tile_w / 2, tile_y - 0.8, lbl,
            fontsize=4.5, fontweight="bold", ha="center", va="top", color="#334155"
        )

    # Preprocessing pill badge below tiles
    badge1 = patches.FancyBboxPatch(
        (5.0, 77.8), 90.0, 3.2,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=0.5, zorder=2
    )
    ax.add_patch(badge1)
    ax.text(50.0, 79.4, r"Patient-Isolated GroupKFold   $\vert$   Macenko Stain Normalization",
            fontsize=4.6, fontweight="bold", ha="center", va="center", color="#1E293B", zorder=3)

    # Connector Arrow: Stage 1 -> Stage 2 (Gap from 77.0 down to 72.0: 5.0 units)
    ax.annotate(
        "", xy=(50.0, 72.5), xytext=(50.0, 76.5),
        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.2, mutation_scale=8),
        zorder=5
    )
    ax.text(53.0, 74.5, r"Input $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$",
            fontsize=4.8, fontweight="bold", color="#1E293B", ha="left", va="center")

    # =========================================================================
    # STAGE 2: SHARED BACKBONE ENCODER (y: 56.0 to 72.0, h: 16.0)
    # =========================================================================
    card(2.0, 56.0, 96.0, 16.0, "#EFF6FF", "#2563EB",
         "Stage 2: Shared Backbone Feature Extraction (VGG16 / MobileNetV2)", "#1E3A8A", title_size=5.2)

    # Conv blocks
    b_xs = [5.0, 20.0, 35.0, 50.0, 68.0]
    b_ws = [12.0, 12.0, 12.0, 12.0, 26.0]
    b_hs = [5.2, 4.8, 4.4, 4.0, 5.2]
    b_ys = [62.0, 62.2, 62.4, 62.6, 62.0]
    b_lbls = ["Block 1", "Block 2", "Block 3", "Block 4", r"Bottleneck $\mathbf{z}$"]
    b_dims = [r"$128^2$", r"$64^2$", r"$32^2$", r"$16^2$", r"$\mathbf{z} \in \mathbb{R}^{h \times w \times d}$"]
    b_cols = ["#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8"]
    t_cols = ["#1E3A8A", "#1E3A8A", "#1E3A8A", "#FFFFFF", "#FFFFFF"]

    for bx, by, bw, bh, bl, bd, bc, tc in zip(b_xs, b_ys, b_ws, b_hs, b_lbls, b_dims, b_cols, t_cols):
        r_ = patches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=bc, edgecolor="#1E40AF", linewidth=0.5, zorder=2
        )
        ax.add_patch(r_)
        ax.text(bx + bw / 2, by + bh / 2 + 0.6, bl,
                fontsize=4.4, fontweight="bold", ha="center", va="center", color=tc, zorder=3)
        ax.text(bx + bw / 2, by + bh / 2 - 0.8, bd,
                fontsize=3.9, ha="center", va="center", color=tc, zorder=3)

    # Inter-block horizontal arrows
    for i in range(4):
        ax.annotate(
            "", xy=(b_xs[i + 1], b_ys[i + 1] + b_hs[i + 1] / 2),
            xytext=(b_xs[i] + b_ws[i], b_ys[i] + b_hs[i] / 2),
            arrowprops=dict(arrowstyle="-|>", color="#1E40AF", lw=0.7, mutation_scale=4.5),
            zorder=4
        )

    # Lateral skip connection line indicated underneath blocks
    ax.annotate(
        "", xy=(81.0, 58.0), xytext=(40.0, 58.8),
        arrowprops=dict(arrowstyle="-|>", color="#0D9488", lw=1.0, linestyle="--",
                        connectionstyle="arc3,rad=-0.12", mutation_scale=5),
        zorder=5
    )
    ax.text(58.0, 57.8, r"Lateral Skip Pathways $\{\mathbf{s}_l\}_{l=1}^L$",
            fontsize=4.6, fontweight="bold", color="#0D9488", ha="center", va="center", zorder=6)

    # Branching connectors from Stage 2 to Stage 3 (Gap from 56.0 down to 51.0: 5.0 units)
    # Left Branch -> Task 1 (Classification)
    ax.annotate(
        "", xy=(25.0, 51.5), xytext=(25.0, 55.5),
        arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.2, mutation_scale=8),
        zorder=5
    )
    ax.text(27.5, 53.5, r"$\mathbf{z} \to$ GAP", fontsize=4.8, fontweight="bold",
            color="#92400E", ha="left", va="center")

    # Right Branch -> Task 2 (Segmentation)
    ax.annotate(
        "", xy=(74.0, 51.5), xytext=(74.0, 55.5),
        arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.2, mutation_scale=8),
        zorder=5
    )
    ax.text(76.5, 53.5, r"$\mathbf{z} + \{\mathbf{s}_l\} \to \text{Dec}_\phi$",
            fontsize=4.8, fontweight="bold", color="#065F46", ha="left", va="center")

    # =========================================================================
    # STAGE 3: DECOUPLED MULTI-TASK HEADS (y: 26.0 to 51.0, h: 25.0)
    # =========================================================================
    # Task 1: Classification Head (Left Card)
    card(2.0, 26.0, 46.5, 25.0, "#FFFBEB", "#D97706",
         "Task 1: Classification Head", "#92400E", title_size=5.1)

    ax.text(4.5, 47.6, r"$\text{GAP} \to \text{MLP} \to \hat{\mathbf{y}}_{cls} \in \Delta^K$",
            fontsize=4.4, color="#78350F", va="top")

    # Miniature probability bar chart (y from 33.5 to 43.5)
    ax_bar = ax.inset_axes([0.055, 0.335, 0.38, 0.100], transform=ax.transAxes)
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

    # Classification Loss Tag (y: 27.0 to 30.0)
    loss_box1 = patches.FancyBboxPatch(
        (4.5, 27.0), 41.5, 3.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#FEF3C7", edgecolor="#D97706", linewidth=0.5, zorder=2
    )
    ax.add_patch(loss_box1)
    ax.text(25.2, 28.5, r"Loss $\mathcal{L}_{cls}$ (Weighted CCE)",
            fontsize=4.6, fontweight="bold", ha="center", va="center", color="#92400E", zorder=3)

    # Task 2: Segmentation Head (Right Card)
    card(51.5, 26.0, 46.5, 25.0, "#ECFDF5", "#059669",
         "Task 2: Segmentation Head", "#065F46", title_size=5.1)

    ax.text(54.0, 47.6, r"$\text{Dec}_\phi(\mathbf{z}, \{\mathbf{s}_l\}) \to \mathbf{l}_{seg} \in \mathbb{R}^{H \times W \times M}$",
            fontsize=4.1, color="#064E3B", va="top")

    # Miniature segmentation overlay thumbnail (y from 33.5 to 43.5)
    ax_seg = ax.inset_axes([0.555, 0.335, 0.38, 0.100], transform=ax.transAxes)
    over = _overlay(recs[1]["img_rgb"], recs[1]["gt"], recs[1]["seg_classes"])
    ax_seg.imshow(over)
    ax_seg.axis("off")
    for sp in ax_seg.spines.values():
        sp.set_visible(True)
        sp.set_color("#059669")
        sp.set_linewidth(0.5)

    # Segmentation Loss Tag (y: 27.0 to 30.0)
    loss_box2 = patches.FancyBboxPatch(
        (54.0, 27.0), 41.5, 3.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#D1FAE5", edgecolor="#059669", linewidth=0.5, zorder=2
    )
    ax.add_patch(loss_box2)
    ax.text(74.7, 28.5, r"Loss $\mathcal{L}_{seg}$ (Mask CE/BCE)",
            fontsize=4.6, fontweight="bold", ha="center", va="center", color="#065F46", zorder=3)

    # Converging connectors into Stage 4 (Gap from 26.0 down to 21.0: 5.0 units)
    ax.annotate(
        "", xy=(25.0, 21.5), xytext=(25.0, 25.5),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, mutation_scale=8),
        zorder=5
    )
    ax.annotate(
        "", xy=(74.0, 21.5), xytext=(74.0, 25.5),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, mutation_scale=8),
        zorder=5
    )

    # =========================================================================
    # STAGE 4: JOINT OPTIMIZATION & EVALUATION PROTOCOL (y: 1.5 to 21.0, h: 19.5)
    # =========================================================================
    card(2.0, 1.5, 96.0, 19.5, "#F5F3FF", "#7C3AED",
         "Stage 4: Joint Optimization & Metric Disaggregation", "#5B21B6", title_size=5.2)

    # GradNorm Formulation Strip (y: 10.4 to 16.6, h: 6.2)
    g_strip = patches.FancyBboxPatch(
        (4.5, 10.4), 91.0, 6.2,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor="#EDE9FE", edgecolor="#8B5CF6", linewidth=0.5, zorder=2
    )
    ax.add_patch(g_strip)
    ax.text(50.0, 14.8, r"$\mathcal{L}_{total} = w_{cls}(t)\,\mathcal{L}_{cls} + w_{seg}(t)\,\mathcal{L}_{seg} \quad (\text{Static or Adaptive Balancing})$",
            fontsize=4.9, fontweight="bold", color="#4C1D95", ha="center", va="center", zorder=3)
    ax.text(50.0, 12.2, r"Adaptive GradNorm: $\mathcal{L}_{grad}(t) = \sum_i |\, G_W^{(i)}(t) - \bar{G}_W(t)[r_i(t)]^\alpha | \to (w_{cls}, w_{seg})$",
            fontsize=4.5, color="#5B21B6", ha="center", va="center", zorder=3)

    # Audit Guard Strip: Clean Scientific Metric Formulation (y: 2.6 to 8.8, h: 6.2)
    a_strip = patches.FancyBboxPatch(
        (4.5, 2.6), 91.0, 6.2,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor="#FFFFFF", edgecolor="#A78BFA", linewidth=0.5, zorder=2
    )
    ax.add_patch(a_strip)
    ax.text(50.0, 7.0, r"Disaggregated Metric: $\overline{\text{Dice}}_{\text{all}} = \rho \cdot q_{\text{empty}} + (1 - \rho) \cdot \overline{\text{Dice}}_{\text{fg}}$",
            fontsize=4.9, fontweight="bold", color="#4C1D95", ha="center", va="center", zorder=3)
    ax.text(50.0, 4.4, r"Isolates Foreground Lesion Overlap ($\overline{\text{Dice}}_{\text{fg}}$) from Negative-Case Floor ($\rho$)",
            fontsize=4.5, color="#6D28D9", ha="center", va="center", zorder=3)

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
