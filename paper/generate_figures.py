#!/usr/bin/env python3
"""
Generate publication-quality figures for the paper:
Figure 1: Visual and Quantitative Impact of Macenko Stain Normalization
Figure 2: The Empty-Mask Background-Dice Inflation Fallacy in Sparse Pathologies
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Ensure output directory exists
OUT_DIR = Path("/home/apath/Work/temp/final/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

# ==============================================================================
# FIGURE 1: Macenko Stain Normalization Artifact & Ablation Impact
# ==============================================================================
def generate_figure_1():
    print("Generating Figure 1: Macenko Stain Normalization...")
    raw_path = Path("/home/apath/Work/temp/final/data/PANDA/train_images/train_images/3dab3238ef15a3c5b3d43e0b777073a5.png")
    mac_path = Path("/home/apath/Work/temp/final/data/PANDA/preprocessed_macenko_fixed/images/3dab3238ef15a3c5b3d43e0b777073a5.png")

    img_raw = np.array(Image.open(raw_path).convert('RGB'))
    img_mac = np.array(Image.open(mac_path).convert('RGB'))

    # If sizes differ, resize macenko to raw
    if img_raw.shape != img_mac.shape:
        img_mac = np.array(Image.fromarray(img_mac).resize((img_raw.shape[1], img_raw.shape[0])))

    # Crop an interesting central 160x160 region showing glandular structure
    h, w, _ = img_raw.shape
    ch, cw = h // 2, w // 2
    crop_size = 140
    r1, r2 = ch - crop_size // 2, ch + crop_size // 2
    c1, c2 = cw - crop_size // 2, cw + crop_size // 2
    
    crop_raw = img_raw[r1:r2, c1:c2]
    crop_mac = img_mac[r1:r2, c1:c2]

    # Compute difference heatmap
    diff = np.mean(np.abs(crop_raw.astype(float) - crop_mac.astype(float)), axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.7), dpi=300)

    axes[0].imshow(crop_raw)
    axes[0].set_title("(a) Raw Biopsy Tile\n(Native Chromatin)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(crop_mac)
    axes[1].set_title("(b) Macenko-Normalized\n(Optical Density Filtered)", fontweight='bold')
    axes[1].axis('off')

    im_diff = axes[2].imshow(diff, cmap='magma')
    axes[2].set_title("(c) Absolute Texture\nDiscrepancy Map", fontweight='bold')
    axes[2].axis('off')
    cbar = fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    # Panel D: Bar chart of empirical classification gain when removing Macenko
    datasets = ['PANDA\n(ISUP 0–5)', 'PanNuke\n(19 tissues)']
    mac_acc = [34.70, 96.68]
    raw_acc = [40.21, 99.36]
    x = np.arange(len(datasets))
    width = 0.32

    rects1 = axes[3].bar(x - width/2, mac_acc, width, label='Macenko ON', color='#6baed6', edgecolor='black', linewidth=0.8)
    rects2 = axes[3].bar(x + width/2, raw_acc, width, label='Raw (Macenko OFF)', color='#238b45', edgecolor='black', linewidth=0.8)

    axes[3].set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    axes[3].set_title('(d) Ablation Accuracy Gain\n(+5.51% PANDA, +2.68% PanNuke)', fontweight='bold')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(datasets)
    axes[3].set_ylim([20, 105])
    axes[3].legend(loc='lower right', frameon=True, framealpha=0.9)
    axes[3].grid(axis='y', linestyle='--', alpha=0.5)

    # Add delta labels on top of bars
    for i in range(len(datasets)):
        diff_val = raw_acc[i] - mac_acc[i]
        axes[3].annotate(f"+{diff_val:.2f}%",
                         xy=(x[i] + width/2, raw_acc[i] + 1.5),
                         ha='center', va='bottom', fontsize=8, fontweight='bold', color='#00441b')

    plt.tight_layout()
    fig_pdf = OUT_DIR / "fig1_macenko_artifact.pdf"
    fig_png = OUT_DIR / "fig1_macenko_artifact.png"
    plt.savefig(fig_pdf, bbox_inches='tight')
    plt.savefig(fig_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_pdf} and {fig_png}")


# ==============================================================================
# FIGURE 2: Empty-Mask Background-Dice Inflation Fallacy
# ==============================================================================
def generate_figure_2():
    print("Generating Figure 2: Empty Mask Dice Fallacy...")
    neg_img_path = Path("/home/apath/Work/temp/final/data/SIIM/preprocessed/images/1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027.png")
    pos_img_path = Path("/home/apath/Work/temp/final/data/SIIM/preprocessed/images/1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951.png")
    pos_mask_path = Path("/home/apath/Work/temp/final/data/SIIM/preprocessed/masks/1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951_mask.png")

    img_neg = np.array(Image.open(neg_img_path).convert('L'))
    img_pos = np.array(Image.open(pos_img_path).convert('L'))
    mask_pos = np.array(Image.open(pos_mask_path).convert('L'))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), dpi=300)

    # Panel A: True Negative Slice
    axes[0].imshow(img_neg, cmap='gray')
    axes[0].set_title("(a) True Negative Slice ($|Y|=0$)\nDice $\\equiv 1.0$ (Zero-Score Convention)", fontweight='bold')
    # Annotation box
    textstr = "Ground Truth: Empty ($|Y|=0$)\nPrediction: Empty ($|\\hat{Y}|=0$)\n" + r"$\mathbf{Dice(Y, \hat{Y}) = 1.0}$" + "\n(Trivially inflated on 78% of cohort)"
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black', linewidth=0.8)
    axes[0].text(0.05, 0.08, textstr, transform=axes[0].transAxes, fontsize=7.5, verticalalignment='bottom', bbox=props)
    axes[0].axis('off')

    # Panel B: True Positive Slice with Mask Contour
    axes[1].imshow(img_pos, cmap='gray')
    # Overlay mask in green
    mask_overlay = np.zeros((*img_pos.shape, 4), dtype=float)
    mask_overlay[mask_pos > 0] = [0.0, 1.0, 0.2, 0.45] # Semi-transparent green
    axes[1].imshow(mask_overlay)
    axes[1].contour(mask_pos > 0, colors=['lime'], linewidths=1.2)
    axes[1].set_title("(b) True Positive Slice ($|Y|>0$)\nTrue Foreground Dice $= 77.74\\%$", fontweight='bold')
    textstr_b = "Pneumothorax Pleural Lesion\nGround Truth Boundary (Green)\n" + r"$\mathbf{Dice_{foreground} = 77.74\%}$" + "\n(Clinical Segmentation Metric)"
    props_b = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black', linewidth=0.8)
    axes[1].text(0.05, 0.08, textstr_b, transform=axes[1].transAxes, fontsize=7.5, verticalalignment='bottom', bbox=props_b)
    axes[1].axis('off')

    # Panel C: Convex combination curve
    rho = np.linspace(0, 1, 200)
    dice_fg = 0.7774
    dice_all = rho * 1.0 + (1 - rho) * dice_fg
    dice_claim = 0.99

    axes[2].plot(rho * 100, dice_all * 100, color='#08519c', linewidth=2.2, label=r'$\overline{\mathrm{Dice}}_{\mathrm{all}} = \rho + (1-\rho) \cdot 77.74\%$')
    axes[2].axhline(y=99.0, color='crimson', linestyle='--', linewidth=1.5, label='Published Claim (99.0% Dice)')
    axes[2].axvline(x=78.0, color='gray', linestyle=':', linewidth=1.5, label=r'SIIM-ACR Negative Rate ($\rho=78\%$)')

    # Mark the SIIM point
    siim_inflated = 0.78 * 1.0 + 0.22 * 0.7774
    axes[2].plot(78.0, siim_inflated * 100, 'o', color='darkblue', markersize=7)
    axes[2].annotate(f"Empty-Mask Inflation\n({siim_inflated*100:.1f}%)",
                     xy=(78.0, siim_inflated * 100),
                     xytext=(45, 87),
                     arrowprops=dict(facecolor='darkblue', shrink=0.08, width=1, headwidth=5),
                     fontweight='bold', fontsize=8, color='darkblue')

    axes[2].set_xlabel('Negative Slice Proportion $\\rho$ (%)', fontweight='bold')
    axes[2].set_ylabel('Reported Macroscopic Dice (%)', fontweight='bold')
    axes[2].set_title('(c) Mathematical Inflation Proof\n(Convex Combination Curve)', fontweight='bold')
    axes[2].set_xlim([0, 100])
    axes[2].set_ylim([70, 101])
    axes[2].legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=7)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_pdf = OUT_DIR / "fig2_empty_mask_dice.pdf"
    fig_png = OUT_DIR / "fig2_empty_mask_dice.png"
    plt.savefig(fig_pdf, bbox_inches='tight')
    plt.savefig(fig_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_pdf} and {fig_png}")


if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
