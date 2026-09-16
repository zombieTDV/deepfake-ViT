"""
Generate crisp, publication-grade architectural PNG diagrams corresponding to all Mermaid diagrams.
Saves to:
  - experiments/results/diagrams/diagram1_forensics_pipeline.png
  - docs/reports/figures/model_architecture_diagram.png
  - docs/reports/figures/forensic_signal_pipeline.png
  - experiments/results/diagrams/diagram2_artifact_scale_decision.png
  - experiments/results/diagrams/diagram3_data_regime_scaling.png
"""

from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(r"C:\document\Study documents\deepfake-ViT")
OUT_DIR = REPO_ROOT / "experiments" / "results" / "diagrams"
DOCS_FIG_DIR = REPO_ROOT / "docs" / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#bdc3c7'
plt.rcParams['axes.linewidth'] = 0.8

PRIMARY = "#1b3a4b"
SECONDARY = "#2980b9"
ACCENT = "#27ae60"
LIGHT_BG = "#f8f9fa"
DARK_TEXT = "#2c3e50"
BORDER = "#bdc3c7"

# -------------------------------------------------------------------------
# DIAGRAM 1: End-to-End Forensics Pipeline & Model Architecture (No Ensemble)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.2), dpi=300)
ax.set_xlim(0, 105)
ax.set_ylim(0, 50)
ax.axis('off')

# Title
ax.text(52.5, 48.0, "End-to-End Dual Foundation Forensics Pipeline (Meta DINOv3 ViT vs. ConvNeXt)", 
        fontsize=12, fontweight='bold', ha='center', color=PRIMARY)

# Box 1: Input Facial Crop
b1 = patches.FancyBboxPatch((2, 16.5), 18, 18, boxstyle="round,pad=1", 
                            fc=LIGHT_BG, ec=PRIMARY, lw=1.5)
ax.add_patch(b1)
ax.text(11, 28.5, "Input Facial Crop", fontsize=9.5, fontweight='bold', ha='center', color=PRIMARY)
ax.text(11, 24.5, "256x256x3 RGB", fontsize=8.5, ha='center', color=DARK_TEXT)
ax.text(11, 20.5, "ImageNet Normalization", fontsize=7.5, ha='center', color="#7f8c8d")
ax.text(11, 17.0, "Zero-Leakage Test Crop", fontsize=7, ha='center', color="#7f8c8d")

vit_y = 28.5
vit_h = 15.5
vit_mid = vit_y + vit_h / 2.0  # 36.25

cnn_y = 6.8
cnn_h = 15.5
cnn_mid = cnn_y + cnn_h / 2.0  # 14.55

# Arrows from Input to Backbones
ax.annotate("", xy=(25, vit_mid), xytext=(20, 29),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.6, mutation_scale=12))
ax.annotate("", xy=(25, cnn_mid), xytext=(20, 22),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6, mutation_scale=12))

# Box 2A: ViT Backbone
b2a = patches.FancyBboxPatch((25, vit_y), 26, vit_h, boxstyle="round,pad=1", 
                             fc="#ebf5fb", ec=SECONDARY, lw=1.5)
ax.add_patch(b2a)
ax.text(38, vit_y + 11.5, "Meta DINOv3 ViT-Plus A1", fontsize=9, fontweight='bold', ha='center', color=SECONDARY)
ax.text(38, vit_y + 8.2, "28.69M Params | Patch 16x16", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(38, vit_y + 5.0, "12 Layers | 6 Heads | Dim 384", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(38, vit_y + 2.0, "SwiGLU Gated MLP | SDPA", fontsize=7.5, fontweight='bold', ha='center', color=PRIMARY)

# Box 2B: ConvNeXt Backbone
b2b = patches.FancyBboxPatch((25, cnn_y), 26, cnn_h, boxstyle="round,pad=1", 
                             fc="#eafaf1", ec=ACCENT, lw=1.5)
ax.add_patch(b2b)
ax.text(38, cnn_y + 11.5, "Meta DINOv3 ConvNeXt-Tiny", fontsize=9, fontweight='bold', ha='center', color=ACCENT)
ax.text(38, cnn_y + 8.2, "28.12M Params | 4 Stages", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(38, cnn_y + 5.0, "Channels: [96, 192, 384, 768]", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(38, cnn_y + 2.0, "7x7 Depthwise Conv | Inverted", fontsize=7.5, fontweight='bold', ha='center', color=PRIMARY)

# Arrows from Backbones to Heads
ax.annotate("", xy=(56, vit_mid), xytext=(51, vit_mid),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.6, mutation_scale=12))
ax.annotate("", xy=(56, cnn_mid), xytext=(51, cnn_mid),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6, mutation_scale=12))

# Box 3A: ViT Head
b3a = patches.FancyBboxPatch((56, vit_y), 20, vit_h, boxstyle="round,pad=1", 
                             fc=LIGHT_BG, ec=SECONDARY, lw=1.3)
ax.add_patch(b3a)
ax.text(66, vit_y + 11.5, "2-Layer MLP Head", fontsize=8.5, fontweight='bold', ha='center', color=PRIMARY)
ax.text(66, vit_y + 8.2, "LN(384) -> Drop(0.2)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(66, vit_y + 5.0, "Linear(384, 384) -> GELU", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(66, vit_y + 2.0, "Linear(384, 2) | 149k Params", fontsize=7.5, fontweight='bold', ha='center', color=SECONDARY)

# Box 3B: ConvNeXt Head
b3b = patches.FancyBboxPatch((56, cnn_y), 20, cnn_h, boxstyle="round,pad=1", 
                             fc=LIGHT_BG, ec=ACCENT, lw=1.3)
ax.add_patch(b3b)
ax.text(66, cnn_y + 11.5, "2-Layer MLP Head", fontsize=8.5, fontweight='bold', ha='center', color=PRIMARY)
ax.text(66, cnn_y + 8.2, "LN(768) -> Drop(0.2)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(66, cnn_y + 5.0, "Linear(768, 384) -> GELU", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(66, cnn_y + 2.0, "Linear(384, 2) | 298k Params", fontsize=7.5, fontweight='bold', ha='center', color=ACCENT)

# Arrows from Heads to Predictions
ax.annotate("", xy=(81, vit_mid), xytext=(76, vit_mid),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.6, mutation_scale=12))
ax.annotate("", xy=(81, cnn_mid), xytext=(76, cnn_mid),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6, mutation_scale=12))

# Box 4A: ViT Prediction
b4a = patches.FancyBboxPatch((81, vit_y), 22, vit_h, boxstyle="round,pad=1", 
                             fc="#ebf5fb", ec=SECONDARY, lw=1.6)
ax.add_patch(b4a)
ax.text(92, vit_y + 11.5, "ViT Binary Prediction", fontsize=8.5, fontweight='bold', ha='center', color=SECONDARY)
ax.text(92, vit_y + 8.2, "P(Fake | X) >= 0.50", fontsize=8, fontweight='bold', ha='center', color=PRIMARY)
ax.text(92, vit_y + 5.0, "Test Acc: 98.53% | AUC: 99.86%", fontsize=7.5, fontweight='bold', ha='center', color=ACCENT)
ax.text(92, vit_y + 2.0, "Fake Recall: 99.04% (FN=100)", fontsize=7.5, ha='center', color=DARK_TEXT)

# Box 4B: ConvNeXt Prediction
b4b = patches.FancyBboxPatch((81, cnn_y), 22, cnn_h, boxstyle="round,pad=1", 
                             fc="#eafaf1", ec=ACCENT, lw=1.6)
ax.add_patch(b4b)
ax.text(92, cnn_y + 11.5, "CNN Binary Prediction", fontsize=8.5, fontweight='bold', ha='center', color=ACCENT)
ax.text(92, cnn_y + 8.2, "P(Fake | X) >= 0.50", fontsize=8, fontweight='bold', ha='center', color=PRIMARY)
ax.text(92, cnn_y + 5.0, "Test Acc: 99.49% | AUC: 99.99%", fontsize=7.5, fontweight='bold', ha='center', color=ACCENT)
ax.text(92, cnn_y + 2.0, "Real Spec: 99.79% (FP=22)", fontsize=7.5, ha='center', color=DARK_TEXT)

# Capacity Parity Banner in center
bp = patches.FancyBboxPatch((26, 23.7), 54, 3.2, boxstyle="round,pad=0.3",
                            fc="#fef9e7", ec="#f39c12", lw=1.2)
ax.add_patch(bp)
ax.text(53, 24.8, "Controlled Capacity Parity (<2% Delta): ViT 28.69M vs. ConvNeXt 28.12M",
        fontsize=8, fontweight='bold', ha='center', color="#b7950b")

plt.tight_layout()

# Save Diagram 1 to all target destinations
p1 = OUT_DIR / "diagram1_forensics_pipeline.png"
p1_model = DOCS_FIG_DIR / "model_architecture_diagram.png"
p1_flow = DOCS_FIG_DIR / "forensic_signal_pipeline.png"

plt.savefig(p1, bbox_inches='tight', dpi=300)
shutil.copy2(p1, p1_model)
shutil.copy2(p1, p1_flow)
plt.close()

print(f"Generated: {p1}")
print(f"Copied to: {p1_model}")
print(f"Copied to: {p1_flow}")


# -------------------------------------------------------------------------
# DIAGRAM 2: Artifact Scale & Architecture Selection
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 45)
ax.axis('off')

ax.text(50, 42, "Forensic Artifact Scale & Architecture Alignment", 
        fontsize=11.5, fontweight='bold', ha='center', color=PRIMARY)

# Root Box
r = patches.FancyBboxPatch((35, 28), 30, 10, boxstyle="round,pad=1", fc=LIGHT_BG, ec=PRIMARY, lw=1.5)
ax.add_patch(r)
ax.text(50, 34, "Deepfake Detection Task", fontsize=9, fontweight='bold', ha='center', color=PRIMARY)
ax.text(50, 30, "Identify Artifact Profile", fontsize=8, ha='center', color=DARK_TEXT)

# Left Branch: Global
ax.annotate("", xy=(22, 23), xytext=(40, 28),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.5, mutation_scale=12))
b_left = patches.FancyBboxPatch((4, 7), 36, 16, boxstyle="round,pad=1", fc="#ebf5fb", ec=SECONDARY, lw=1.5)
ax.add_patch(b_left)
ax.text(22, 19, "Global / Semantic Artifacts", fontsize=8.5, fontweight='bold', ha='center', color=SECONDARY)
ax.text(22, 16, "• Whole-Face Diffusion (DiT, PixArt, SD)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(22, 13, "• Unconditional GANs (StyleGAN2/3)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(22, 10, "• Illumination & Iris Symmetry Asymmetry", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(22, 7.5, "=> Recommended: Meta DINOv3 ViT-Plus A1 (28.69M)", fontsize=7, fontweight='bold', ha='center', color=PRIMARY)

# Right Branch: Local
ax.annotate("", xy=(78, 23), xytext=(60, 28),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5, mutation_scale=12))
b_right = patches.FancyBboxPatch((60, 7), 36, 16, boxstyle="round,pad=1", fc="#eafaf1", ec=ACCENT, lw=1.5)
ax.add_patch(b_right)
ax.text(78, 19, "Local / Boundary Artifacts", fontsize=8.5, fontweight='bold', ha='center', color=ACCENT)
ax.text(78, 16, "• FaceSwap Blending Seams (InSwap, SimSwap)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(78, 13, "• High-Frequency Noise & JPEG Artifacts", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(78, 10, "• Real-Time Video Streaming Constraints", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(78, 7.5, "=> Recommended: Meta DINOv3 ConvNeXt-Tiny (28.12M)", fontsize=7, fontweight='bold', ha='center', color=PRIMARY)

plt.tight_layout()
p2 = OUT_DIR / "diagram2_artifact_scale_decision.png"
plt.savefig(p2, bbox_inches='tight', dpi=300)
plt.close()
print(f"Generated: {p2}")


# -------------------------------------------------------------------------
# DIAGRAM 3: Small vs. Large Data Regime Scaling Map
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 45)
ax.axis('off')

ax.text(50, 42, "Dataset Scale & Pretraining Scaling Regime Map", 
        fontsize=11.5, fontweight='bold', ha='center', color=PRIMARY)

# Left Side: Small Data
b_s = patches.FancyBboxPatch((4, 4), 43, 34, boxstyle="round,pad=1", fc=LIGHT_BG, ec=PRIMARY, lw=1.5)
ax.add_patch(b_s)
ax.text(25.5, 34, "Small Data Regime (< 10k Samples)", fontsize=9, fontweight='bold', ha='center', color=PRIMARY)

b_s1 = patches.FancyBboxPatch((6, 20), 39, 11, boxstyle="round,pad=0.5", fc="#fadbd8", ec="#e74c3c", lw=1)
ax.add_patch(b_s1)
ax.text(25.5, 27, "Training From Scratch", fontsize=8, fontweight='bold', ha='center', color="#c0392b")
ax.text(25.5, 23, "• CNN: Converges stably (Inductive Bias)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(25.5, 20.5, "• ViT: Fails / Overfits severely", fontsize=7.5, fontweight='bold', ha='center', color="#c0392b")

b_s2 = patches.FancyBboxPatch((6, 6), 39, 11, boxstyle="round,pad=0.5", fc="#d5f5e3", ec=ACCENT, lw=1)
ax.add_patch(b_s2)
ax.text(25.5, 13, "Pretrained Foundation Model", fontsize=8, fontweight='bold', ha='center', color=ACCENT)
ax.text(25.5, 9.5, "• CNN: Effective, fast fine-tuning", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(25.5, 7, "• ViT: Strong performance (Features Transfer)", fontsize=7.5, fontweight='bold', ha='center', color=ACCENT)

# Right Side: Large Data
b_l = patches.FancyBboxPatch((53, 4), 43, 34, boxstyle="round,pad=1", fc=LIGHT_BG, ec=PRIMARY, lw=1.5)
ax.add_patch(b_l)
ax.text(74.5, 34, "Large Data Regime (> 100k Samples)", fontsize=9, fontweight='bold', ha='center', color=PRIMARY)

b_l1 = patches.FancyBboxPatch((55, 20), 39, 11, boxstyle="round,pad=0.5", fc="#fef9e7", ec="#f39c12", lw=1)
ax.add_patch(b_l1)
ax.text(74.5, 27, "Training From Scratch", fontsize=8, fontweight='bold', ha='center', color="#d35400")
ax.text(74.5, 23, "• CNN: Competitive, but saturates early", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(74.5, 20.5, "• ViT: Learns 2D geometry directly", fontsize=7.5, ha='center', color=DARK_TEXT)

b_l2 = patches.FancyBboxPatch((55, 6), 39, 11, boxstyle="round,pad=0.5", fc="#d4e6f1", ec=SECONDARY, lw=1)
ax.add_patch(b_l2)
ax.text(74.5, 13, "Pretrained Foundation Model", fontsize=8, fontweight='bold', ha='center', color=SECONDARY)
ax.text(74.5, 9.5, "• CNN: 99.49% Acc (Superior Real Specificity)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(74.5, 7, "• ViT: 98.53% Acc (Superior Diffusion Recall)", fontsize=7.5, fontweight='bold', ha='center', color=SECONDARY)

plt.tight_layout()
p3 = OUT_DIR / "diagram3_data_regime_scaling.png"
plt.savefig(p3, bbox_inches='tight', dpi=300)
plt.close()
print(f"Generated: {p3}")
