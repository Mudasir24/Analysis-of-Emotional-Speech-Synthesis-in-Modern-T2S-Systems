"""
PCA + t-SNE + LDA Visualization of Synthetic WavLM Embeddings
===============================================================
Run from your tts_emotion folder:
    python visualize.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CSV_PATH   = r"C:\Users\JIYA KHANNA\OneDrive\Desktop\tts_emotion\synthetic_audio\embeddings_summary.csv"
OUTPUT_DIR = r"C:\Users\JIYA KHANNA\OneDrive\Desktop\tts_emotion"

EMOTION_COLORS = {
    "Neutral":  "#7f7f7f",
    "Angry":    "#d62728",
    "Happy":    "#2ca02c",
    "Sad":      "#1f77b4",
    "Surprise": "#ff7f0e",
}

# ─────────────────────────────────────────────
# LOAD EMBEDDINGS
# ─────────────────────────────────────────────

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} entries found")

print("Loading .npy embeddings...")
embeddings, labels = [], []
skipped = 0

for _, row in df.iterrows():
    npy_path = row["npy_path"]
    if not os.path.exists(npy_path):
        skipped += 1
        continue
    embeddings.append(np.load(npy_path))
    labels.append(row["emotion"])

print(f"  Loaded: {len(embeddings)} | Skipped: {skipped}")

embeddings = np.array(embeddings)   # (N, 768)
labels     = np.array(labels)

# ─────────────────────────────────────────────
# NORMALIZE
# ─────────────────────────────────────────────

print("Normalizing...")
scaler = StandardScaler()
X = scaler.fit_transform(embeddings)

# ─────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────

print("Running PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
ev = pca.explained_variance_ratio_ * 100
print(f"  PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%")

# ─────────────────────────────────────────────
# LDA
# ─────────────────────────────────────────────

print("Running LDA...")
# LDA max components = n_classes - 1 = 4 for 5 emotions
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, labels)
lda_ev = lda.explained_variance_ratio_ * 100
print(f"  LD1={lda_ev[0]:.1f}%  LD2={lda_ev[1]:.1f}%")

# ─────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────

print("Running t-SNE (may take ~1 min)...")
tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate=200,
    max_iter=1000,
    random_state=42,
    verbose=1
)
X_tsne = tsne.fit_transform(X)
print("  t-SNE done.")

# ─────────────────────────────────────────────
# PLOT — 1 row, 3 subplots
# ─────────────────────────────────────────────

emotions = list(EMOTION_COLORS.keys())

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle(
    "WavLM Embeddings — Synthetic Audio (Parler-TTS)\n5 Emotions × 200 samples",
    fontsize=14, fontweight="bold"
)

configs = [
    (axes[0], X_pca,  f"PCA  (PC1={ev[0]:.1f}%, PC2={ev[1]:.1f}%)",      "PC1",  "PC2"),
    (axes[1], X_lda,  f"LDA  (LD1={lda_ev[0]:.1f}%, LD2={lda_ev[1]:.1f}%)", "LD1",  "LD2"),
    (axes[2], X_tsne, "t-SNE  (perplexity=40, iter=1000)",                  "Dim1", "Dim2"),
]

for ax, X_2d, title, xlabel, ylabel in configs:
    for emotion in emotions:
        mask = labels == emotion
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=EMOTION_COLORS[emotion],
            label=emotion,
            alpha=0.6, s=25, edgecolors="none"
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Emotion", fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "embeddings_pca_lda_tsne.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to: {out_path}")
plt.show()
