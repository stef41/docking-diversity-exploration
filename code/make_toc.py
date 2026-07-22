"""Generate JCIM TOC graphic (3.33 x 1.875 in, 300+ dpi).

Design: horizontal split.
- Left ~55%: horizontal bar chart of unique interaction profiles per method,
  sorted descending. Top-3 methods highlighted; rest gray. Bar labels show
  method name. A small annotation notes the headline finding.
- Right ~45%: docking-pose thumbnail cropped from fig_docking_3d.png,
  overlaid with a short "diversity of interaction profiles" tagline.

Values (from paper Table 2 + SI Table S11):
  Method            Mean unique interaction profiles (n=70)
  Random            307.6
  IMGEP naive       230.8
  IMGEP adaptive    328.3
  Curiosity-IMGEP   331.5
  UCB / Aff-Div     104.6
  Genetic Alg.      218.9
  MAP-Elites        350.3
  Novelty Search    290.7
  NSGA-II           282.4  (mean of per-target reconstructed profile counts)
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("/tmp/acs-review/revision/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCKING_IMG = Path("/tmp/acs-review/experiments/figures_v3/fig_docking_3d.png")

METHOD_ROWS = [
    ("MAP-Elites",        350.3,  True),
    ("Curiosity-IMGEP",   331.5,  True),
    ("IMGEP (adaptive)",  328.3, False),
    ("Random",            307.6, False),
    ("Novelty Search",    290.7, False),
    ("NSGA-II",           282.4,  True),
    ("IMGEP (naive)",     230.8, False),
    ("Genetic Alg.",      218.9, False),
    ("Aff-Div (UCB)",     104.6, False),
]

# 3.33 x 1.875 in is the ACS JCIM TOC bound. Render at 600 dpi for print.
FIG_W = 3.33
FIG_H = 1.875
DPI = 600

# Colors
COL_HL = "#1f77b4"    # Blue for QD/IMGEP/NSGA highlighted methods
COL_HL2 = "#2ca02c"   # Green for the top-1 (MAP-Elites)
COL_HL3 = "#d62728"   # Red for NSGA-II (Pareto)
COL_BG = "#b0b0b0"    # Gray for baselines/non-QD


def method_color(name):
    if name == "MAP-Elites":
        return COL_HL2
    if name == "Curiosity-IMGEP":
        return COL_HL
    if name == "NSGA-II":
        return COL_HL3
    return COL_BG


def make_toc():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    # 2 panels: left barchart (~54%), right docking pose (~44%)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.15, 1.0],
        left=0.22, right=0.985,   # leave 22% for y-axis method labels
        top=0.86, bottom=0.18,    # room for x-axis label + tagline
        wspace=0.10,
    )

    # ---- Left: horizontal bar chart --------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(METHOD_ROWS))
    vals = [v for _, v, _ in METHOD_ROWS]
    labels = [n for n, _, _ in METHOD_ROWS]
    colors = [method_color(n) for n, _, _ in METHOD_ROWS]

    ax.barh(ys, vals, color=colors, edgecolor="none", height=0.72)
    ax.invert_yaxis()
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=5.0)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_xlim(0, 405)
    ax.tick_params(axis="x", labelsize=4.6, length=2, pad=1)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("Unique interaction profiles",
                  fontsize=5.2, labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.grid(axis="x", color="0.85", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Annotate top-3 highlighted bars with their values
    for i, (name, v, hl) in enumerate(METHOD_ROWS):
        if hl:
            ax.text(v + 6, i, f"{v:.0f}", fontsize=4.6, va="center",
                    color=colors[i], fontweight="bold")

    # ---- Right: docking-pose thumbnail -----------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    if DOCKING_IMG.exists():
        img = mpimg.imread(DOCKING_IMG)
        # Center-crop to roughly square before resizing to panel aspect
        h, w = img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        cropped = img[y0:y0 + side, x0:x0 + side]
        ax2.imshow(cropped, aspect="auto")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)

    # Header tagline spanning the whole figure top
    fig.text(
        0.5, 0.955,
        "Diversity-aware search enumerates broader residue-interaction profiles",
        fontsize=5.4, ha="center", va="top", fontweight="bold",
    )

    # Save both PDF (vector-preferred for ACS) and PNG at 600 dpi.
    pdf_path = OUT_DIR / "toc_graphic.pdf"
    png_path = OUT_DIR / "toc_graphic.png"
    tiff_path = OUT_DIR / "toc_graphic.tif"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=DPI)
    # TIFF at 300 dpi (ACS accepts TIFF; smaller file)
    fig.savefig(tiff_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {pdf_path}  ({pdf_path.stat().st_size:,} B)")
    print(f"Wrote {png_path}  ({png_path.stat().st_size:,} B)")
    print(f"Wrote {tiff_path} ({tiff_path.stat().st_size:,} B)")


if __name__ == "__main__":
    make_toc()
