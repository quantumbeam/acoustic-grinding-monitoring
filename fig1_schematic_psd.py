#!/usr/bin/env python3
"""
Fig.1 plot for a D50-estimation paper.

Default:
- Schematic particle size distribution (smooth PDF)
- Single vertical line indicating D50
- No CDF (to avoid confusion with full PSD reconstruction)
- Horizontal-ish aspect ratio for Fig.1 right panel
- Vector outputs: PDF + SVG (+ PNG)

Optional:
- --show_cdf to add a thin dashed CDF (still schematic)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def lognormal_pdf(x: np.ndarray, median: float, gsd: float) -> np.ndarray:
    mu = np.log(median)
    sigma = np.log(gsd)
    return (1.0 / (x * sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -((np.log(x) - mu) ** 2) / (2.0 * sigma**2)
    )


def mixture_pdf(x: np.ndarray, params):
    y = np.zeros_like(x, dtype=float)
    wsum = 0.0
    for w, med, gsd in params:
        y += w * lognormal_pdf(x, med, gsd)
        wsum += w
    if wsum > 0:
        y /= wsum
    return y


def normalize_peak(y: np.ndarray) -> np.ndarray:
    m = float(np.max(y))
    return y / m if m > 0 else y


def cdf_from_pdf(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    area = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * dx)
    area = np.concatenate([[0.0], area])
    if area[-1] > 0:
        area /= area[-1]
    return area


def main() -> None:
    p = argparse.ArgumentParser()

    # --- Defaults tuned for Fig.1 (D50 paper) ---
    p.add_argument("--outdir", type=str, default="results/paper_plots")
    p.add_argument("--basename", type=str, default="fig1_d50_schematic")

    # Slightly horizontal for Fig.1 right panel
    p.add_argument("--fig_w_in", type=float, default=3.6)
    p.add_argument("--fig_h_in", type=float, default=3.0)

    # X range
    p.add_argument("--xmin_um", type=float, default=5.0)
    p.add_argument("--xmax_um", type=float, default=350.0)

    # Mixture parameters (purely schematic)
    p.add_argument("--w1", type=float, default=0.7)
    p.add_argument("--d50_1", type=float, default=130.0)
    p.add_argument("--gsd_1", type=float, default=1.6)

    p.add_argument("--w2", type=float, default=0.3)
    p.add_argument("--d50_2", type=float, default=60.0)
    p.add_argument("--gsd_2", type=float, default=1.4)

    # Styling
    p.add_argument("--linewidth", type=float, default=1.8)
    p.add_argument("--d50_linewidth", type=float, default=1.6)
    p.add_argument("--d50_color", type=str, default="#d62728")

    # Optional extras
    p.add_argument("--show_axis_labels", action="store_true", default=False)
    p.add_argument("--show_cdf", action="store_true", default=False)

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # X grid
    x = np.linspace(args.xmin_um, args.xmax_um, 900)

    params = [
        (args.w1, args.d50_1, args.gsd_1),
        (args.w2, args.d50_2, args.gsd_2),
    ]

    pdf = mixture_pdf(x, params)
    pdf_n = normalize_peak(pdf)

    # Representative D50 (weighted median proxy for schematic use)
    d50 = args.w1 * args.d50_1 + args.w2 * args.d50_2

    # Optional CDF
    if args.show_cdf:
        cdf = cdf_from_pdf(x, pdf)

    # Matplotlib style (clean, Fig.1-friendly)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(args.fig_w_in, args.fig_h_in))

    # --- Plot ---
    ax.plot(x, pdf_n, linewidth=args.linewidth)
    ax.axvline(
        d50,
        linestyle="--",
        linewidth=args.d50_linewidth,
        color=args.d50_color,
    )
    ax.annotate(
        "D50",
        xy=(d50, 1.0),
        xycoords="data",
        xytext=(6, 6),
        textcoords="offset points",
        color=args.d50_color,
        ha="left",
        va="bottom",
        fontsize=13,
        clip_on=False,
    )

    if args.show_cdf:
        ax.plot(x, cdf, linestyle=":", linewidth=1.4, alpha=0.8)

    # Axes formatting
    ax.set_xlim(args.xmin_um, args.xmax_um)
    ax.set_ylim(0.0, 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.tick_params(axis="x", which="both", direction="out", length=3)

    if args.show_axis_labels:
        ax.set_xlabel("Particle diameter (µm)")
    else:
        ax.set_xlabel("")

    fig.tight_layout(pad=0.25)

    # Outputs
    pdf_path = outdir / f"{args.basename}.pdf"
    svg_path = outdir / f"{args.basename}.svg"
    png_path = outdir / f"{args.basename}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved:\n- {pdf_path}\n- {svg_path}\n- {png_path}")


if __name__ == "__main__":
    main()
