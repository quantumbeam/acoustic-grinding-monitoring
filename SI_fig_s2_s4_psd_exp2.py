import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = "exp2"
TRIALS = ["1st", "2nd", "3rd"]
GRIND_MINS = [3, 5, 7, 10, 15, 20, 25]
BASE_OUTPUT_DIR = os.path.join("results", "SI_figs", "PSD")
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 9.6
AX_LABEL_FONTSIZE = 15
TICK_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 10


def try_set_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update(
        {
            "axes.labelsize": AX_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "font.size": TICK_FONTSIZE,
        }
    )


def read_distribution(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    sizes = []
    volumes = []
    in_table = False
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not in_table:
                if line.strip().startswith("SizeClasses"):
                    in_table = True
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                size = float(parts[0])
                vol = float(parts[1])
            except ValueError:
                continue
            if np.isfinite(size) and np.isfinite(vol):
                sizes.append(size)
                volumes.append(vol)
    return np.array(sizes, dtype=float), np.array(volumes, dtype=float)


def find_psd_file(material: str, trial: str, grind_min: int) -> str | None:
    dir_path = os.path.join("data", "powder_size_distribution", EXPERIMENT, material, trial)
    pattern = os.path.join(dir_path, f"*grind{grind_min}min*.csv")
    candidates = sorted(glob.glob(pattern))
    return candidates[0] if candidates else None


def build_figure(material: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cmap = plt.get_cmap("rainbow")
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True, sharey=True)
    panel_labels = ["(a)", "(b)", "(c)"]
    manifest_rows = []

    for i, trial in enumerate(TRIALS):
        ax = axes[i]
        for j, grind_min in enumerate(GRIND_MINS):
            psd_file = find_psd_file(material, trial, grind_min)
            if psd_file is None:
                raise FileNotFoundError(f"Missing PSD file: material={material}, trial={trial}, grind={grind_min}min")

            sizes, volumes = read_distribution(psd_file)
            if sizes.size == 0 or volumes.size == 0:
                raise ValueError(f"Distribution table empty: {psd_file}")

            ax.plot(
                sizes,
                volumes,
                color=cmap(j / (len(GRIND_MINS) - 1)),
                linewidth=1.5,
                label=f"{grind_min} min",
            )
            manifest_rows.append(
                {
                    "material": material,
                    "trial": trial,
                    "grind_min": grind_min,
                    "psd_file": psd_file,
                }
            )

        ax.set_xscale("log")
        ax.set_ylabel("Volume density (%)")
        ax.set_title(f"{panel_labels[i]} {trial}", loc="left", fontsize=PANEL_TITLE_FONTSIZE)
        ax.grid(alpha=0.2, linewidth=0.4)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    axes[-1].set_xlabel("Particle diameter (μm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Grinding time",
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    base = f"psd_{material.lower()}_3to25min"
    png_path = os.path.join(output_dir, f"{base}.png")
    pdf_path = os.path.join(output_dir, f"{base}.pdf")
    manifest_path = os.path.join(output_dir, f"{base}_manifest.csv")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create SI PSD figures from exp2 data: "
            "Fig. S2 (NaCl) and Fig. S4 (MSG), each with 1st/2nd/3rd trial panels."
        )
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        default=["NaCl", "MSG"],
        choices=["NaCl", "MSG", "Citricacid"],
        help="Materials to generate.",
    )
    args = parser.parse_args()

    try_set_plot_style()

    fig_meta = {
        "NaCl": os.path.join(BASE_OUTPUT_DIR, "NaCl"),
        "MSG": os.path.join(BASE_OUTPUT_DIR, "MSG"),
        "Citricacid": os.path.join(BASE_OUTPUT_DIR, "Citricacid"),
    }

    for material in args.materials:
        build_figure(material=material, output_dir=fig_meta[material])


if __name__ == "__main__":
    main()
