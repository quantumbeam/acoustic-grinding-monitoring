"""Referee 2 (comment 1): power-law sensitivity underlying the exponent alpha.

Section 4.2 quotes the exponent of S_AE ~ D50^alpha as a screening index for
judging whether threshold-based stopping is likely to work for a given powder.
This script plots the log-log calibration data behind those numbers so that the
fit can be inspected in the ESI.

The calibration points are the seven grinding durations, averaged over the three
replicates.  S_AE is the final value of the 4-point moving average of each run,
matching Eq. (2) and Section 3.2, and D50 is the median diameter from laser
diffraction -- i.e. exactly the data set used for the monotone Bernstein
regression in Section 3.2, read here from the dataset that
``run_03_train_particle_size_ae_model.py`` exports, so that no raw data are
required.

Outputs (analysis_results/run_11_plot_material_screening_powerlaw/):
    material_screening_loglog.pdf/.png   log-log scatter with the fitted lines
    material_screening_loglog_fit.csv    alpha, R^2 and the plotted points
"""

import collections
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

MATERIALS = ["NaCl", "Citricacid", "MSG"]
MATERIAL_LABELS = {"NaCl": "NaCl", "Citricacid": "Citric acid", "MSG": "MSG"}
MATERIAL_STYLE = {
    "NaCl": dict(color="black", marker="o"),
    "Citricacid": dict(color="#E74C3C", marker="x"),
    "MSG": dict(color="#2E6DB4", marker="^"),
}
DATASET = os.path.join(
    "analysis_results",
    "run_03_train_particle_size_ae_model",
    "training_trials_particle_size_ae_dataset.csv",
)
OUTPUT_DIR = os.path.join("analysis_results", "run_11_plot_material_screening_powerlaw")


def load_calibration_points():
    """Return {material: (D50 array, S_AE array)} averaged over the replicates."""
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(DATASET, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["reagent"]][float(row["grind_min"])].append(
                (float(row["particle_size"]), float(row["ae_power_mV2"]))
            )

    points = {}
    for material in MATERIALS:
        d50, s_ae = [], []
        for minutes in sorted(grouped[material]):
            replicates = grouped[material][minutes]
            d50.append(np.mean([value for value, _ in replicates]))
            s_ae.append(np.mean([value for _, value in replicates]))
        points[material] = (np.array(d50), np.array(s_ae))
    return points


def fit_power_law(d50, s_ae):
    slope, intercept = np.polyfit(np.log10(d50), np.log10(s_ae), 1)
    r = np.corrcoef(np.log10(d50), np.log10(s_ae))[0, 1]
    return slope, intercept, r ** 2


def configure_style():
    try:
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.titlesize": 20,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "lines.linewidth": 1.4,
            "lines.markersize": 7,
            "legend.frameon": False,
        }
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    points = load_calibration_points()
    configure_style()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    records = []
    for material in MATERIALS:
        d50, s_ae = points[material]
        slope, intercept, r2 = fit_power_law(d50, s_ae)
        style = MATERIAL_STYLE[material]
        label = (
            f"{MATERIAL_LABELS[material]}  "
            r"$\alpha$" + f" = {slope:.2f} ($R^2$ = {r2:.3f})"
        )
        ax.plot(d50, s_ae, linestyle="none", label=label, **style)
        grid = np.logspace(np.log10(d50.min()), np.log10(d50.max()), 50)
        ax.plot(grid, 10 ** intercept * grid ** slope,
                color=style["color"], linestyle="--", linewidth=1.2)
        for size, power in zip(d50, s_ae):
            records.append([MATERIAL_LABELS[material], size, power, slope, r2])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D_{50}$ ($\mu$m)")
    ax.set_ylabel(r"$S_\mathrm{AE}$ (a.u.)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    for extension in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"material_screening_loglog.{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    csv_path = os.path.join(OUTPUT_DIR, "material_screening_loglog_fit.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material", "D50_um", "S_AE_mV2", "alpha", "R2"])
        writer.writerows(records)
    print(f"wrote {csv_path}")

    for material in MATERIALS:
        d50, s_ae = points[material]
        slope, _, r2 = fit_power_law(d50, s_ae)
        print(f"{MATERIAL_LABELS[material]:12s} alpha={slope:5.2f}  R^2={r2:.3f}  "
              f"D50 {d50[0]:.1f} -> {d50[-1]:.1f} um")


if __name__ == "__main__":
    main()
