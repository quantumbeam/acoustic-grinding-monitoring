"""Referee 1 (scientific comment 4): exhaustive robotic grinding vs. manual grinding.

The reviewer asked how the powder compares with a manually ground reference when
the robot is asked to grind "as much as possible".  The revision experiments of
2026-08-20 provide both states for all three materials
(``data/powder_size_distribution/additional_experiments/exhaustive_grinding``),
The manual reference was ground for 10 min, chosen to match the NET grinding time
of the time-scheduled trials rather than their total elapsed duration: a 25 min
trial is 33 AE acquisitions of 12 s of grinding, i.e. 6.6 min of grinding, the
rest of the elapsed time being the periodic scraping.  The operator therefore
received about 1.5 times the robot's net grinding time, so the comparison is
conservative with respect to the robot; the 25 min condition of exp2 provides the
robotic counterpart.

The script also carries the diagnostic for the anomaly found while organising the
data: for citric acid the 60 min robotic product is *coarser* than the 25 min
product of exp2, which is the opposite of NaCl and MSG.  Agglomeration of the
fines and true under-grinding leave different fingerprints, so the script reports
the whole distribution rather than the median alone:

* true under-grinding shifts every percentile up and leaves the coarse tail
  heavier;
* agglomeration removes the fine tail and creates a new mode at tens of
  micrometres while leaving the coarse tail unchanged.

The measurements select the second: for citric acid the coarse tail is
unchanged between the 25 min and 60 min products (5.6 % vs 5.4 % of the volume
above 500 um, D90 +13 %) while the volume below 1 um falls from 14.9 % to 1.2 %
and the mode moves from 6.3 um to 36.2 um.  Both were measured at a dispersion
pressure of 4 bar, which is the maximum the analyser provides, so the fines are
not recoverable by dispersing harder and the difference is not an artefact of
dispersion energy.  NaCl and MSG show no such loss of fines.  The 60 min citric
acid product is therefore reported as agglomerated.

Percentiles use the same estimator as ``run_06_plot_initial_particle_size.py`` (class edges,
interpolation in log-diameter), which reproduces the instrument's own ``Dx (50)``
header entry for every file.

Outputs (analysis_results/run_07_compare_exhaustive_and_manual_grinding/):
    exhaustive_grinding_psd.pdf/.png    three-panel comparison figure
    exhaustive_grinding_per_measurement.csv
    exhaustive_grinding_summary.csv     per-condition mean +/- SD
    exhaustive_grinding_fines.csv       cumulative volume below size thresholds
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_06_plot_initial_particle_size import (
    class_edges,
    check_against_header,
    configure_plot_style,
    distribution_metrics,
    read_measurement,
)

MATERIALS = ["NaCl", "Citricacid", "MSG"]
MATERIAL_LABELS = {"NaCl": "NaCl", "Citricacid": "Citric acid", "MSG": "MSG"}

BASE = os.path.join("data", "powder_size_distribution")
REVISION_DIR = os.path.join(BASE, "additional_experiments", "exhaustive_grinding")
INITIAL_DIR = os.path.join(BASE, "additional_experiments", "initial_psd")

# label -> (glob pattern relative to a material, plot style)
CONDITIONS = {
    "starting powder": (
        os.path.join(INITIAL_DIR, "*_{material}_initial_meas*.csv"),
        dict(color="#F0A202", linestyle="-", linewidth=1.0, alpha=0.9),
    ),
    "robot, 25 min": (
        os.path.join(BASE, "exp2", "{material}", "*", "*grind25min*.csv"),
        dict(color="#2E6DB4", linestyle="--"),
    ),
    "robot, 60 min": (
        os.path.join(REVISION_DIR, "*_{material}_grind_robot60min_meas*.csv"),
        dict(color="black", linestyle="-"),
    ),
    "manual, 10 min": (
        os.path.join(REVISION_DIR, "*_{material}_grind_manual10min_meas*.csv"),
        dict(color="#E74C3C", linestyle=":"),
    ),
}
# Order used in the figure legend and in the output tables.
CONDITION_ORDER = ["starting powder", "robot, 25 min", "robot, 60 min", "manual, 10 min"]

# The 25 min condition is three independent runs; the revision measurements are
# three replicate measurements of one prepared powder.  Recorded so that the
# standard deviations in the tables are not over-interpreted.
REPLICATE_KIND = {
    "starting powder": "replicate measurements",
    "robot, 25 min": "independent runs",
    "robot, 60 min": "replicate measurements",
    "manual, 10 min": "replicate measurements",
}

FINES_THRESHOLDS_UM = (1, 5, 10, 50, 100, 500)
OUTPUT_DIR = os.path.join("analysis_results", "run_07_compare_exhaustive_and_manual_grinding")
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 9.0


def files_for(condition, material):
    pattern, _ = CONDITIONS[condition]
    files = sorted(glob.glob(pattern.format(material=material)))
    if not files:
        raise FileNotFoundError(f"No files for {material} / {condition}: {pattern}")
    return files


def cumulative_below(sizes, volumes, thresholds=FINES_THRESHOLDS_UM):
    edges = class_edges(sizes)
    cumulative = np.concatenate([[0.0], np.cumsum(volumes)])
    cumulative = 100.0 * cumulative / cumulative[-1]
    return {
        f"below_{t}um_percent": float(np.interp(np.log(t), np.log(edges), cumulative))
        for t in thresholds
    }


def collect():
    rows, curves = [], {}
    for material in MATERIALS:
        for condition in CONDITION_ORDER:
            files = files_for(condition, material)
            material_curves = []
            for index, path in enumerate(files, start=1):
                metadata, sizes, volumes = read_measurement(path)
                metrics = distribution_metrics(sizes, volumes)
                check_against_header(metrics, metadata, path)
                row = {
                    "material": MATERIAL_LABELS[material],
                    "condition": condition,
                    "replicate_kind": REPLICATE_KIND[condition],
                    "replicate": index,
                    "file": path,
                    "D10_um": round(metrics["D10"], 1),
                    "D50_um": round(metrics["D50"], 1),
                    "D90_um": round(metrics["D90"], 1),
                    "span": round(metrics["span"], 2),
                    "mode_um": round(metrics["mode_um"], 1),
                    "D43_um": round(metrics["D43_um"], 1),
                }
                row.update(
                    {k: round(v, 1) for k, v in cumulative_below(sizes, volumes).items()}
                )
                rows.append(row)
                material_curves.append((sizes, volumes))
            curves[(material, condition)] = material_curves
    return pd.DataFrame(rows), curves


def summarize(frame):
    value_columns = [
        c
        for c in frame.columns
        if c.endswith(("_um", "percent")) or c == "span"
    ]
    grouped = frame.groupby(["material", "condition"], sort=False)
    summary = grouped[value_columns].agg(["mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary["n"] = grouped.size()
    summary["replicate_kind"] = grouped["replicate_kind"].first()
    return summary.round(2).reset_index()


def mean_curve(curve_list):
    reference = curve_list[0][0]
    for sizes, _ in curve_list[1:]:
        if sizes.shape != reference.shape or not np.allclose(sizes, reference):
            raise ValueError("Measurements do not share a size grid")
    return reference, np.mean([volumes for _, volumes in curve_list], axis=0)


def build_figure(curves, summary):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True)
    panel_labels = ["(a)", "(b)", "(c)"]

    for axis, material, panel in zip(axes, MATERIALS, panel_labels):
        for condition in CONDITION_ORDER:
            _, style = CONDITIONS[condition]
            sizes, volumes = mean_curve(curves[(material, condition)])
            axis.plot(sizes, volumes, label=condition, **style)

        median = {
            condition: summary.loc[
                (summary["material"] == MATERIAL_LABELS[material])
                & (summary["condition"] == condition),
                "D50_um_mean",
            ].iloc[0]
            for condition in ("robot, 25 min", "robot, 60 min", "manual, 10 min")
        }
        axis.set_title(
            f"{panel} {MATERIAL_LABELS[material]}: "
            r"$D_{50}$ = "
            f"{median['robot, 25 min']:.0f} / {median['robot, 60 min']:.0f} / "
            f"{median['manual, 10 min']:.0f} " + r"$\mathrm{\mu m}$ "
            "(robot 25 min / robot 60 min / manual 10 min)",
            loc="left",
            fontsize=11,
        )
        axis.set_ylabel("Volume density (%)")
        axis.set_xscale("log")
        axis.set_xlim(0.1, 2000.0)
        axis.grid(alpha=0.2, linewidth=0.4)

    axes[-1].set_xlabel(r"Particle diameter ($\mathrm{\mu m}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    return fig


def report_fines(summary):
    """Print the agglomeration diagnostic for the 25 min vs 60 min comparison.

    Read the coarse end and the fine end against each other: an unchanged
    ``below_500um_percent`` with a collapsing ``below_1um_percent`` is
    agglomeration, not under-grinding.
    """
    print("\nFine-end diagnostic (mean volume percentage below each size)")
    columns = [f"below_{t}um_percent_mean" for t in FINES_THRESHOLDS_UM]
    view = summary[summary["condition"].isin(["robot, 25 min", "robot, 60 min"])]
    print(
        view.set_index(["material", "condition"])[["D10_um_mean", "D50_um_mean", "D90_um_mean"] + columns]
        .to_string()
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_plot_style()

    frame, curves = collect()
    summary = summarize(frame)

    fig = build_figure(curves, summary)
    base = os.path.join(OUTPUT_DIR, "exhaustive_grinding_psd")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    frame.to_csv(os.path.join(OUTPUT_DIR, "exhaustive_grinding_per_measurement.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "exhaustive_grinding_summary.csv"), index=False)
    frame[
        ["material", "condition", "replicate", "D10_um", "D50_um", "D90_um"]
        + [f"below_{t}um_percent" for t in FINES_THRESHOLDS_UM]
    ].to_csv(os.path.join(OUTPUT_DIR, "exhaustive_grinding_fines.csv"), index=False)

    print(f"Saved figure: {base}.pdf")
    print(
        summary.set_index(["material", "condition"])[
            ["n", "replicate_kind", "D10_um_mean", "D50_um_mean", "D90_um_mean", "span_mean", "mode_um_mean"]
        ].to_string()
    )
    report_fines(summary)


if __name__ == "__main__":
    main()
