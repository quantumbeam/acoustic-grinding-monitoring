"""Referee 1 (scientific comment 1): initial condition of the starting powders.

The reviewer asked us to introduce the initial conditions of the samples,
including the initial particle size distribution.  The grinding trials always
start from a sieved size fraction (Section 2.3), so the initial condition is the
sieved starting powder, which was measured here for the three materials with
three replicate measurements each
(``data/powder_size_distribution/additional_experiments/initial_psd``).  Note that the
three files of a material are repeat measurements of the same prepared powder,
not three independent grinding runs; see that directory's README.

Conventions follow the rest of the repository, with one point that matters for
the numbers:

* the ``SizeClasses`` column of the MasterSizer export holds the LOWER edge of
  each class and ``VolumeDensity`` the volume percentage inside it, so the
  cumulative curve is evaluated at the class edges and the percentiles are
  obtained by interpolation in log-diameter.  This reproduces the instrument's
  own ``Dx (50)`` header entry to better than 0.01 %, which the script asserts
  for every file; interpolating on the class values instead (as if they were
  representative diameters) biases D50 low by roughly 10 %.
* the earliest ground state measured in the paper (the 3 min condition of exp2)
  is recomputed with the same estimator so that the comparison in the ESI table
  is internally consistent.

Outputs (analysis_results/run_06_plot_initial_particle_size/):
    initial_psd_starting_powders.pdf/.png   three-panel distribution figure
    initial_psd_per_measurement.csv         percentiles of every measurement
    initial_psd_summary.csv                 per-material mean +/- SD (n = 3)
    initial_psd_vs_3min.csv                 initial state vs the 3 min condition
    initial_psd_instrument_settings.csv     dispersion / optical settings used
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MATERIALS = ["NaCl", "Citricacid", "MSG"]
MATERIAL_LABELS = {"NaCl": "NaCl", "Citricacid": "Citric acid", "MSG": "MSG"}
REPLICATE_STYLE = [
    dict(color="black", linestyle="-"),
    dict(color="#E74C3C", linestyle="--"),
    dict(color="#2E6DB4", linestyle=":"),
]
REFERENCE_STYLE = dict(color="#7F7F7F", linestyle="-", linewidth=1.0, alpha=0.9)

# Nominal sieve fraction used to prepare the starting powder (Section 2.3).
SIEVE_FRACTION_UM = {
    "NaCl": (250.0, 500.0),
    "Citricacid": (500.0, None),
    "MSG": (500.0, None),
}
REFERENCE_GRIND_MIN = 3

INITIAL_DIR = os.path.join(
    "data", "powder_size_distribution", "additional_experiments", "initial_psd"
)
OUTPUT_DIR = os.path.join("analysis_results", "run_06_plot_initial_particle_size")
PERCENTILES = (10, 50, 90)
D50_TOLERANCE_PERCENT = 0.01

FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 9.0


def configure_plot_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.titlesize": 16,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "lines.linewidth": 1.4,
            "legend.frameon": False,
        }
    )


def read_measurement(file_path):
    """Return (metadata, lower class edges, volume percentage per class)."""
    metadata = {}
    sizes, volumes = [], []
    in_table = False
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not in_table:
                if line.strip().startswith("SizeClasses"):
                    in_table = True
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0]:
                    metadata[parts[0]] = parts[1]
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                size, volume = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if np.isfinite(size) and np.isfinite(volume):
                sizes.append(size)
                volumes.append(volume)
    return metadata, np.array(sizes, dtype=float), np.array(volumes, dtype=float)


def class_edges(sizes):
    """Class boundaries. The classes are geometric, so the last upper edge is
    obtained from the constant size ratio of the series."""
    ratio = float(np.exp(np.diff(np.log(sizes)).mean()))
    return np.concatenate([sizes, [sizes[-1] * ratio]])


def percentiles(sizes, volumes, fractions=PERCENTILES):
    edges = class_edges(sizes)
    cumulative = np.concatenate([[0.0], np.cumsum(volumes)])
    cumulative = 100.0 * cumulative / cumulative[-1]
    return {
        f"D{int(f)}": float(np.exp(np.interp(f, cumulative, np.log(edges))))
        for f in fractions
    }


def distribution_metrics(sizes, volumes):
    edges = class_edges(sizes)
    centres = np.sqrt(edges[:-1] * edges[1:])
    metrics = percentiles(sizes, volumes)
    weights = volumes / volumes.sum()
    metrics["mode_um"] = float(centres[int(np.argmax(volumes))])
    metrics["D43_um"] = float(np.sum(weights * centres))
    metrics["span"] = (metrics["D90"] - metrics["D10"]) / metrics["D50"]
    return metrics


def check_against_header(metrics, metadata, file_path):
    """The instrument reports Dx (50); use it to validate the estimator."""
    reported = metadata.get("Dx (50)")
    if reported is None:
        return None
    reported = float(reported)
    deviation = 100.0 * abs(metrics["D50"] - reported) / reported
    if deviation > D50_TOLERANCE_PERCENT:
        raise ValueError(
            f"D50 estimator disagrees with the instrument header for {file_path}: "
            f"{metrics['D50']:.3f} um vs {reported:.3f} um ({deviation:.3f} %)"
        )
    return reported


def initial_files(material):
    files = sorted(glob.glob(os.path.join(INITIAL_DIR, f"*_{material}_initial_meas*.csv")))
    if not files:
        raise FileNotFoundError(f"No initial-distribution file for {material} in {INITIAL_DIR}")
    return files


def reference_files(material):
    pattern = os.path.join(
        "data",
        "powder_size_distribution",
        "exp2",
        material,
        "*",
        f"*grind{REFERENCE_GRIND_MIN}min*.csv",
    )
    return sorted(glob.glob(pattern))


def collect(material, files, condition):
    rows, curves = [], []
    for index, file_path in enumerate(files, start=1):
        metadata, sizes, volumes = read_measurement(file_path)
        if sizes.size == 0:
            raise ValueError(f"Empty distribution table: {file_path}")
        metrics = distribution_metrics(sizes, volumes)
        reported = check_against_header(metrics, metadata, file_path)
        rows.append(
            {
                "material": MATERIAL_LABELS[material],
                "condition": condition,
                "measurement": index,
                "file": file_path,
                "D10_um": round(metrics["D10"], 1),
                "D50_um": round(metrics["D50"], 1),
                "D90_um": round(metrics["D90"], 1),
                "span": round(metrics["span"], 2),
                "mode_um": round(metrics["mode_um"], 1),
                "D43_um": round(metrics["D43_um"], 1),
                "D50_instrument_header_um": None if reported is None else round(reported, 1),
                "weighted_residual_percent": float(metadata.get("Weighted Residual", "nan")),
            }
        )
        curves.append((sizes, volumes))
    return rows, curves


def summarize(rows, material, condition):
    subset = [r for r in rows if r["material"] == MATERIAL_LABELS[material] and r["condition"] == condition]
    summary = {"material": MATERIAL_LABELS[material], "condition": condition, "n": len(subset)}
    for key in ("D10_um", "D50_um", "D90_um", "span", "mode_um", "D43_um"):
        values = np.array([r[key] for r in subset], dtype=float)
        mean, sd = values.mean(), values.std(ddof=1)
        decimals = 2 if key == "span" else 1
        summary[f"{key}_mean"] = round(float(mean), decimals)
        summary[f"{key}_sd"] = round(float(sd), decimals)
        summary[f"{key}_cv_percent"] = round(float(100.0 * sd / mean), 1)
    return summary


def mean_curve(curves):
    """Mean volume density of replicate measurements on their common grid."""
    reference = curves[0][0]
    for sizes, _ in curves[1:]:
        if sizes.shape != reference.shape or not np.allclose(sizes, reference):
            raise ValueError("Replicate measurements do not share a size grid")
    return reference, np.mean([volumes for _, volumes in curves], axis=0)


def build_figure(initial_curves, reference_curves, initial_summary):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True)
    panel_labels = ["(a)", "(b)", "(c)"]

    for axis, material, panel in zip(axes, MATERIALS, panel_labels):
        lower, upper = SIEVE_FRACTION_UM[material]
        axis.axvspan(
            lower,
            upper if upper is not None else 1500.0,
            color="#F0C808",
            alpha=0.18,
            linewidth=0,
            label="Nominal sieve fraction",
        )

        sizes, volumes = mean_curve(reference_curves[material])
        axis.plot(sizes, volumes, label=f"After {REFERENCE_GRIND_MIN} min (mean, n = 3)", **REFERENCE_STYLE)

        for index, (sizes, volumes) in enumerate(initial_curves[material]):
            axis.plot(
                sizes,
                volumes,
                label=f"Starting powder {index + 1}",
                **REPLICATE_STYLE[index],
            )

        row = initial_summary[material]
        axis.set_title(
            f"{panel} {MATERIAL_LABELS[material]}: "
            f"$D_{{50}}$ = {row['D50_um_mean']:.0f} $\\pm$ {row['D50_um_sd']:.0f} "
            r"$\mathrm{\mu m}$",
            loc="left",
        )
        axis.set_ylabel("Volume density (%)")
        axis.set_xscale("log")
        axis.set_xlim(1.0, 1500.0)
        axis.grid(alpha=0.2, linewidth=0.4)

    axes[-1].set_xlabel(r"Particle diameter ($\mathrm{\mu m}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    return fig


def instrument_settings(files_by_condition):
    keys = [
        "Particle Refractive Index",
        "Particle Absorption Index",
        "Particle Density",
        "Are particles non-spherical?",
        "Analysis Model",
        "Low Size",
        "High Size",
        "Air Pressure Demand",
        "Feed Rate Demand",
    ]
    rows = []
    for material, conditions in files_by_condition.items():
        for condition, files in conditions.items():
            metadata, _, _ = read_measurement(files[0])
            row = {"material": MATERIAL_LABELS[material], "condition": condition}
            row.update({key: metadata.get(key, "") for key in keys})
            rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_plot_style()

    rows = []
    initial_curves, reference_curves = {}, {}
    files_by_condition = {}

    for material in MATERIALS:
        files = initial_files(material)
        material_rows, curves = collect(material, files, "starting powder")
        rows.extend(material_rows)
        initial_curves[material] = curves

        ref_files = reference_files(material)
        ref_rows, ref_curves = collect(material, ref_files, f"{REFERENCE_GRIND_MIN} min")
        rows.extend(ref_rows)
        reference_curves[material] = ref_curves

        files_by_condition[material] = {"starting powder": files, f"{REFERENCE_GRIND_MIN} min": ref_files}

    summary_rows = []
    initial_summary = {}
    for material in MATERIALS:
        starting = summarize(rows, material, "starting powder")
        ground = summarize(rows, material, f"{REFERENCE_GRIND_MIN} min")
        initial_summary[material] = starting
        summary_rows.extend([starting, ground])

    comparison_rows = []
    for material in MATERIALS:
        starting = initial_summary[material]
        ground = next(
            r for r in summary_rows
            if r["material"] == MATERIAL_LABELS[material] and r["condition"] == f"{REFERENCE_GRIND_MIN} min"
        )
        comparison_rows.append(
            {
                "material": MATERIAL_LABELS[material],
                "D50_starting_um": starting["D50_um_mean"],
                "D50_starting_sd_um": starting["D50_um_sd"],
                "D50_starting_cv_percent": starting["D50_um_cv_percent"],
                f"D50_{REFERENCE_GRIND_MIN}min_um": ground["D50_um_mean"],
                f"D50_{REFERENCE_GRIND_MIN}min_sd_um": ground["D50_um_sd"],
                "size_reduction_factor": round(starting["D50_um_mean"] / ground["D50_um_mean"], 2),
                "span_starting": starting["span_mean"],
                f"span_{REFERENCE_GRIND_MIN}min": ground["span_mean"],
            }
        )

    fig = build_figure(initial_curves, reference_curves, initial_summary)
    base = os.path.join(OUTPUT_DIR, "initial_psd_starting_powders")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "initial_psd_per_measurement.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "initial_psd_summary.csv"), index=False)
    pd.DataFrame(comparison_rows).to_csv(os.path.join(OUTPUT_DIR, "initial_psd_vs_3min.csv"), index=False)
    pd.DataFrame(instrument_settings(files_by_condition)).to_csv(
        os.path.join(OUTPUT_DIR, "initial_psd_instrument_settings.csv"), index=False
    )

    print(f"Saved figure: {base}.pdf")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()
    print(pd.DataFrame(comparison_rows).to_string(index=False))


if __name__ == "__main__":
    main()
