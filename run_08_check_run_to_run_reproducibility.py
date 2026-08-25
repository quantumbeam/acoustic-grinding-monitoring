"""Referee 1 (scientific comment 2): run-to-run reproducibility of the AE decay.

The reviewer asks whether samples with similar initial size distributions give
similar decay curves.  Every grinding condition in this study was repeated three
times with fresh 3.000 g charges from the same lot, i.e. with nominally
identical initial size distributions, so the three replicates of the 25 min runs
answer the within-material part of the question directly.

This script overlays the three replicate S_AE trajectories for each material and
quantifies their spread.  Conventions follow the rest of the repository:

* the AE feature is the total spectral power over 100 kHz-1 MHz, in mV^2;
* the smoothed feature is the 4-point moving average used in Eq. (2);
* the steady-state level is the final raw acquisition of a run, matching the
  values quoted in Section 4.2 (27.4 / 10.2 / 260.4 mV^2);
* the x axis is the measured elapsed process time (Referee 1, technical
  comment 3), taken from the acquisition timestamps.

Outputs (analysis_results/run_08_check_run_to_run_reproducibility/):
    reproducibility_exp2_25min.pdf/.png          three-panel overlay figure
    reproducibility_exp2_25min_logy.pdf/.png     same with logarithmic y axis
    reproducibility_exp2_25min_<material>.pdf    single-material versions
    reproducibility_exp2_25min_summary.csv       per-material spread statistics
    reproducibility_exp2_25min_series.csv        the plotted trajectories
"""

import glob
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ae_fft import calculate_fft_power

MATERIALS = ["NaCl", "Citricacid", "MSG"]
MATERIAL_LABELS = {"NaCl": "NaCl", "Citricacid": "Citric acid", "MSG": "MSG"}
TRIALS = ["1st", "2nd", "3rd"]
TRIAL_LABELS = {"1st": "Run 1", "2nd": "Run 2", "3rd": "Run 3"}
TRIAL_STYLE = {
    "1st": dict(color="black", marker="o", linestyle="-"),
    "2nd": dict(color="#E74C3C", marker="x", linestyle="--"),
    "3rd": dict(color="#2E6DB4", marker="^", linestyle=":"),
}
GRIND_MIN = 25
MOVING_AVERAGE_WINDOW = 4
# One AE measurement per block of 24 grinding cycles at 0.5 s per cycle.
NET_GRINDING_S_PER_ACQUISITION = 24 * 0.5
OUTPUT_DIR = os.path.join("analysis_results", "run_08_check_run_to_run_reproducibility")
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ae_power_cache.json")


def norm_path(path):
    return os.path.normpath(os.path.abspath(path))


def parse_timestamp(path):
    match = re.search(r"(\d{8}_\d{6})", os.path.basename(path))
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") if match else datetime.min


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def moving_average(values, window=MOVING_AVERAGE_WINDOW):
    if len(values) < window:
        return np.array([])
    return np.convolve(values, np.ones(window) / window, mode="valid")


def elapsed_minutes(file_paths):
    """Elapsed process time (min) of each acquisition, from the file timestamps."""
    stamps = [parse_timestamp(path) for path in file_paths]
    offsets = np.array([(t - stamps[0]).total_seconds() for t in stamps])
    return (offsets + NET_GRINDING_S_PER_ACQUISITION) / 60.0


def load_run(cache, material, trial):
    files = sorted(
        glob.glob(f"data/ae/exp2/{material}/{trial}/*grind{GRIND_MIN}min*.csv"),
        key=lambda p: (parse_timestamp(p), os.path.basename(p)),
    )
    values, kept = [], []
    for path in files:
        power = cache.get(norm_path(path))
        if power is None:
            power = calculate_fft_power(path)
        if power is None:
            continue
        values.append(power * 1e6)
        kept.append(path)
    return np.array(values), elapsed_minutes(kept)


def configure_plot_style():
    try:
        import scienceplots  # noqa: F401

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
            "lines.markersize": 5,
            "legend.frameon": False,
        }
    )


def draw_panel(ax, runs, material, yscale):
    for trial, (values, times) in runs.items():
        smoothed = moving_average(values)
        ax.plot(
            times,
            values,
            alpha=0.35,
            **{k: v for k, v in TRIAL_STYLE[trial].items() if k != "marker"},
        )
        ax.plot(
            times[MOVING_AVERAGE_WINDOW - 1 :],
            smoothed,
            label=TRIAL_LABELS[trial],
            **TRIAL_STYLE[trial],
        )
    ax.set_xlabel("Grinding time (min)")
    ax.set_ylabel(r"Total spectral power (a.u.)")
    ax.set_yscale(yscale)
    ax.set_xlim(0.0, max(t[-1] for _, t in runs.values()) * 1.02)
    ax.set_title(MATERIAL_LABELS[material])
    ax.legend(loc="upper right")


def summarize(material, runs):
    """Spread of the three replicate trajectories, aligned by acquisition index."""
    values = [v for v, _ in runs.values()]
    n = min(len(v) for v in values)
    matrix = np.array([v[:n] for v in values])
    smoothed = np.array([moving_average(v[:n]) for v in values])

    # Pointwise coefficient of variation across the three runs.
    cv_raw = 100.0 * matrix.std(axis=0, ddof=1) / matrix.mean(axis=0)
    cv_smooth = 100.0 * smoothed.std(axis=0, ddof=1) / smoothed.mean(axis=0)

    initial = matrix[:, 0]
    steady = matrix[:, -1]                     # final raw acquisition (Section 4.2)
    plateau = matrix[:, -MOVING_AVERAGE_WINDOW:].mean(axis=1)

    def stats(x):
        return x.mean(), x.std(ddof=1), 100.0 * x.std(ddof=1) / x.mean()

    initial_mean, initial_sd, initial_cv = stats(initial)
    steady_mean, steady_sd, steady_cv = stats(steady)
    plateau_mean, plateau_sd, plateau_cv = stats(plateau)

    return {
        "material": MATERIAL_LABELS[material],
        "n_runs": len(values),
        "n_acquisitions": n,
        "initial_S_AE_mean_mV2": round(initial_mean, 1),
        "initial_S_AE_sd_mV2": round(initial_sd, 1),
        "initial_S_AE_cv_percent": round(initial_cv, 1),
        "steady_state_S_AE_mean_mV2": round(steady_mean, 1),
        "steady_state_S_AE_sd_mV2": round(steady_sd, 1),
        "steady_state_S_AE_cv_percent": round(steady_cv, 1),
        "plateau_last4_mean_mV2": round(plateau_mean, 1),
        "plateau_last4_sd_mV2": round(plateau_sd, 1),
        "plateau_last4_cv_percent": round(plateau_cv, 1),
        "pointwise_cv_raw_mean_percent": round(float(cv_raw.mean()), 1),
        "pointwise_cv_raw_max_percent": round(float(cv_raw.max()), 1),
        "pointwise_cv_movavg_mean_percent": round(float(cv_smooth.mean()), 1),
        "pointwise_cv_movavg_max_percent": round(float(cv_smooth.max()), 1),
        "initial_to_steady_ratio": round(initial_mean / steady_mean, 1),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_plot_style()
    cache = load_cache()

    data, rows, series_rows = {}, [], []
    for material in MATERIALS:
        runs = {}
        for trial in TRIALS:
            values, times = load_run(cache, material, trial)
            if values.size:
                runs[trial] = (values, times)
                smoothed = moving_average(values)
                for i, (t, v) in enumerate(zip(times, values), start=1):
                    rec = {
                        "material": MATERIAL_LABELS[material],
                        "run": TRIAL_LABELS[trial],
                        "acquisition": i,
                        "grinding_time_min": round(float(t), 3),
                        "S_AE_mV2": round(float(v), 3),
                    }
                    j = i - MOVING_AVERAGE_WINDOW
                    rec["S_AE_movavg_mV2"] = (
                        round(float(smoothed[j]), 3) if 0 <= j < len(smoothed) else ""
                    )
                    series_rows.append(rec)
        if len(runs) < 2:
            print(f"Skip {material}: fewer than two replicates with data.")
            continue
        data[material] = runs
        rows.append(summarize(material, runs))

    for yscale, suffix in (("linear", ""), ("log", "_logy")):
        fig, axes = plt.subplots(1, len(data), figsize=(6.0 * len(data), 4.6))
        axes = np.atleast_1d(axes)
        for ax, (material, runs) in zip(axes, data.items()):
            draw_panel(ax, runs, material, yscale)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(OUTPUT_DIR, f"reproducibility_exp2_{GRIND_MIN}min{suffix}.{ext}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close(fig)

    for material, runs in data.items():
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        draw_panel(ax, runs, material, "linear")
        fig.tight_layout()
        path = os.path.join(
            OUTPUT_DIR, f"reproducibility_exp2_{GRIND_MIN}min_{material}.pdf"
        )
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    summary_path = os.path.join(OUTPUT_DIR, f"reproducibility_exp2_{GRIND_MIN}min_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    series_path = os.path.join(OUTPUT_DIR, f"reproducibility_exp2_{GRIND_MIN}min_series.csv")
    pd.DataFrame(series_rows).to_csv(series_path, index=False)
    print(f"Saved series: {series_path}")

    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
