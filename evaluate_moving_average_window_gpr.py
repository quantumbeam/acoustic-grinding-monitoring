import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import LeaveOneGroupOut

from fft_processing import calculate_fft_power


@dataclass
class Sample:
    d50: float
    grind_min: float
    trial: str
    reagent: str
    ae_series_mV2: np.ndarray


def norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def get_d50(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Dx (50)"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        return float(parts[1])
    except (OSError, ValueError):
        return None
    return None


def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def load_cache(cache_file):
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def update_cache_policy_a(cache, cache_file, file_paths):
    """Policy A: cache hit -> no recompute; compute only missing entries."""
    updated = 0
    skipped = 0
    for file_path in file_paths:
        key = norm_path(file_path)
        if key in cache:
            skipped += 1
            continue
        power = calculate_fft_power(file_path)
        if power is None:
            continue
        cache[key] = float(power)
        updated += 1
    if updated:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)
        print(f"Cache updated: {updated} new values (skipped {skipped}).")
        print(f"Cache saved to: {cache_file}")
    else:
        print(f"Cache hit for all required files (skipped {skipped}).")
    return cache


def build_samples(experiment, reagent, trial):
    ae_base_path = os.path.join("ae_data", experiment)
    psd_base_path = os.path.join("powder_size_distribution_data", experiment)

    reagent_pattern = "*" if reagent == "all" else reagent
    trial_pattern = "*" if trial == "all" else trial

    all_psd_files = []
    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, reagent_pattern)):
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, trial_pattern)):
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, "*.csv")))

    required_ae_files = set()
    psd_info = []
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue
        reagent_name = path_parts[-3]
        trial_name = path_parts[-2]

        d50 = get_d50(psd_file)
        if d50 is None:
            continue

        match = re.search(r"(grind\d+min)", os.path.basename(psd_file))
        if not match:
            continue
        grind_key = match.group(1)
        grind_match = re.search(r"grind(\d+)min", grind_key)
        if not grind_match:
            continue
        grind_min = float(grind_match.group(1))
        ae_session_path = os.path.join(ae_base_path, reagent_name, trial_name)
        ae_files = sorted(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))
        if not ae_files:
            continue
        required_ae_files.update(ae_files)
        psd_info.append((psd_file, reagent_name, trial_name, d50, grind_min, ae_files))

    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ae_power_cache.json")
    cache = load_cache(cache_file)
    cache = update_cache_policy_a(cache, cache_file, list(required_ae_files))

    samples = []
    for _, reagent_name, trial_name, d50, grind_min, ae_files in psd_info:
        ae_power_timeseries = [cache.get(norm_path(f)) for f in ae_files]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]
        if len(ae_power_timeseries) < 2:
            continue
        ae_power_mV2 = np.array(ae_power_timeseries, dtype=float) * 1e6
        samples.append(
            Sample(
                d50=float(d50),
                grind_min=grind_min,
                trial=trial_name,
                reagent=reagent_name,
                ae_series_mV2=ae_power_mV2,
            )
        )

    return samples


def build_dataset(samples, window_size):
    rows = []
    for s in samples:
        smoothed = moving_average(s.ae_series_mV2, window_size=window_size)
        if smoothed.size == 0:
            continue
        final_ae = float(smoothed[-1])
        rows.append((s.d50, final_ae, s.grind_min, s.trial, s.reagent))
    return np.array(rows, dtype=object)


def fit_gpr(x_train, y_train):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        n_restarts_optimizer=10,
        normalize_y=True,
    )
    gpr.fit(x_train, y_train)
    return gpr


def compute_metrics(y_true, y_pred_mean, y_pred_std):
    y_true = np.asarray(y_true, dtype=float)
    y_pred_mean = np.asarray(y_pred_mean, dtype=float)
    y_pred_std = np.asarray(y_pred_std, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred_mean) ** 2)))
    var = np.maximum(y_pred_std ** 2, 1e-9)
    nlpd = float(np.mean(0.5 * np.log(2.0 * np.pi * var) + 0.5 * (y_true - y_pred_mean) ** 2 / var))
    return rmse, nlpd


def configure_plot_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "lines.linewidth": 1.2,
            "lines.markersize": 6,
            "legend.frameon": False,
        }
    )


def select_representative_samples(samples, reagent, trial_pref, grind_pref):
    candidates = [s for s in samples if s.reagent == reagent]
    if not candidates:
        return []
    if trial_pref == "all":
        trials = sorted({s.trial for s in candidates})
        selected = []
        for t in trials:
            matches = [
                s for s in candidates
                if s.trial == t and int(s.grind_min) == int(grind_pref)
            ]
            if matches:
                selected.append(matches[0])
                continue
            fallback = [s for s in candidates if s.trial == t]
            if fallback:
                selected.append(fallback[0])
        return selected

    preferred = [
        s for s in candidates
        if s.trial == trial_pref and int(s.grind_min) == int(grind_pref)
    ]
    if preferred:
        return [preferred[0]]
    preferred_trial = [s for s in candidates if s.trial == trial_pref]
    if preferred_trial:
        return [preferred_trial[0]]
    return [candidates[0]]


def plot_recommended_timeseries(sample, window_size, out_dir, base_name):
    if sample is None:
        return None, None
    series = sample.ae_series_mV2
    if len(series) == 0:
        return None, None
    smoothed = moving_average(series, window_size=window_size)
    x_vals = np.arange(1, len(series) + 1)
    smooth_x = np.arange(window_size, len(series) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, series, "o-", color="black", label="Original Data")
    if len(smoothed):
        plt.plot(
            smooth_x,
            smoothed,
            "x--",
            color="red",
            label=f"{window_size}-point Moving Average",
        )
    plt.xlabel("Number of motions")
    plt.ylabel(r"Total spectral power($\mathrm{mV}^2$)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{base_name}.png")
    out_pdf = os.path.join(out_dir, f"{base_name}.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    return out_png, out_pdf


def evaluate_cv(data_array, cv_method):
    d50_vals = np.array(data_array[:, 0], dtype=float)
    ae_vals = np.array(data_array[:, 1], dtype=float)
    trial_labels = np.array(data_array[:, 3], dtype=str)
    reagent_labels = np.array(data_array[:, 4], dtype=str)
    grind_vals = np.array(data_array[:, 2], dtype=float)

    if cv_method == "loocv":
        y_true_all = []
        y_mean_all = []
        y_std_all = []
        for i in range(len(d50_vals)):
            mask = np.ones(len(d50_vals), dtype=bool)
            mask[i] = False
            x_train = d50_vals[mask].reshape(-1, 1)
            y_train = ae_vals[mask]
            x_test = d50_vals[i].reshape(1, -1)
            y_test = ae_vals[i]
            gpr = fit_gpr(x_train, y_train)
            y_mean, y_std = gpr.predict(x_test, return_std=True)
            y_true_all.append(y_test)
            y_mean_all.append(y_mean[0])
            y_std_all.append(y_std[0])
        return compute_metrics(y_true_all, y_mean_all, y_std_all)

    groups = [f"{r}|{g}" for r, g in zip(reagent_labels, grind_vals)]
    logo = LeaveOneGroupOut()
    y_true_all = []
    y_mean_all = []
    y_std_all = []
    for train_idx, test_idx in logo.split(d50_vals, ae_vals, groups=groups):
        x_train = d50_vals[train_idx].reshape(-1, 1)
        y_train = ae_vals[train_idx]
        x_test = d50_vals[test_idx].reshape(-1, 1)
        y_test = ae_vals[test_idx]
        gpr = fit_gpr(x_train, y_train)
        y_mean, y_std = gpr.predict(x_test, return_std=True)
        y_true_all.extend(y_test.tolist())
        y_mean_all.extend(y_mean.tolist())
        y_std_all.extend(y_std.tolist())
    return compute_metrics(y_true_all, y_mean_all, y_std_all)


def select_window(results, rmse_tol):
    min_nlpd = min(row["nlpd"] for row in results)
    min_rmse = min(row["rmse"] for row in results)
    candidates = [
        row for row in results
        if np.isclose(row["nlpd"], min_nlpd) or row["nlpd"] == min_nlpd
    ]
    filtered = [row for row in candidates if row["rmse"] <= min_rmse * (1.0 + rmse_tol)]
    if filtered:
        return min(filtered, key=lambda r: r["window_size"])
    return min(candidates, key=lambda r: r["window_size"])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate moving-average window size via GPR CV (RMSE/NLPD)."
    )
    parser.add_argument("--experiment", type=str, default="exp2")
    parser.add_argument("--reagent", type=str, default="all")
    parser.add_argument("--trial", type=str, default="all")
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7],
    )
    parser.add_argument("--cv", type=str, default="group", choices=["group", "loocv", "both"])
    parser.add_argument(
        "--include-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate a combined (all reagents) model in addition to per-reagent.",
    )
    parser.add_argument("--rmse-tol", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default=os.path.join("results", "moving_average"))
    parser.add_argument(
        "--plot-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot RMSE/NLPD vs window size.",
    )
    parser.add_argument(
        "--plot-timeseries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot time series with recommended moving average.",
    )
    parser.add_argument("--timeseries-trial", type=str, default="all")
    parser.add_argument("--timeseries-grind-min", type=float, default=25)
    args = parser.parse_args()

    samples = build_samples(args.experiment, args.reagent, args.trial)
    if not samples:
        print("No samples found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    reagent_names = sorted({s.reagent for s in samples})
    if args.include_all:
        reagent_names.append("all")

    summary_rows = []
    recommended_rows = []

    if args.plot_metrics or args.plot_timeseries:
        configure_plot_style()

    cv_methods = ["group", "loocv"] if args.cv == "both" else [args.cv]

    for cv_method in cv_methods:
        for reagent in reagent_names:
            if reagent == "all":
                subset = samples
            else:
                subset = [s for s in samples if s.reagent == reagent]

            if len(subset) < 3:
                print(f"Skip {reagent}: not enough samples.")
                continue

            results = []
            for w in args.window_sizes:
                data_array = build_dataset(subset, window_size=w)
                if data_array.size == 0 or len(data_array) < 3:
                    continue
                rmse, nlpd = evaluate_cv(data_array, cv_method)
                results.append(
                    {
                        "reagent": reagent,
                        "window_size": w,
                        "rmse": rmse,
                        "nlpd": nlpd,
                        "n_samples": len(data_array),
                        "cv_method": cv_method,
                    }
                )

            if not results:
                print(f"No results for {reagent} ({cv_method}).")
                continue

            best = select_window(results, args.rmse_tol)
            best_row = best.copy()
            best_row["rmse_tol"] = args.rmse_tol
            recommended_rows.append(best_row)
            summary_rows.extend(results)

            print(
                f"{reagent} [{cv_method}]: 推奨N={best['window_size']} "
                f"(RMSE={best['rmse']:.3f}, NLPD={best['nlpd']:.3f})"
            )

            if args.plot_metrics and results:
                windows = [row["window_size"] for row in results]
                rmse_vals = [row["rmse"] for row in results]
                nlpd_vals = [row["nlpd"] for row in results]

                fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
                axes[0].plot(windows, rmse_vals, "o-", color="black")
                axes[0].set_ylabel("RMSE")
                axes[0].axvline(best["window_size"], color="red", linestyle="--", alpha=0.7)

                axes[1].plot(windows, nlpd_vals, "o-", color="black")
                axes[1].set_ylabel("NLPD")
                axes[1].set_xlabel("Moving average window size")
                axes[1].axvline(best["window_size"], color="red", linestyle="--", alpha=0.7)

                fig.tight_layout()
                base = f"moving_average_gpr_cv_{args.experiment}_{cv_method}_{reagent}"
                out_png = os.path.join(args.out_dir, f"{base}.png")
                out_pdf = os.path.join(args.out_dir, f"{base}.pdf")
                fig.savefig(out_png, dpi=300)
                fig.savefig(out_pdf)
                plt.close(fig)
                print(f"Saved CV plot: {out_png}")
                print(f"Saved CV plot: {out_pdf}")

            if args.plot_timeseries and reagent != "all":
                selected_samples = select_representative_samples(
                    samples,
                    reagent,
                    args.timeseries_trial,
                    args.timeseries_grind_min,
                )
                for sample in selected_samples:
                    base = (
                        f"moving_average_recommended_{args.experiment}_{cv_method}_"
                        f"{reagent}_{sample.trial}_{int(args.timeseries_grind_min)}min"
                    )
                    out_png, out_pdf = plot_recommended_timeseries(
                        sample,
                        best["window_size"],
                        args.out_dir,
                        base,
                    )
                    if out_png and out_pdf:
                        print(f"Saved timeseries plot: {out_png}")
                        print(f"Saved timeseries plot: {out_pdf}")

    summary_path = os.path.join(
        args.out_dir, f"moving_average_gpr_cv_{args.experiment}_{args.cv}.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved CV summary: {summary_path}")

    rec_path = os.path.join(
        args.out_dir, f"moving_average_gpr_cv_recommended_{args.experiment}_{args.cv}.csv"
    )
    pd.DataFrame(recommended_rows).to_csv(rec_path, index=False)
    print(f"Saved recommendations: {rec_path}")


if __name__ == "__main__":
    main()
