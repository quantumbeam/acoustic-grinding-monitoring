import argparse
import glob
import json
import math
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from scipy.optimize import lsq_linear
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import r2_score

from fft_processing import calculate_fft_power

# Style setting (LaTeX-less environment)
plt.style.use(["science", "ieee", "no-latex"])


def norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def update_ae_cache(cache_file_path, required_files):
    print("--- Loading AE Power Cache (Policy A) ---")

    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                ae_cache = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load cache file. Rebuilding. Error: {e}")
            ae_cache = {}
    else:
        print("Cache file not found. A new one will be created.")
        ae_cache = {}

    updated_count = 0
    skipped_count = 0

    for file_path in required_files:
        key = norm_path(file_path)

        if key in ae_cache:
            skipped_count += 1
            continue

        new_power = calculate_fft_power(file_path)
        if new_power is None:
            continue

        ae_cache[key] = float(new_power)
        updated_count += 1

    if updated_count > 0:
        print(f"Cache updated: {updated_count} new values (skipped {skipped_count}).")
        try:
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump(ae_cache, f, indent=4)
            print(f"Cache saved to: {cache_file_path}")
        except IOError as e:
            print(f"Error saving cache file: {e}")
    else:
        print(f"Cache hit for all required files (skipped {skipped_count}).")

    return ae_cache


def get_d50(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Dx (50)"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        return float(parts[1])
    except (IOError, ValueError) as e:
        print(f"Could not read or parse D50 from {file_path}: {e}")
    return None


def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def infer_monotone_direction(x_data: np.ndarray, y_data: np.ndarray) -> str:
    x = np.asarray(x_data, dtype=float).reshape(-1)
    y = np.asarray(y_data, dtype=float).reshape(-1)
    if x.size < 2:
        return "increasing"
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return "increasing"
    return "increasing" if corr >= 0.0 else "decreasing"


def scale_to_unit_interval(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    den = max(float(x_max - x_min), 1e-12)
    t = (x - float(x_min)) / den
    return np.clip(t, 0.0, 1.0)


def bernstein_design(t: np.ndarray, degree: int) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    n = int(degree)
    phi = np.zeros((t.size, n + 1), dtype=float)
    for k in range(n + 1):
        c = float(math.comb(n, k))
        phi[:, k] = c * np.power(t, k) * np.power(1.0 - t, n - k)
    return phi


def order_transform_matrix(degree: int, monotone: str) -> np.ndarray:
    n = int(degree)
    sign = 1.0 if monotone == "increasing" else -1.0
    tmat = np.zeros((n + 1, n + 1), dtype=float)
    tmat[:, 0] = 1.0
    for k in range(1, n + 1):
        tmat[k:, k] = sign
    return tmat


def build_bernstein_monotone_mean(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
    monotone: str,
):
    x = np.asarray(x_data, dtype=float).reshape(-1)
    y = np.asarray(y_data, dtype=float).reshape(-1)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Invalid x range.")
    if x_max <= x_min:
        x_max = x_min + 1e-6

    t = scale_to_unit_interval(x, x_min=x_min, x_max=x_max)
    phi = bernstein_design(t, degree=degree)

    tmat = order_transform_matrix(degree=degree, monotone=monotone)
    a = phi @ tmat

    lb = np.concatenate([[-np.inf], np.zeros(degree, dtype=float)])
    ub = np.full(degree + 1, np.inf, dtype=float)
    res = lsq_linear(a, y, bounds=(lb, ub), method="trf")
    if not res.success:
        raise RuntimeError(f"Constrained least squares failed: {res.message}")

    beta = np.asarray(res.x, dtype=float)
    coef = tmat @ beta

    return {
        "type": "bernstein",
        "degree": int(degree),
        "monotone": monotone,
        "x_min": x_min,
        "x_max": x_max,
        "coef": coef,
    }


def predict_bernstein_monotone_mean(mean_model: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    t = scale_to_unit_interval(x, x_min=mean_model["x_min"], x_max=mean_model["x_max"])
    phi = bernstein_design(t, degree=int(mean_model["degree"]))
    return np.asarray(phi @ np.asarray(mean_model["coef"], dtype=float), dtype=float).reshape(-1)


def fit_residual_gp(x_data: np.ndarray, residual: np.ndarray, n_restarts: int = 10):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
    )
    gp.fit(x_data.reshape(-1, 1), residual)
    return gp


def predict_hard_monotone_gp(model: dict, x: np.ndarray, return_std: bool = False, use_monotone_mean: bool = False):
    x = np.asarray(x, dtype=float).reshape(-1)
    y_mean_mono = predict_bernstein_monotone_mean(model["mean_model"], x)
    r_mu, r_std = model["residual_gp"].predict(x.reshape(-1, 1), return_std=True)

    if use_monotone_mean:
        y_mean = y_mean_mono
    else:
        y_mean = y_mean_mono + r_mu

    if return_std:
        return y_mean, r_std
    return y_mean


def compute_nmpiw(y_std: np.ndarray, y_data: np.ndarray) -> tuple[float, float]:
    mpiw = float(np.mean(2.0 * 1.96 * y_std))
    std_y = float(np.std(y_data))
    if std_y == 0.0:
        return mpiw, float("nan")
    return mpiw, mpiw / std_y


def monotone_violation_count(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    monotone_direction: str,
    atol: float = 1e-10,
) -> int:
    dy = np.diff(y_grid)
    if monotone_direction == "decreasing":
        return int(np.sum(dy > atol))
    return int(np.sum(dy < -atol))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train monotone hard-mean(Bernstein) + residual GP models on AE and PSD data (both directions)."
    )
    parser.add_argument("--reagent", type=str, default="all", choices=["NaCl", "Citricacid", "MSG", "all"])
    parser.add_argument("--trial", type=str, default="all", choices=["1st", "2nd", "3rd", "all"])
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument(
        "--constraint",
        type=str,
        default="auto",
        choices=["auto", "increasing", "decreasing"],
        help="Monotone direction for each hard mean fit.",
    )
    args = parser.parse_args()

    target_reagent = None if args.reagent == "all" else args.reagent
    target_trial = None if args.trial == "all" else args.trial
    experiment = "exp2"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, "ae_power_cache.json")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 18,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
        }
    )

    print("--- Scanning for required files ---")

    ae_base_path = os.path.join("data/ae", experiment)
    psd_base_path = os.path.join("data/powder_size_distribution", experiment)

    reagent_pattern = target_reagent if target_reagent else "*"
    trial_pattern = target_trial if target_trial else "*"

    all_psd_files = []
    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, reagent_pattern)):
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, trial_pattern)):
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, "*.csv")))

    required_ae_files = set()
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue

        reagent = path_parts[-3]
        trial = path_parts[-2]

        match = re.search(r"(grind\d+min)", os.path.basename(psd_file))
        if not match:
            continue

        grind_key = match.group(1)
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        pattern = os.path.join(ae_session_path, f"*{grind_key}*.csv")
        required_ae_files.update(glob.glob(pattern))

    print(f"Found {len(required_ae_files)} required AE files.")
    ae_cache = update_ae_cache(cache_file, list(required_ae_files))

    print(f"--- Creating dataset for {experiment} (shared for both directions) ---")
    collected_data = []
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue

        reagent = path_parts[-3]
        trial = path_parts[-2]

        d50 = get_d50(psd_file)
        if d50 is None:
            continue

        match = re.search(r"(grind\d+min)", os.path.basename(psd_file))
        if not match:
            continue

        grind_key = match.group(1)
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        ae_files = sorted(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))
        if not ae_files:
            continue

        ae_power_timeseries = [ae_cache.get(norm_path(f)) for f in ae_files]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]

        if len(ae_power_timeseries) < 4:
            continue

        ae_power_mv2 = np.array(ae_power_timeseries, dtype=float) * 1e6
        smoothed = moving_average(ae_power_mv2, window_size=4)
        if smoothed.size == 0:
            continue

        final_ae_power = float(smoothed[-1])
        collected_data.append((float(d50), final_ae_power, float(re.search(r"grind(\d+)min", grind_key).group(1)), trial, reagent))

    if not collected_data:
        print("No matched data points found.")
        raise SystemExit(1)

    print(f"Collected {len(collected_data)} data points.")
    data_array = np.array(collected_data, dtype=object)

    dataset_path = os.path.join(output_dir, f"{experiment}_monotone_hard_gp_bernstein_dataset_raw.csv")
    dataset_df = pd.DataFrame(data_array, columns=["d50", "ae_power_mV2", "grind_min", "trial", "reagent"])
    dataset_df.to_csv(dataset_path, index=False)
    print(f"Saved dataset: {dataset_path}")

    markers = {"1st": "o", "2nd": "x", "3rd": "^"}
    colors = {"1st": "black", "2nd": "red", "3rd": "blue"}

    all_metrics = []

    for current_reagent in np.unique(data_array[:, 4]):
        print(f"\n=== Reagent: {current_reagent} ===")

        mask = data_array[:, 4] == current_reagent
        reagent_data = data_array[mask]

        d50_vals = np.array(reagent_data[:, 0], dtype=float)
        ae_vals = np.array(reagent_data[:, 1], dtype=float)
        trial_labels = reagent_data[:, 3]

        settings = [
            ("particle2ae", d50_vals, ae_vals),
            ("ae2particle", ae_vals, d50_vals),
        ]

        for direction, x_raw, y_raw in settings:
            x_data = np.asarray(x_raw, dtype=float)
            y_data = np.asarray(y_raw, dtype=float)

            monotone_direction = infer_monotone_direction(x_data, y_data) if args.constraint == "auto" else args.constraint

            mean_model = build_bernstein_monotone_mean(
                x_data,
                y_data,
                degree=args.degree,
                monotone=monotone_direction,
            )
            y_mono = predict_bernstein_monotone_mean(mean_model, x_data)
            residual = y_data - y_mono
            residual_gp = fit_residual_gp(x_data, residual, n_restarts=args.n_restarts)

            model = {
                "method": "monotone_hard_gp_bernstein",
                "experiment": experiment,
                "direction": direction,
                "reagent": current_reagent,
                "mean_model": mean_model,
                "residual_gp": residual_gp,
                "mean_model_type": "bernstein",
                "constraint_direction": monotone_direction,
                "degree": int(args.degree),
            }

            model_path = os.path.join(
                output_dir,
                f"monotone_hard_gp_bernstein_model_{direction}_{current_reagent}_{experiment}.joblib",
            )
            joblib.dump(model, model_path)

            x_plot = np.linspace(x_data.min() * 0.9, x_data.max() * 1.1, 500)
            y_plot_mean_total, y_plot_std = predict_hard_monotone_gp(model, x_plot, return_std=True, use_monotone_mean=False)
            y_plot_mean_mono = predict_hard_monotone_gp(model, x_plot, return_std=False, use_monotone_mean=True)

            y_train_pred_total, y_train_std = predict_hard_monotone_gp(model, x_data, return_std=True, use_monotone_mean=False)
            r2_total = float(r2_score(y_data, y_train_pred_total))
            mpiw, nmpiw = compute_nmpiw(y_train_std, y_data)

            mono_violation = monotone_violation_count(
                x_plot,
                y_plot_mean_mono,
                monotone_direction=monotone_direction,
            )
            residual_mean_abs = float(np.mean(np.abs(residual)))

            all_metrics.append(
                {
                    "direction": direction,
                    "reagent": current_reagent,
                    "method": "monotone_hard_gp_bernstein",
                    "r_squared_total": r2_total,
                    "average_variance": float(np.mean(y_train_std ** 2)),
                    "mpiw": mpiw,
                    "nmpiw": nmpiw,
                    "monotone_mean_violations": mono_violation,
                    "constraint_direction": monotone_direction,
                    "bernstein_degree": int(args.degree),
                    "residual_abs_mean": residual_mean_abs,
                    "residual_gp_kernel_optimized": str(residual_gp.kernel_),
                    "model_path": model_path,
                }
            )

            plt.figure(figsize=(12, 8))
            for t in np.unique(trial_labels):
                m = trial_labels == t
                plt.scatter(
                    x_data[m],
                    y_data[m],
                    marker=markers.get(t, "o"),
                    c=colors.get(t, "black"),
                    s=100,
                    label=t,
                )

            plt.plot(x_plot, y_plot_mean_mono, "k--", label="Monotone mean (hard, Bernstein)")
            plt.plot(x_plot, y_plot_mean_total, "k-", label="Monotone mean + residual GP")
            plt.fill_between(
                x_plot,
                y_plot_mean_total - 1.96 * y_plot_std,
                y_plot_mean_total + 1.96 * y_plot_std,
                alpha=0.2,
                label=r"$\pm 1.96\sigma$ (residual GP)",
            )

            if direction == "particle2ae":
                plt.xlabel(r"$D_{50}~(\mathrm{\mu m})$")
                plt.ylabel(r"Total spectral power ($\mathrm{mV}^2$)")
            else:
                plt.xlabel(r"Total spectral power ($\mathrm{mV}^2$)")
                plt.ylabel(r"$D_{50}~(\mathrm{\mu m})$")

            plt.legend()
            plot_pdf = os.path.join(output_dir, f"{experiment}_monotone_hard_gp_bernstein_plot_{direction}_{current_reagent}.pdf")
            plot_png = os.path.join(output_dir, f"{experiment}_monotone_hard_gp_bernstein_plot_{direction}_{current_reagent}.png")
            plt.savefig(plot_pdf, dpi=300)
            plt.savefig(plot_png, dpi=300)
            plt.close()

            print(
                f"[{direction}] R2(total): {r2_total:.4f} | "
                f"constraint={monotone_direction} | Saved: {model_path}"
            )

    metrics_path = os.path.join(output_dir, f"{experiment}_monotone_hard_gp_bernstein_metrics_both_directions.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    print(f"\nSaved metrics: {metrics_path}")
