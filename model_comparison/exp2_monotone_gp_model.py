import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from tqdm import tqdm

from fft_processing import calculate_fft_power
from monotone_svgp import MonotoneConfig, MonotoneSVGPRegressor, save_model_npz

plt.style.use(["science", "ieee", "no-latex"])


def norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def update_ae_cache(cache_file_path, required_files):
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                ae_cache = json.load(f)
        except (IOError, json.JSONDecodeError):
            ae_cache = {}
    else:
        ae_cache = {}

    updated_count = 0
    for file_path in tqdm(required_files, desc="Checking AE files"):
        key = norm_path(file_path)
        if key in ae_cache:
            continue
        new_power = calculate_fft_power(file_path)
        if new_power is None:
            continue
        ae_cache[key] = float(new_power)
        updated_count += 1

    if updated_count > 0:
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(ae_cache, f, indent=2)
    return ae_cache


def get_d50(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Dx (50)"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        return float(parts[1])
    except (IOError, ValueError):
        return None
    return None


def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def build_shared_dataset(experiment: str, reagent: str, trial: str):
    ae_base_path = os.path.join("data/ae", experiment)
    psd_base_path = os.path.join("data/powder_size_distribution", experiment)

    reagent_pattern = reagent if reagent != "all" else "*"
    trial_pattern = trial if trial != "all" else "*"

    all_psd_files = []
    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, reagent_pattern)):
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, trial_pattern)):
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, "*.csv")))

    required_ae_files = set()
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue

        reagent_name = path_parts[-3]
        trial_name = path_parts[-2]
        match = re.search(r"(grind\d+min)", os.path.basename(psd_file))
        if not match:
            continue

        grind_key = match.group(1)
        ae_session_path = os.path.join(ae_base_path, reagent_name, trial_name)
        required_ae_files.update(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, "ae_power_cache.json")
    ae_cache = update_ae_cache(cache_file, list(required_ae_files))

    collected_data = []
    for psd_file in tqdm(all_psd_files, desc="Matching data"):
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

        ae_power_timeseries = [ae_cache.get(norm_path(f)) for f in ae_files]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]
        if len(ae_power_timeseries) < 4:
            continue

        ae_power_mV2 = np.array(ae_power_timeseries, dtype=float) * 1e6
        smoothed = moving_average(ae_power_mV2, window_size=4)
        if smoothed.size == 0:
            continue

        collected_data.append((float(d50), float(smoothed[-1]), grind_min, trial_name, reagent_name))

    if not collected_data:
        raise RuntimeError("No matched data points found.")

    return np.array(collected_data, dtype=object)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def infer_derivative_sign(x_data: np.ndarray, y_data: np.ndarray) -> int:
    x = np.asarray(x_data, dtype=float).reshape(-1)
    y = np.asarray(y_data, dtype=float).reshape(-1)
    if x.size < 2:
        return -1
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return -1
    return 1 if corr >= 0.0 else -1


def main():
    parser = argparse.ArgumentParser(description="Train monotone SVGP models on exp2 data.")
    parser.add_argument("--reagent", type=str, default="all", choices=["NaCl", "Citricacid", "MSG", "all"])
    parser.add_argument("--trial", type=str, default="all", choices=["1st", "2nd", "3rd", "all"])
    parser.add_argument("--m", type=int, default=50, help="Number of derivative constraint points")
    parser.add_argument("--nu", type=float, default=0.05, help="Derivative softness for monotone probit penalty")
    parser.add_argument("--inducing", type=int, default=50)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--monotone-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--constraint",
        type=str,
        default="auto",
        choices=["auto", "decreasing", "increasing"],
        help="Monotone derivative direction for all models or inferred per model.",
    )
    args = parser.parse_args()

    experiment = "exp2"
    output_dir = os.path.join("model_comparison", "monotone_gp")
    os.makedirs(output_dir, exist_ok=True)

    data_array = build_shared_dataset(experiment=experiment, reagent=args.reagent, trial=args.trial)

    dataset_path = os.path.join(output_dir, f"{experiment}_monotone_gp_dataset_raw.csv")
    pd.DataFrame(data_array, columns=["d50", "ae_power_mV2", "grind_min", "trial", "reagent"]).to_csv(
        dataset_path, index=False
    )

    all_metrics = []
    markers = {"1st": "o", "2nd": "x", "3rd": "^"}
    colors = {"1st": "black", "2nd": "red", "3rd": "blue"}

    for current_reagent in np.unique(data_array[:, 4]):
        mask = data_array[:, 4] == current_reagent
        reagent_data = data_array[mask]

        d50_vals = np.array(reagent_data[:, 0], dtype=float)
        ae_vals = np.array(reagent_data[:, 1], dtype=float)
        trial_labels = reagent_data[:, 3]

        for direction in ["particle2ae", "ae2particle"]:
            if direction == "particle2ae":
                x_data = d50_vals.reshape(-1, 1)
                y_data = ae_vals.reshape(-1, 1)
                x_label = r"$D_{50}~(\mathrm{\mu m})$"
                y_label = r"Total spectral power ($\mathrm{mV}^2$)"
            else:
                x_data = ae_vals.reshape(-1, 1)
                y_data = d50_vals.reshape(-1, 1)
                x_label = r"Total spectral power ($\mathrm{mV}^2$)"
                y_label = r"$D_{50}~(\mathrm{\mu m})$"

            cfg = MonotoneConfig(
                num_derivative_points=args.m,
                derivative_softness=args.nu,
                inducing_points=args.inducing,
                monotone_weight=args.monotone_weight,
                learning_rate=args.lr,
                steps=args.steps,
                seed=args.seed,
                derivative_sign=-1,
            )
            if args.constraint == "decreasing":
                cfg.derivative_sign = -1
            elif args.constraint == "increasing":
                cfg.derivative_sign = 1
            else:
                cfg.derivative_sign = infer_derivative_sign(x_data, y_data)

            reg = MonotoneSVGPRegressor(config=cfg).fit(x_data, y_data, verbose=args.verbose)

            model_path = os.path.join(
                output_dir,
                f"monotone_gp_model_{direction}_{current_reagent}_{experiment}.npz",
            )
            save_model_npz(
                model_path,
                reg,
                extra_config={
                    "experiment": experiment,
                    "direction": direction,
                    "reagent": str(current_reagent),
                    "dataset_hash": int(pd.util.hash_pandas_object(pd.DataFrame(reagent_data)).sum()),
                },
            )

            x_plot = np.linspace(float(np.min(x_data) * 0.9), float(np.max(x_data) * 1.1), 500).reshape(-1, 1)
            y_mean, y_var = reg.predict_f(x_plot)
            y_std = np.sqrt(np.maximum(y_var, 0.0))

            y_pred_train, _ = reg.predict_f(x_data)
            y_pred_train = y_pred_train.reshape(-1)
            y_true_train = y_data.reshape(-1)
            mono_stats = reg.evaluate_monotonicity(x_plot)

            all_metrics.append(
                {
                    "direction": direction,
                    "reagent": current_reagent,
                    "rmse_train": rmse(y_true_train, y_pred_train),
                    "mae_train": mae(y_true_train, y_pred_train),
                    "average_variance": float(np.mean(y_var)),
                    "violation_rate": mono_stats["violation_rate"],
                    "max_violation_derivative": mono_stats["max_violation_derivative"],
                    "mean_derivative": mono_stats["mean_derivative"],
                    "final_elbo": reg.training_summary.get("final_elbo"),
                    "final_monotone_penalty": reg.training_summary.get("final_monotone_penalty"),
                    "constraint_direction": "increasing" if cfg.derivative_sign > 0 else "decreasing",
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
            plt.plot(x_plot, y_mean, "k-")
            plt.fill_between(x_plot.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std, alpha=0.2)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.tight_layout()
            plot_png = os.path.join(output_dir, f"{experiment}_monotone_gp_plot_{direction}_{current_reagent}.png")
            plt.savefig(plot_png, dpi=300)
            plt.close()

            print(
                f"[{current_reagent}][{direction}] "
                f"RMSE={all_metrics[-1]['rmse_train']:.4f}, "
                f"violation_rate={all_metrics[-1]['violation_rate']:.4f}, "
                f"constraint={all_metrics[-1]['constraint_direction']}"
            )

    metrics_path = os.path.join(output_dir, f"{experiment}_monotone_gp_metrics_both_directions.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
