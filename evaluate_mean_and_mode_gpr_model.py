import argparse
import glob
import json
import os
import re
from math import erf, sqrt

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from tqdm import tqdm

from fft_processing import calculate_fft_power

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee", "no-latex"])
except Exception:
    pass


def norm_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def update_ae_cache(cache_file_path, required_files):
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                ae_cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            ae_cache = {}
    else:
        ae_cache = {}

    updated_count = 0
    skipped_count = 0

    for file_path in tqdm(required_files, desc="Checking AE files"):
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
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(ae_cache, f, indent=4)
        print(f"Cache updated: {updated_count} new values (skipped {skipped_count}).")
        print(f"Cache saved to: {cache_file_path}")
    else:
        print(f"Cache hit for all required files (skipped {skipped_count}).")

    return ae_cache


def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def read_distribution(file_path):
    sizes = []
    volumes = []
    in_table = False
    try:
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
                    volume = float(parts[1])
                except ValueError:
                    continue
                if np.isfinite(size) and np.isfinite(volume):
                    sizes.append(size)
                    volumes.append(volume)
    except OSError as e:
        print(f"Could not read distribution from {file_path}: {e}")
    return sizes, volumes


def compute_metric_from_distribution(metric, sizes, volumes):
    if not sizes or not volumes:
        return None
    size_arr = np.array(sizes, dtype=float)
    vol_arr = np.array(volumes, dtype=float)
    if size_arr.size == 0 or vol_arr.size == 0:
        return None
    total = float(np.sum(vol_arr))
    if total <= 0.0:
        return None
    if metric in (10, 50, 90):
        cum = np.cumsum(vol_arr)
        target = total * (metric / 100.0)
        return float(np.interp(target, cum, size_arr))
    if metric == "Dmean":
        return float(np.sum(size_arr * vol_arr) / total)
    if metric == "Dmode":
        return float(size_arr[int(np.argmax(vol_arr))])
    return None


def get_metric_value(file_path, metric):
    sizes, volumes = read_distribution(file_path)
    return compute_metric_from_distribution(metric, sizes, volumes)


def get_metric_y_value(metric_value, sizes, volumes):
    if metric_value is None:
        return None
    if not sizes or not volumes:
        return None
    size_arr = np.array(sizes, dtype=float)
    vol_arr = np.array(volumes, dtype=float)
    if size_arr.size == 0 or vol_arr.size == 0:
        return None
    return float(np.interp(metric_value, size_arr, vol_arr))


def select_psd_files_exp2(reagent_filter, trial_filter):
    psd_base_path = os.path.join("data/powder_size_distribution", "exp2")
    reagent_pattern = reagent_filter if reagent_filter else "*"
    trial_pattern = trial_filter if trial_filter else "*"

    groups = {}
    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, reagent_pattern)):
        reagent = os.path.basename(psd_reagent_dir)
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, trial_pattern)):
            trial = os.path.basename(psd_trial_dir)
            for psd_file in glob.glob(os.path.join(psd_trial_dir, "*.csv")):
                m = re.search(r"grind(\d+)min", os.path.basename(psd_file))
                if not m:
                    continue
                grind_min = int(m.group(1))
                key = (reagent, grind_min)
                if key not in groups:
                    groups[key] = []
                groups[key].append((trial, psd_file))

    selected = []
    trial_priority = {"1st": 0, "2nd": 1, "3rd": 2}
    for (reagent, grind_min), entries in sorted(groups.items()):
        entries.sort(key=lambda item: (trial_priority.get(item[0], 99), item[0]))
        trial, psd_file = entries[0]
        selected.append((reagent, trial, grind_min, psd_file))
    return selected


def plot_psd_distributions_exp2(output_root, reagent_filter, trial_filter):
    selected = select_psd_files_exp2(reagent_filter, trial_filter)
    if not selected:
        print("No PSD files found for distribution plots.")
        return

    output_dir = os.path.join(output_root, "psd_distributions")
    os.makedirs(output_dir, exist_ok=True)

    for reagent, trial, grind_min, psd_file in selected:
        sizes, volumes = read_distribution(psd_file)
        if not sizes or not volumes:
            continue
        size_arr = np.array(sizes, dtype=float)
        vol_arr = np.array(volumes, dtype=float)

        d50 = compute_metric_from_distribution(50, sizes, volumes)
        dmean = compute_metric_from_distribution("Dmean", sizes, volumes)
        dmode = compute_metric_from_distribution("Dmode", sizes, volumes)

        plt.figure(figsize=(10, 6))
        plt.plot(size_arr, vol_arr, color="black", linewidth=1.5)

        marker_map = {
            "mean": "s",
            "median (D50)": "o",
            "mode": "^",
        }
        color_map = {
            "mean": "blue",
            "median (D50)": "red",
            "mode": "green",
        }
        linestyle_map = {
            "mean": (0, (5, 3)),
            "median (D50)": (0, (1, 1)),
            "mode": (0, (3, 2, 1, 2)),
        }
        for label, value in [
            ("mean", dmean),
            ("median (D50)", d50),
            ("mode", dmode),
        ]:
            if value is None:
                continue
            y_val = get_metric_y_value(value, sizes, volumes)
            plt.axvline(
                value,
                color=color_map.get(label, "black"),
                linestyle=linestyle_map.get(label, "--"),
                linewidth=1.2,
                alpha=0.8,
            )
            if y_val is not None:
                plt.scatter(
                    [value],
                    [y_val],
                    color=color_map.get(label, "black"),
                    s=55,
                    zorder=3,
                    label=label,
                    marker=marker_map.get(label, "o"),
                )
            else:
                plt.scatter(
                    [value],
                    [0.0],
                    color=color_map.get(label, "black"),
                    s=55,
                    zorder=3,
                    label=label,
                    marker=marker_map.get(label, "o"),
                )

        plt.xscale("log")
        plt.xlabel("Diameter (μm)")
        plt.ylabel("Volume fraction")
        plt.legend()
        plt.tight_layout()

        base_name = f"psd_distribution_{reagent}_{grind_min}min"
        out_pdf = os.path.join(output_dir, f"{base_name}.pdf")
        plt.savefig(out_pdf, dpi=300)
        plt.close()
        print(f"Saved PSD plot: {out_pdf}")


def parse_timestamp(file_path):
    base = os.path.basename(file_path)
    m = re.search(r"(\\d{8})_(\\d{6})", base)
    if not m:
        return None
    try:
        return pd.to_datetime(f"{m.group(1)}_{m.group(2)}", format="%Y%m%d_%H%M%S")
    except ValueError:
        return None


def plot_psd_timeseries_exp2(output_root, reagent_filter, trial_filter):
    psd_base_path = os.path.join("data/powder_size_distribution", "exp2")
    reagent_pattern = reagent_filter if reagent_filter else "*"
    trial_pattern = trial_filter if trial_filter else "*"

    output_dir = os.path.join(output_root, "psd_distributions")
    os.makedirs(output_dir, exist_ok=True)

    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, reagent_pattern)):
        reagent = os.path.basename(psd_reagent_dir)
        by_time = {}
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, trial_pattern)):
            psd_files = glob.glob(os.path.join(psd_trial_dir, "*.csv"))
            for psd_file in psd_files:
                m = re.search(r"grind(\d+)min", os.path.basename(psd_file))
                if not m:
                    continue
                grind_min = int(m.group(1))
                sizes, volumes = read_distribution(psd_file)
                if not sizes or not volumes:
                    continue
                d50 = compute_metric_from_distribution(50, sizes, volumes)
                dmean = compute_metric_from_distribution("Dmean", sizes, volumes)
                dmode = compute_metric_from_distribution("Dmode", sizes, volumes)
                by_time.setdefault(grind_min, []).append((d50, dmean, dmode))

        if not by_time:
            continue

        times = sorted(by_time.keys())
        d50_vals = []
        d50_err = []
        dmean_vals = []
        dmean_err = []
        dmode_vals = []
        dmode_err = []
        for t in times:
            vals = np.array(by_time[t], dtype=float)
            d50_vals.append(float(np.mean(vals[:, 0])))
            d50_err.append(float(np.std(vals[:, 0], ddof=1)) if len(vals) > 1 else 0.0)
            dmean_vals.append(float(np.mean(vals[:, 1])))
            dmean_err.append(float(np.std(vals[:, 1], ddof=1)) if len(vals) > 1 else 0.0)
            dmode_vals.append(float(np.mean(vals[:, 2])))
            dmode_err.append(float(np.std(vals[:, 2], ddof=1)) if len(vals) > 1 else 0.0)

        plt.figure(figsize=(12, 7))
        plt.errorbar(
            times,
            d50_vals,
            yerr=d50_err,
            color="red",
            marker="o",
            linestyle=(0, (1, 1)),
            capsize=4,
            label=f"{dx_label('D50')} (mean±SD)",
        )
        plt.errorbar(
            times,
            dmean_vals,
            yerr=dmean_err,
            color="blue",
            marker="s",
            linestyle=(0, (5, 3)),
            capsize=4,
            label=f"{dx_label('Dmean')} (mean±SD)",
        )
        plt.errorbar(
            times,
            dmode_vals,
            yerr=dmode_err,
            color="green",
            marker="^",
            linestyle=(0, (3, 2, 1, 2)),
            capsize=4,
            label=f"{dx_label('Dmode')} (mean±SD)",
        )
        plt.xlabel("Grind time (min)")
        plt.ylabel("Diameter (μm)")
        plt.legend()
        plt.tight_layout()

        base_name = f"psd_timeseries_{reagent}_error_bar"
        out_pdf = os.path.join(output_dir, f"{base_name}.pdf")
        out_png = os.path.join(output_dir, f"{base_name}.png")
        plt.savefig(out_pdf, dpi=300)
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved PSD timeseries plot: {out_pdf}")
        print(f"Saved PSD timeseries plot: {out_png}")


def fit_gpr_and_save(X_data, y_data, model_path, n_restarts=10):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
    )
    gpr.fit(X_data, y_data)
    joblib.dump(gpr, model_path)
    r2 = gpr.score(X_data, y_data)
    return gpr, float(r2)


def compute_nmpiw(y_std, y_data):
    mpiw = float(np.mean(2.0 * 1.96 * y_std))
    std_y = float(np.std(y_data))
    if std_y == 0.0:
        return mpiw, float("nan")
    return mpiw, mpiw / std_y


def configure_plot_style():
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


def dx_label(dx_name, for_axis=True):
    if dx_name == "D50":
        return r"$D_{50}$" if for_axis else "D50"
    if dx_name == "Dmode":
        return r"$D_{mode}$" if for_axis else "Dmode"
    if dx_name == "Dmean":
        return r"$D_{mean}$" if for_axis else "Dmean"
    return dx_name


def build_exp2_dataset(dx_value, reagent_filter, trial_filter, cache_file):
    experiment = "exp2"
    ae_base_path = os.path.join("data/ae", experiment)
    psd_base_path = os.path.join("data/powder_size_distribution", experiment)

    reagent_pattern = reagent_filter if reagent_filter else "*"
    trial_pattern = trial_filter if trial_filter else "*"

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
        grind_match = re.search(r"grind(\d+)min", grind_key)
        if not grind_match:
            continue
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        pattern = os.path.join(ae_session_path, f"*{grind_key}*.csv")
        required_ae_files.update(glob.glob(pattern))

    print(f"Found {len(required_ae_files)} required AE files.")
    ae_cache = update_ae_cache(cache_file, list(required_ae_files))

    collected_data = []
    for psd_file in tqdm(all_psd_files, desc="Matching data"):
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue

        reagent = path_parts[-3]
        trial = path_parts[-2]

        d_value = get_metric_value(psd_file, dx_value)
        if d_value is None:
            continue

        match = re.search(r"(grind\d+min)", os.path.basename(psd_file))
        if not match:
            continue

        grind_key = match.group(1)
        grind_match = re.search(r"grind(\d+)min", grind_key)
        if not grind_match:
            continue
        grind_min = float(grind_match.group(1))
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
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

        final_ae_power = float(smoothed[-1])
        collected_data.append((float(d_value), final_ae_power, grind_min, trial, reagent))

    return collected_data


def train_exp2_models(dx_name, dx_value, output_dir, reagent_filter, trial_filter, cache_file):
    print(f"--- Training models for {dx_name} ---")
    collected_data = build_exp2_dataset(dx_value, reagent_filter, trial_filter, cache_file)
    if not collected_data:
        print("No matched data points found.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    data_array = np.array(collected_data, dtype=object)

    dataset_path = os.path.join(output_dir, "gpr_dataset_raw.csv")
    dataset_df = pd.DataFrame(
        data_array,
        columns=["particle_size", "ae_power_mV2", "grind_min", "trial", "reagent"],
    )
    dataset_df.to_csv(dataset_path, index=False)
    print(f"Saved dataset: {dataset_path}")

    markers = {"1st": "o", "2nd": "x", "3rd": "^"}
    colors = {"1st": "black", "2nd": "red", "3rd": "blue"}

    all_metrics = []
    for current_reagent in np.unique(data_array[:, 4]):
        print(f"\n=== Reagent: {current_reagent} ===")

        mask = data_array[:, 4] == current_reagent
        reagent_data = data_array[mask]

        size_vals = np.array(reagent_data[:, 0], dtype=float)
        ae_vals = np.array(reagent_data[:, 1], dtype=float)
        trial_labels = reagent_data[:, 3]

        direction = "particle2ae"
        X_data = size_vals.reshape(-1, 1)
        y_data = ae_vals

        model_path = os.path.join(output_dir, f"gpr_model_{direction}_{current_reagent}.joblib")
        gpr, r2 = fit_gpr_and_save(X_data, y_data, model_path)

        X_plot = np.linspace(X_data.min() * 0.9, X_data.max() * 1.1, 500).reshape(-1, 1)
        y_mean, y_std = gpr.predict(X_plot, return_std=True)
        mpiw, nmpiw = compute_nmpiw(y_std, y_data)

        all_metrics.append(
            {
                "direction": direction,
                "reagent": current_reagent,
                "r_squared": r2,
                "average_variance": float(np.mean(y_std**2)),
                "mpiw": mpiw,
                "nmpiw": nmpiw,
                "kernel_optimized": str(gpr.kernel_),
                "model_path": model_path,
            }
        )

        plt.figure(figsize=(12, 8))
        for t in np.unique(trial_labels):
            m = trial_labels == t
            plt.scatter(
                X_data[m],
                y_data[m],
                marker=markers.get(t, "o"),
                c=colors.get(t, "black"),
                s=100,
                label=t,
            )

        plt.plot(X_plot, y_mean, "k-")
        plt.fill_between(
            X_plot.ravel(),
            y_mean - 1.96 * y_std,
            y_mean + 1.96 * y_std,
            alpha=0.2,
        )
        plt.xlabel(f"{dx_label(dx_name)} (\\mathrm{{\\mu m}})")
        plt.ylabel(r"Total spectral power ($\mathrm{mV}^2$)")
        plt.legend()
        plot_path_pdf = os.path.join(output_dir, f"gpr_plot_{direction}_{current_reagent}.pdf")
        plt.savefig(plot_path_pdf, dpi=300)
        plt.close()

        print(f"[{direction}] R2: {r2:.4f} | Saved: {model_path}")

        direction = "ae2particle"
        X_data = ae_vals.reshape(-1, 1)
        y_data = size_vals

        model_path = os.path.join(output_dir, f"gpr_model_{direction}_{current_reagent}.joblib")
        gpr, r2 = fit_gpr_and_save(X_data, y_data, model_path)

        X_plot = np.linspace(X_data.min() * 0.9, X_data.max() * 1.1, 500).reshape(-1, 1)
        y_mean, y_std = gpr.predict(X_plot, return_std=True)
        mpiw, nmpiw = compute_nmpiw(y_std, y_data)

        all_metrics.append(
            {
                "direction": direction,
                "reagent": current_reagent,
                "r_squared": r2,
                "average_variance": float(np.mean(y_std**2)),
                "mpiw": mpiw,
                "nmpiw": nmpiw,
                "kernel_optimized": str(gpr.kernel_),
                "model_path": model_path,
            }
        )

        plt.figure(figsize=(12, 8))
        for t in np.unique(trial_labels):
            m = trial_labels == t
            plt.scatter(
                X_data[m],
                y_data[m],
                marker=markers.get(t, "o"),
                c=colors.get(t, "black"),
                s=100,
                label=t,
            )

        plt.plot(X_plot, y_mean, "k-")
        plt.fill_between(
            X_plot.ravel(),
            y_mean - 1.96 * y_std,
            y_mean + 1.96 * y_std,
            alpha=0.2,
        )
        plt.xlabel(r"Total spectral power ($\mathrm{mV}^2$)")
        plt.ylabel(f"{dx_label(dx_name)} (\\mathrm{{\\mu m}})")
        plt.legend()
        plot_path_pdf = os.path.join(output_dir, f"gpr_plot_{direction}_{current_reagent}.pdf")
        plt.savefig(plot_path_pdf, dpi=300)
        plt.close()

        print(f"[{direction}] R2: {r2:.4f} | Saved: {model_path}")

    metrics_path = os.path.join(output_dir, "gpr_metrics_both_directions.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    print(f"\nSaved metrics: {metrics_path}")
    return metrics_path


def aggregate_metrics(metrics_paths, output_dir):
    rows = []
    for metric_name, metrics_path in metrics_paths:
        if not metrics_path or not os.path.exists(metrics_path):
            continue
        try:
            df = pd.read_csv(metrics_path)
        except OSError:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["metric"] = metric_name
        rows.append(df)
    if not rows:
        print("No metrics to aggregate.")
        return None
    combined = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(output_dir, "gpr_metrics_both_directions.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved aggregated metrics: {out_path}")
    return out_path


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]


def parse_timestamp_key(filename: str):
    base = os.path.basename(filename)
    m = re.match(r"(\d{8})_(\d{6})", base)
    if m:
        return (m.group(1), m.group(2), base)
    return ("", "", base)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def evaluate_exp3_control(dx_name, dx_value, output_dir, cache_file):
    print(f"--- Evaluating exp3 control for {dx_name} ---")
    experiment = "exp3"
    psd_base_path = os.path.join("data/powder_size_distribution", experiment)
    ae_base_path = os.path.join("data/ae", experiment)
    ae_scale_to_mV2 = 1e6

    if not os.path.isdir(psd_base_path):
        print("No exp3 PSD data found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    def load_ae_power_cache(cache_path):
        if not os.path.exists(cache_path):
            return {}
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save_ae_power_cache(cache_path, cache_data):
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, sort_keys=True)
        except OSError:
            pass

    ae_power_cache = load_ae_power_cache(cache_file)
    cache_dirty = False

    def get_cached_ae_power(file_path):
        nonlocal cache_dirty
        if file_path is None:
            return None
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            return None
        cached = ae_power_cache.get(file_path)
        if isinstance(cached, dict) and cached.get("mtime") == mtime:
            cached_power = cached.get("power")
            if cached_power is not None and np.isfinite(cached_power):
                return float(cached_power)
        p = calculate_fft_power(file_path)
        if p is None or not np.isfinite(p):
            return None
        ae_power_cache[file_path] = {"mtime": mtime, "power": float(p)}
        cache_dirty = True
        return float(p)

    materials = [
        os.path.basename(d)
        for d in glob.glob(os.path.join(psd_base_path, "*"))
        if os.path.isdir(d)
    ]
    materials.sort(key=natural_keys)

    rows = []

    for material in materials:
        model_a_path = os.path.join(output_dir, f"gpr_model_particle2ae_{material}.joblib")
        if not os.path.exists(model_a_path):
            print(f"Warning: Method P2AE model not found for {material}")
            continue
        gpr_a = joblib.load(model_a_path)

        model_b_path = os.path.join(output_dir, f"gpr_model_ae2particle_{material}.joblib")
        if not os.path.exists(model_b_path):
            print(f"Warning: Method AE2P model not found for {material}")
            continue
        gpr_b = joblib.load(model_b_path)

        target_files_sample = glob.glob(os.path.join(psd_base_path, material, "1st", "*.csv"))
        targets = []
        for f in target_files_sample:
            m = re.search(r"_for_?(\d+)um", os.path.basename(f))
            if m:
                targets.append(int(m.group(1)))
        targets = sorted(set(targets))

        for target_val in targets:
            x_target = np.array([[float(target_val)]], dtype=float)
            ae_mu, ae_std = gpr_a.predict(x_target, return_std=True)
            a_ae_th = float(ae_mu[0])

            for trial in ["1st", "2nd", "3rd"]:
                psd_dir = os.path.join(psd_base_path, material, trial)
                psd_candidates = glob.glob(os.path.join(psd_dir, "*.csv"))
                psd_file = None
                for f in psd_candidates:
                    if re.search(f"_for_?{target_val}um", os.path.basename(f)):
                        psd_file = f
                        break

                measured_value = get_metric_value(psd_file, dx_value) if psd_file else None
                measured_value_f = safe_float(measured_value)

                ae_dir = os.path.join(ae_base_path, material, trial)
                ae_candidates = glob.glob(os.path.join(ae_dir, "*.csv"))
                ae_files = [
                    f
                    for f in ae_candidates
                    if re.search(f"_for_?{target_val}um", os.path.basename(f))
                ]
                ae_files.sort(key=parse_timestamp_key)

                last_file = ae_files[-1] if len(ae_files) >= 1 else None
                second_last_file = ae_files[-2] if len(ae_files) >= 2 else None

                ae_values = {}
                ae_series = []
                for f in ae_files:
                    p = get_cached_ae_power(f)
                    if p is None or not np.isfinite(p):
                        continue
                    v = float(p) * ae_scale_to_mV2
                    ae_values[f] = v
                    ae_series.append(v)
                ae_last = ae_values.get(last_file)
                ae_second = ae_values.get(second_last_file)

                ae_smoothed_last = None
                if len(ae_series) >= 4:
                    smoothed = moving_average(np.array(ae_series, dtype=float), window_size=4)
                    if smoothed.size > 0 and np.isfinite(smoothed[-1]):
                        ae_smoothed_last = float(smoothed[-1])

                a_is_expected_cross = False
                if ae_last is not None and ae_second is not None:
                    if (ae_last < a_ae_th) and (ae_second >= a_ae_th):
                        a_is_expected_cross = True

                a_total_deviation = None
                a_total_deviation_percent = None
                if measured_value_f is not None:
                    a_total_deviation = float(target_val) - measured_value_f
                    if target_val != 0:
                        a_total_deviation_percent = (a_total_deviation / float(target_val)) * 100.0

                b_pred = None
                b_sigma = None
                ae_for_estimation = ae_smoothed_last if ae_smoothed_last is not None else ae_last
                if ae_for_estimation is not None:
                    y, s = gpr_b.predict(
                        np.array([[ae_for_estimation]], dtype=float), return_std=True
                    )
                    b_pred = float(y[0])
                    b_sigma = float(s[0])

                b_error = None
                b_error_percent = None
                b_error_in_range = None
                if b_pred is not None and measured_value_f is not None:
                    b_error = measured_value_f - b_pred
                    if measured_value_f != 0:
                        b_error_percent = (b_error / measured_value_f) * 100.0
                    if b_sigma is not None:
                        b_error_in_range = abs(b_error) <= b_sigma

                rows.append(
                    {
                        "Material": material,
                        "Trial": trial,
                        "Target_Size": target_val,
                        "Common_Measured_Value": measured_value_f,
                        "Common_AE_Last_mV2": ae_last,
                        "P2AE_AE_Threshold": a_ae_th,
                        "P2AE_Is_ExpectedCross": a_is_expected_cross,
                        "P2AE_Total_Deviation": a_total_deviation,
                        "P2AE_Total_Deviation_Percent": a_total_deviation_percent,
                        "AE2P_Predicted_Value": b_pred,
                        "AE2P_Predicted_Sigma": b_sigma,
                        "AE2P_Estimation_Error": b_error,
                        "AE2P_Estimation_Error_Percent": b_error_percent,
                        "AE2P_Est_Error_In_GPR_Range": b_error_in_range,
                    }
                )

    df = pd.DataFrame(rows)
    if cache_dirty:
        save_ae_power_cache(cache_file, ae_power_cache)

    if df.empty:
        print("No data processed.")
        return

    detail_path = os.path.join(output_dir, "evaluation_detail.csv")
    df.to_csv(detail_path, index=False)
    print(f"Saved detailed results to: {detail_path}")

    plot_df = df.copy()
    plot_df["AE2P_Measured_Error"] = (
        plot_df["Common_Measured_Value"] - plot_df["AE2P_Predicted_Value"]
    )
    plot_df["AE2P_m"] = plot_df["AE2P_Measured_Error"] / plot_df["AE2P_Predicted_Sigma"]
    plot_df["AE2P_p"] = plot_df["AE2P_m"].apply(
        lambda v: normal_cdf(v) if np.isfinite(v) else np.nan
    )
    plot_df = plot_df.dropna(subset=["AE2P_m", "AE2P_p", "Material", "Trial", "Target_Size"])
    m_boundary = 1.96
    plot_df["AE2P_p_ge_threshold"] = plot_df["AE2P_m"].abs() <= m_boundary

    if not plot_df.empty:
        export_cols = [
            "Material",
            "Trial",
            "Target_Size",
            "Common_Measured_Value",
            "AE2P_Predicted_Value",
            "AE2P_Predicted_Sigma",
            "AE2P_Measured_Error",
            "AE2P_m",
            "AE2P_p",
            "AE2P_p_ge_threshold",
        ]
        export_cols = [c for c in export_cols if c in plot_df.columns]
        export_df = plot_df[export_cols].copy()
        export_df["AE2P_Predicted_95PI"] = export_df["AE2P_Predicted_Sigma"] * 1.96
        export_df = export_df.drop(columns=["AE2P_Predicted_Sigma"])
        export_path = os.path.join(output_dir, "ae2p_upper_error_points.csv")
        export_df.to_csv(export_path, index=False)
        print(f"Saved detail plot data to: {export_path}")

        plot_df = plot_df.reset_index(drop=True)
        plot_df["x_label"] = plot_df.apply(
            lambda row: f"{row['Material']} {row['Target_Size']} $\\mu$m",
            axis=1,
        )
        unique_labels = list(dict.fromkeys(plot_df["x_label"].tolist()))
        x_lookup = {label: idx for idx, label in enumerate(unique_labels)}
        x_vals = plot_df["x_label"].map(x_lookup).astype(float).values

        plt.figure(figsize=(14, 8))
        mask_true = plot_df["AE2P_p_ge_threshold"] == True
        mask_false = plot_df["AE2P_p_ge_threshold"] == False
        plt.scatter(
            x_vals[mask_true],
            plot_df.loc[mask_true, "AE2P_m"].values,
            s=80,
            c="black",
            marker="o",
            label="Within prediction range",
        )
        plt.scatter(
            x_vals[mask_false],
            plot_df.loc[mask_false, "AE2P_m"].values,
            s=80,
            c="black",
            marker="^",
            label="Outside prediction range",
        )
        plt.axhline(m_boundary, color="black", linewidth=1.0, alpha=0.6, linestyle="--")
        plt.axhline(-m_boundary, color="black", linewidth=1.0, alpha=0.6, linestyle="--")
        max_abs = float(np.nanmax(np.abs(plot_df["AE2P_m"].values)))
        max_abs = max(max_abs, m_boundary)
        if np.isfinite(max_abs) and max_abs > 0.0:
            plt.ylim(-max_abs * 1.1, max_abs * 1.1)
        plt.xticks(
            np.arange(len(unique_labels), dtype=float),
            unique_labels,
            rotation=45,
            ha="right",
            fontsize=plt.rcParams["xtick.labelsize"],
        )
        plt.xlabel("")
        plt.ylabel("Normalized Error")
        plt.legend()
        plt.tight_layout()
        plot_pdf_path = os.path.join(output_dir, "ae2p_error.pdf")
        plt.savefig(plot_pdf_path)
        plt.close()
        print(f"Saved detail plot to: {plot_pdf_path}")

    summary_rows = []
    for (mat, tgt), g in df.groupby(["Material", "Target_Size"]):
        mu_trial = g["Common_Measured_Value"].mean()
        sigma_trial = g["Common_Measured_Value"].std(ddof=1)

        mu_gpr = g["AE2P_Predicted_Value"].mean()
        mean_sigma_gpr = g["AE2P_Predicted_Sigma"].mean()
        mean_est_error = g["AE2P_Estimation_Error"].mean()
        mean_est_error_pct = g["AE2P_Estimation_Error_Percent"].mean()
        ae2p_in_gpr_range = None
        if pd.notnull(mean_est_error) and pd.notnull(mean_sigma_gpr):
            ae2p_in_gpr_range = abs(mean_est_error) <= mean_sigma_gpr

        mean_total_dev = g["P2AE_Total_Deviation"].mean()
        mean_total_dev_pct = g["P2AE_Total_Deviation_Percent"].mean()

        meas_str = f"{mu_trial:.2f} ± {sigma_trial:.2f}" if pd.notnull(mu_trial) else "N/A"
        gpr_str = f"{mu_gpr:.2f} ± {mean_sigma_gpr:.2f}" if pd.notnull(mu_gpr) else "N/A"

        summary_rows.append(
            {
                "Material": mat,
                "Target_Size": tgt,
                "AE2P_GPR_Prediction": gpr_str,
                "Common_Measured_Mean": meas_str,
                "AE2P_Estimation_Error": mean_est_error,
                "AE2P_Estimation_Error_Percent": mean_est_error_pct,
                "AE2P_Est_Error_In_GPR_Range": ae2p_in_gpr_range,
                "P2AE_Total_Deviation": mean_total_dev,
                "P2AE_Total_Deviation_Percent": mean_total_dev_pct,
                "num_AE2P_mu_GPR": mu_gpr,
                "num_AE2P_sigma_GPR": mean_sigma_gpr,
                "num_Common_mu_trial": mu_trial,
                "num_Common_sigma_trial": sigma_trial,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    out_summary = os.path.join(output_dir, "evaluation_summary_for_table.csv")
    df_summary.to_csv(out_summary, index=False)
    print(f"Summary table saved to: {out_summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Train exp2 GPR models and evaluate exp3 control for D50/Dmean/Dmode."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["D50", "Dmean", "Dmode"],
        choices=["D50", "Dmean", "Dmode"],
    )
    parser.add_argument(
        "--reagent",
        type=str,
        default="all",
        choices=["NaCl", "Citricacid", "MSG", "all"],
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="all",
        choices=["1st", "2nd", "3rd", "all"],
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-psd-plot", action="store_true")
    args = parser.parse_args()

    configure_plot_style()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, "ae_power_cache.json")

    metric_map = {"D50": 50, "Dmode": "Dmode", "Dmean": "Dmean"}
    reagent_filter = None if args.reagent == "all" else args.reagent
    trial_filter = None if args.trial == "all" else args.trial
    metrics_paths = []
    output_root = os.path.join("results", "SI_figs", "mean_and_mode")

    for metric in args.metrics:
        output_dir = os.path.join(output_root, metric)
        dx_value = metric_map[metric]

        if not args.skip_train:
            metrics_path = train_exp2_models(
                metric, dx_value, output_dir, reagent_filter, trial_filter, cache_file
            )
            metrics_paths.append((metric, metrics_path))

        if not args.skip_eval:
            evaluate_exp3_control(metric, dx_value, output_dir, cache_file)

    if metrics_paths:
        aggregate_metrics(metrics_paths, output_root)

    if not args.skip_psd_plot:
        plot_psd_distributions_exp2(output_root, reagent_filter, trial_filter)
        plot_psd_timeseries_exp2(output_root, reagent_filter, trial_filter)


if __name__ == "__main__":
    main()
