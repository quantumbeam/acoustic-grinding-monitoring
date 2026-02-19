import pandas as pd
import numpy as np
import os
import glob
import re
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from fft_processing import calculate_fft_power

import matplotlib.pyplot as plt
import scienceplots

# Style setting (force non-TeX rendering for portable execution)
plt.style.use(['science', 'ieee', 'no-latex'])


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def norm_path(p: str) -> str:
    """Normalize path to be CWD-independent."""
    return os.path.normpath(os.path.abspath(p))


# ------------------------------------------------------------
# Cache (Policy A: no recompute if exists)
# ------------------------------------------------------------
def update_ae_cache(cache_file_path, required_files):
    """
    Cache policy A:
      - If cache has the entry, DO NOT recompute.
      - Compute only for missing entries.
    """
    print("--- Loading AE Power Cache (Policy A) ---")

    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r') as f:
                ae_cache = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load cache file. Rebuilding. Error: {e}")
            ae_cache = {}
    else:
        print("Cache file not found. A new one will be created.")
        ae_cache = {}

    updated_count = 0
    skipped_count = 0

    for file_path in tqdm(required_files, desc="Checking AE files"):
        key = norm_path(file_path)

        # Cache hit → skip recomputation
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
            with open(cache_file_path, 'w') as f:
                json.dump(ae_cache, f, indent=4)
            print(f"Cache saved to: {cache_file_path}")
        except IOError as e:
            print(f"Error saving cache file: {e}")
    else:
        print(f"Cache hit for all required files (skipped {skipped_count}).")

    return ae_cache


# ------------------------------------------------------------
# PSD utilities
# ------------------------------------------------------------
def parse_timestamp_from_filename(file_path: str) -> datetime | None:
    match = re.search(r'(\d{8}_\d{6})', os.path.basename(file_path))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def cumulative_norm_variance(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for i in range(len(values)):
        segment = values[:i + 1]
        mean_seg = float(np.mean(segment))
        if mean_seg != 0.0:
            result[i] = float(np.var(segment) / (mean_seg ** 2))
    return result


def log_detrended_variance(values: np.ndarray, time_min: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    valid_mask = values > 0.0
    if np.count_nonzero(valid_mask) < 2:
        return result
    t_valid = time_min[valid_mask]
    y_valid = np.log(values[valid_mask])
    slope, intercept = np.polyfit(t_valid, y_valid, 1)
    fit = slope * t_valid + intercept
    residuals = y_valid - fit
    res_full = np.full(values.shape, np.nan, dtype=float)
    res_full[valid_mask] = residuals
    for i in range(len(values)):
        segment = res_full[:i + 1]
        segment = segment[~np.isnan(segment)]
        if segment.size >= 2:
            result[i] = float(np.var(segment))
    return result


def log_detrended_residuals(values: np.ndarray, time_min: np.ndarray) -> np.ndarray:
    valid_mask = values > 0.0
    if np.count_nonzero(valid_mask) < 2:
        return np.array([], dtype=float)
    t_valid = time_min[valid_mask]
    y_valid = np.log(values[valid_mask])
    slope, intercept = np.polyfit(t_valid, y_valid, 1)
    fit = slope * t_valid + intercept
    return y_valid - fit


def log_trend_fit(time_min: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    valid_mask = values > 0.0
    if np.count_nonzero(valid_mask) < 2:
        return None
    t_valid = time_min[valid_mask]
    y_valid = np.log(values[valid_mask])
    slope, intercept = np.polyfit(t_valid, y_valid, 1)
    fit = slope * t_valid + intercept
    corr = float(np.corrcoef(y_valid, fit)[0, 1])
    return t_valid, y_valid, fit, corr


def get_d50(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().startswith('Dx (50)'):
                    parts = line.split(',')
                    if len(parts) > 1:
                        return float(parts[1])
    except (IOError, ValueError) as e:
        print(f"Could not read or parse D50 from {file_path}: {e}")
    return None


def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def linear_fit_r2(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    y_pred = slope * x_vals + intercept
    ss_res = float(np.sum((y_vals - y_pred) ** 2))
    ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
    r2 = float('nan') if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
    return slope, intercept, r2


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze AE variance trends for discussion plots.')
    parser.add_argument('--reagent', type=str, default='all',
                        choices=['NaCl', 'Citricacid', 'MSG', 'all'])
    parser.add_argument('--trial', type=str, default='all',
                        choices=['1st', '2nd', '3rd', 'all'])
    parser.add_argument('--variance-mode', type=str, default='log_detrended',
                        choices=['cumulative', 'log_detrended'],
                        help='Mode for variance over time.')
    args = parser.parse_args()

    TARGET_REAGENT = None if args.reagent == 'all' else args.reagent
    TARGET_TRIAL = None if args.trial == 'all' else args.trial
    EXPERIMENT = 'exp2'

    # Cache file fixed to script location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_FILE = os.path.join(SCRIPT_DIR, 'ae_power_cache.json')

    VARIANCE_OUTPUT_DIR = os.path.join('results', 'discussion', 'variance_analysis')

    # ------------------------------------------------------------
    # Plot settings
    # ------------------------------------------------------------
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 32,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 18,
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans'
    })
    axes_label_size = plt.rcParams['axes.labelsize']

    # ------------------------------------------------------------
    # Scan files
    # ------------------------------------------------------------
    print("--- Scanning for required files ---")

    ae_base_path = os.path.join('data/ae', EXPERIMENT)
    psd_base_path = os.path.join('data/powder_size_distribution', EXPERIMENT)

    reagent_pattern = TARGET_REAGENT if TARGET_REAGENT else '*'
    trial_pattern = TARGET_TRIAL if TARGET_TRIAL else '*'

    all_psd_files = []
    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, reagent_pattern)):
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, trial_pattern)):
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, '*.csv')))

    required_ae_files = set()
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue

        reagent = path_parts[-3]
        trial = path_parts[-2]

        match = re.search(r'(grind\d+min)', os.path.basename(psd_file))
        if not match:
            continue

        grind_key = match.group(1)
        grind_match = re.search(r'grind(\d+)min', grind_key)
        if not grind_match:
            continue
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        pattern = os.path.join(ae_session_path, f"*{grind_key}*.csv")
        required_ae_files.update(glob.glob(pattern))

    print(f"Found {len(required_ae_files)} required AE files.")
    ae_cache = update_ae_cache(CACHE_FILE, list(required_ae_files))

    # ------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------
    print(f"--- Preparing variance data for {EXPERIMENT} ---")
    ae_series_by_trial = {}
    ae_d50_points = []

    for psd_file in tqdm(all_psd_files, desc="Matching data for AE vs D50"):
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue

        reagent = path_parts[-3]
        trial = path_parts[-2]

        d50 = get_d50(psd_file)
        if d50 is None:
            continue

        match = re.search(r'(grind\d+min)', os.path.basename(psd_file))
        if not match:
            continue

        grind_key = match.group(1)
        grind_match = re.search(r'grind(\d+)min', grind_key)
        if not grind_match:
            continue
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        ae_files = sorted(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))
        if not ae_files:
            continue

        ae_power_timeseries = [ae_cache.get(norm_path(f)) for f in ae_files]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]

        if len(ae_power_timeseries) < 4:
            continue

        # Convert to mV^2
        ae_power_mV2 = np.array(ae_power_timeseries, dtype=float) * 1e6

        smoothed = moving_average(ae_power_mV2, window_size=4)
        if smoothed.size == 0:
            continue

        final_ae_power = float(smoothed[-1])
        ae_d50_points.append((reagent, trial, float(d50), final_ae_power))

    for psd_file in tqdm(all_psd_files, desc="Matching data"):
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue

        reagent = path_parts[-3]
        trial = path_parts[-2]

        match = re.search(r'(grind\d+min)', os.path.basename(psd_file))
        if not match:
            continue

        grind_key = match.group(1)
        if grind_key != "grind25min":
            continue
        grind_match = re.search(r'grind(\d+)min', grind_key)
        if not grind_match:
            continue
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        ae_files = sorted(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))
        if not ae_files:
            continue

        ae_power_timeseries = [ae_cache.get(norm_path(f)) for f in ae_files]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]

        if len(ae_power_timeseries) < 4:
            continue

        # Convert to mV^2
        ae_power_mV2 = np.array(ae_power_timeseries, dtype=float) * 1e6

        if (reagent, trial) not in ae_series_by_trial:
            timestamps = [parse_timestamp_from_filename(f) for f in ae_files]
            if all(ts is not None for ts in timestamps):
                order = np.argsort(timestamps)
                times_sorted = [timestamps[i] for i in order]
                values_sorted = ae_power_mV2[order]
                t0 = times_sorted[0]
                time_min = np.array(
                    [(ts - t0).total_seconds() / 60.0 for ts in times_sorted],
                    dtype=float
                )
            else:
                values_sorted = ae_power_mV2
                time_min = np.arange(1, len(values_sorted) + 1, dtype=float)
            ae_series_by_trial[(reagent, trial)] = (time_min, values_sorted)

    if not ae_series_by_trial and not ae_d50_points:
        print("No matched data points found.")
        exit()

    # ------------------------------------------------------------
    # Variance outputs
    # ------------------------------------------------------------
    os.makedirs(VARIANCE_OUTPUT_DIR, exist_ok=True)
    markers = {'1st': 'o', '2nd': 'x', '3rd': '^'}
    colors = {'1st': 'black', '2nd': 'red', '3rd': 'blue'}

    if ae_series_by_trial:
        trial_avg_path = os.path.join(
            VARIANCE_OUTPUT_DIR,
            "ae_dataset_raw_trial_avg.csv"
        )
        trial_rows = []
        variance_mode = args.variance_mode
        for (reagent, trial), (time_min, values_sorted) in ae_series_by_trial.items():
            trend_corr = float('nan')
            resid_std = float('nan')
            signal_last = float(values_sorted[-1])
            if variance_mode == 'log_detrended':
                fit_result = log_trend_fit(time_min, values_sorted)
                if fit_result is not None:
                    _, _, _, trend_corr = fit_result
                residuals = log_detrended_residuals(values_sorted, time_min)
                if residuals.size >= 2:
                    resid_std = float(np.std(residuals, ddof=1))
            n_points = int(len(values_sorted))
            trial_rows.append({
                "reagent": reagent,
                "trial": trial,
                "log_resid_std": resid_std,
                "signal_last_mV2": signal_last,
                "log_trend_corr": trend_corr,
                "n_points": n_points
            })
        trial_avg_df = pd.DataFrame(trial_rows)
        if not trial_avg_df.empty:
            numeric_cols = trial_avg_df.select_dtypes(include=[np.number]).columns
            reagent_means = trial_avg_df.groupby("reagent")[numeric_cols].mean().reset_index()
            reagent_means["trial"] = "ave"
            reagent_means["reagent"] = reagent_means["reagent"].astype(str) + "_ave"
            trial_avg_df = pd.concat([trial_avg_df, reagent_means], ignore_index=True)
        trial_avg_df.to_csv(trial_avg_path, index=False)
        print(f"Saved trial-avg dataset: {trial_avg_path}")

        reagents = sorted({key[0] for key in ae_series_by_trial})
        for current_reagent in reagents:
            reagent_trials = sorted({key[1] for key in ae_series_by_trial if key[0] == current_reagent})

            plt.figure(figsize=(12, 8))
            for t in reagent_trials:
                time_min, values_sorted = ae_series_by_trial[(current_reagent, t)]
                plt.plot(
                    time_min,
                    values_sorted,
                    marker=markers.get(t, 'o'),
                    color=colors.get(t, 'black'),
                    label=t
                )
            plt.xlabel('Time (min)', fontsize=axes_label_size)
            plt.ylabel(r'Total spectral power ($\mathrm{mV}^2$)', fontsize=axes_label_size)
            plt.legend()
            plot_path = os.path.join(
                VARIANCE_OUTPUT_DIR,
                f"ae_power_timeseries_{current_reagent}.png"
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()

            if variance_mode == 'log_detrended':
                for t in reagent_trials:
                    time_min, values_sorted = ae_series_by_trial[(current_reagent, t)]
                    fit_result = log_trend_fit(time_min, values_sorted)
                    if fit_result is None:
                        continue
                    t_valid, y_valid, fit, trend_corr = fit_result

                    plt.figure(figsize=(12, 8))
                    plt.scatter(t_valid, y_valid, s=60, c='black', label='log(raw)')
                    plt.plot(t_valid, fit, 'r-', label='linear fit')
                    plt.xlabel('Time (min)', fontsize=axes_label_size)
                    plt.ylabel('log(Total spectral power)', fontsize=axes_label_size)
                    plt.title(f'log-trend fit r={trend_corr:.3f}')
                    plt.legend()
                    plot_path = os.path.join(
                        VARIANCE_OUTPUT_DIR,
                        f"log_trend_fit_{current_reagent}_{t}.png"
                    )
                    plt.savefig(plot_path, dpi=300)
                    plt.close()

                    values_valid = values_sorted[values_sorted > 0.0]
                    exp_fit = np.exp(fit)
                    plt.figure(figsize=(12, 8))
                    plt.scatter(t_valid, values_valid, s=60, c='black', label='raw')
                    plt.plot(t_valid, exp_fit, 'r-', label='exp fit (linear log)')
                    plt.xlabel('Time (min)', fontsize=axes_label_size)
                    plt.ylabel(r'Total spectral power ($\mathrm{mV}^2$)', fontsize=axes_label_size)
                    plt.title(f'exp-trend fit r={trend_corr:.3f}')
                    plt.legend()
                    plot_path = os.path.join(
                        VARIANCE_OUTPUT_DIR,
                        f"exp_trend_fit_{current_reagent}_{t}.png"
                    )
                    plt.savefig(plot_path, dpi=300)
                    plt.close()

    if ae_d50_points:
        ae_d50_array = np.array(ae_d50_points, dtype=object)
        for current_reagent in np.unique(ae_d50_array[:, 0]):
            mask = ae_d50_array[:, 0] == current_reagent
            reagent_data = ae_d50_array[mask]
            d50_vals = np.array(reagent_data[:, 2], dtype=float)
            ae_vals = np.array(reagent_data[:, 3], dtype=float)
            trial_labels = reagent_data[:, 1]

            if d50_vals.size < 2:
                continue

            slope, intercept, r2 = linear_fit_r2(d50_vals, ae_vals)
            x_plot = np.linspace(d50_vals.min() * 0.9, d50_vals.max() * 1.1, 200)
            y_plot = slope * x_plot + intercept

            plt.figure(figsize=(12, 8))
            for t in np.unique(trial_labels):
                t_mask = trial_labels == t
                plt.scatter(
                    d50_vals[t_mask],
                    ae_vals[t_mask],
                    s=70,
                    marker=markers.get(t, 'o'),
                    color=colors.get(t, 'black'),
                    label=t
                )
            plt.plot(x_plot, y_plot, 'k--', label='linear fit')
            plt.xlabel(r'D50 ($\mu$m)', fontsize=axes_label_size)
            plt.ylabel(r'AE power ($\mathrm{mV}^2$)', fontsize=axes_label_size)
            plt.title(rf'AE vs D50 ($r^2$={r2:.3f})')
            plt.legend()
            plot_path = os.path.join(
                VARIANCE_OUTPUT_DIR,
                f"ae_vs_d50_linear_{current_reagent}.png"
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()
