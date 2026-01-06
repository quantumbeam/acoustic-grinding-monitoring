import pandas as pd
import numpy as np
import os
import glob
import re
import argparse
import json
from tqdm import tqdm
import joblib
from fft_processing import calculate_fft_power

# For GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import matplotlib.pyplot as plt
import scienceplots

# Style setting (LaTeX-less environment)
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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GPR model on AE and PSD data.')
    parser.add_argument('--reagent', type=str, default='all',
                        choices=['NaCl', 'Citricacid', 'Ajinomoto', 'all'])
    parser.add_argument('--trial', type=str, default='all',
                        choices=['1st', '2nd', '3rd', 'all'])
    args = parser.parse_args()

    TARGET_REAGENT = None if args.reagent == 'all' else args.reagent
    TARGET_TRIAL = None if args.trial == 'all' else args.trial
    EXPERIMENT = 'exp2'

    # Cache file fixed to script location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_FILE = os.path.join(SCRIPT_DIR, 'ae_power_cache.json')

    MODEL_DIR = 'results'

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

    # ------------------------------------------------------------
    # Scan files
    # ------------------------------------------------------------
    print("--- Scanning for required files ---")

    ae_base_path = os.path.join('ae_data', EXPERIMENT)
    psd_base_path = os.path.join('powder_size_distribution_data', EXPERIMENT)

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
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        pattern = os.path.join(ae_session_path, f"*{grind_key}*.csv")
        required_ae_files.update(glob.glob(pattern))

    print(f"Found {len(required_ae_files)} required AE files.")
    ae_cache = update_ae_cache(CACHE_FILE, list(required_ae_files))

    # ------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------
    print(f"--- Creating dataset for {EXPERIMENT} ---")
    collected_data = []

    for psd_file in tqdm(all_psd_files, desc="Matching data"):
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
        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        ae_files = sorted(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))
        if not ae_files:
            continue

        ae_power_timeseries = [
            ae_cache.get(norm_path(f)) for f in ae_files
        ]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]

        if len(ae_power_timeseries) < 4:
            continue

        ae_power_mV2 = np.array(ae_power_timeseries, dtype=float) * 1e6
        smoothed = moving_average(ae_power_mV2, window_size=4)
        if smoothed.size == 0:
            continue

        final_ae_power = float(smoothed[-1])
        collected_data.append((float(d50), final_ae_power, trial, reagent))

    if not collected_data:
        print("No matched data points found.")
        exit()

    print(f"Collected {len(collected_data)} data points.")
    data_array = np.array(collected_data, dtype=object)

    # ------------------------------------------------------------
    # GPR
    # ------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_metrics = []

    for current_reagent in np.unique(data_array[:, 3]):
        print(f"\n--- GPR for Reagent: {current_reagent} ---")

        mask = data_array[:, 3] == current_reagent
        reagent_data = data_array[mask]

        X_data = np.array(reagent_data[:, 1], dtype=float).reshape(-1, 1)
        y_data = np.array(reagent_data[:, 0], dtype=float)
        trial_labels = reagent_data[:, 2]

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            n_restarts_optimizer=10,
            normalize_y=True
        )
        gpr.fit(X_data, y_data)

        model_path = os.path.join(MODEL_DIR, f"gpr_model_{current_reagent}_{EXPERIMENT}.joblib")
        joblib.dump(gpr, model_path)

        r2 = gpr.score(X_data, y_data)
        print(f"R2: {r2:.4f}")
        print(f"Kernel: {gpr.kernel_}")

        X_plot = np.linspace(X_data.min()*0.9, X_data.max()*1.1, 500).reshape(-1, 1)
        y_mean, y_std = gpr.predict(X_plot, return_std=True)

        all_metrics.append({
            "reagent": current_reagent,
            "r_squared": float(r2),
            "average_variance": float(np.mean(y_std**2)),
            "kernel_optimized": str(gpr.kernel_)
        })

        plt.figure(figsize=(12, 8))
        markers = {'1st': 'o', '2nd': 'x', '3rd': '^'}
        colors = {'1st': 'black', '2nd': 'red', '3rd': 'blue'}

        for t in np.unique(trial_labels):
            m = trial_labels == t
            plt.scatter(X_data[m], y_data[m],
                        marker=markers.get(t, 'o'),
                        c=colors.get(t, 'black'),
                        s=100, label=t)

        plt.plot(X_plot, y_mean, 'k-')
        plt.fill_between(
            X_plot.ravel(),
            y_mean - 1.96*y_std,
            y_mean + 1.96*y_std,
            alpha=0.2
        )

        plt.xlabel(r'Total Power Spectrum ($\mathrm{mV}^2$)')
        plt.ylabel(r'$D_{50}~(\mathrm{\mu m})$')
        plt.legend()

        plot_path = os.path.join(MODEL_DIR, f"gpr_plot_{current_reagent}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(MODEL_DIR, "gpr_metrics.csv"), index=False
    )
