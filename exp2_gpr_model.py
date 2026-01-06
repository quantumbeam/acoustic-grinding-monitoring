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

def update_ae_cache(cache_file_path, required_files):
    """
    Loads the cache and verifies the values for required files.
    If a file is missing from cache OR the calculated value differs 
    from the cached value, the cache is updated.
    """
    print("--- Verifying and Updating AE Power Cache ---")
    
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r') as f:
                ae_cache = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load cache file '{cache_file_path}'. Rebuilding. Error: {e}")
            ae_cache = {}
    else:
        print("Cache file not found. A new one will be created.")
        ae_cache = {}

    updated_count = 0
    
    # 変更点: missing_filesだけでなく、required_files全てをループして値を検証する
    for file_path in tqdm(required_files, desc="Checking AE files"):
        # 現在のfft_processingロジックで再計算 (単位: V^2)
        new_power = calculate_fft_power(file_path)
        
        if new_power is not None:
            relative_path = os.path.relpath(file_path)
            cached_power = ae_cache.get(relative_path)
            
            # キャッシュがない、または値が異なる場合（浮動小数点の誤差許容）は更新
            if cached_power is None or not np.isclose(new_power, cached_power, rtol=1e-7):
                ae_cache[relative_path] = new_power
                updated_count += 1
    
    if updated_count > 0:
        print(f"Cache updated with {updated_count} new or corrected values.")
        try:
            with open(cache_file_path, 'w') as f:
                json.dump(ae_cache, f, indent=4)
            print(f"Successfully saved cache file: '{cache_file_path}'")
        except IOError as e:
            print(f"\nError saving updated cache file: {e}")
    else:
        print("Cache verified. All values match current calculation logic.")

    return ae_cache


def get_d50(file_path):
    """
    Parses a powder size distribution CSV to find the D50 value.
    """
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
    """Calculates the moving average of a 1D array."""
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Train a GPR model on AE and PSD data.')
    parser.add_argument('--reagent', type=str, default='all',
                        choices=['NaCl', 'Citricacid', 'Ajinomoto', 'all'],
                        help="Specify the reagent to process. Default is 'all'.")
    parser.add_argument('--trial', type=str, default='all',
                        choices=['1st', '2nd', '3rd', 'all'],
                        help="Specify the trial to process. Default is 'all'.")
    args = parser.parse_args()
    
    # --- Configuration from Arguments ---
    TARGET_REAGENT = None if args.reagent == 'all' else args.reagent
    TARGET_TRIAL = None if args.trial == 'all' else args.trial
    EXPERIMENT = 'exp2'
    CACHE_FILE = 'ae_power_cache.json'
    MODEL_DIR = 'results'

    # --- Step 1: Pre-scan for all required AE files based on filters ---
    print("--- Scanning for required files ---")
    ae_base_path = os.path.join('ae_data', EXPERIMENT)
    psd_base_path = os.path.join('powder_size_distribution_data', EXPERIMENT)
    
    reagent_pattern = TARGET_REAGENT if TARGET_REAGENT else '*'
    trial_pattern = TARGET_TRIAL if TARGET_TRIAL else '*'
    
    all_psd_files = []
    # ディレクトリ構成に合わせてglobパターンを調整
    psd_reagent_dirs = glob.glob(os.path.join(psd_base_path, reagent_pattern))
    for psd_reagent_dir in psd_reagent_dirs:
        psd_trial_dirs = glob.glob(os.path.join(psd_reagent_dir, trial_pattern))
        for psd_trial_dir in psd_trial_dirs:
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, '*.csv')))

    required_ae_files = set()
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        # パス構成依存（-3, -2など）
        if len(path_parts) >= 3:
            reagent = path_parts[-3]
            trial = path_parts[-2]
            match = re.search(r'(grind\d+min)', os.path.basename(psd_file))
            if match:
                grind_duration_key = match.group(1)
                ae_session_path = os.path.join(ae_base_path, reagent, trial)
                ae_search_pattern = os.path.join(ae_session_path, f"*{grind_duration_key}*.csv")
                required_ae_files.update(glob.glob(ae_search_pattern))
    
    print(f"Found {len(required_ae_files)} required AE files for this run.")

    # --- Step 2: Load or Update AE Power Cache ---
    # ここでキャッシュの検証・更新が行われます
    ae_cache = update_ae_cache(CACHE_FILE, list(required_ae_files))

    # --- Step 3: Data Collection using Cache ---
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
        grind_duration_key = match.group(1)

        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        ae_search_pattern = os.path.join(ae_session_path, f"*{grind_duration_key}*.csv")
        ae_files_for_session = sorted(glob.glob(ae_search_pattern))

        if not ae_files_for_session:
            continue

        # キャッシュから値 (V^2) を取得
        ae_power_timeseries_v2 = [ae_cache.get(os.path.relpath(f)) for f in ae_files_for_session]
        ae_power_timeseries_v2 = [p for p in ae_power_timeseries_v2 if p is not None]

        if len(ae_power_timeseries_v2) < 4:
            continue
        
        # 単位変換: V^2 -> mV^2 (x 10^6)
        ae_power_timeseries_mv2 = np.array(ae_power_timeseries_v2) * 1e6

        smoothed_ae_power = moving_average(ae_power_timeseries_mv2)
        final_ae_power = smoothed_ae_power[-1]

        collected_data.append((d50, final_ae_power, trial, reagent))

    # --- GPR Modeling ---
    if not collected_data:
        print("\nNo matched data points found. Cannot train GPR model.")
    else:
        print(f"\nSuccessfully collected {len(collected_data)} data points (Units: mV²).")
        data_array = np.array(collected_data, dtype=object)
        
        unique_reagents_in_data = np.unique(data_array[:, 3])
        
        all_metrics = []

        os.makedirs(MODEL_DIR, exist_ok=True)

        for current_reagent in unique_reagents_in_data:
            print(f"\n--- Processing GPR for Reagent: {current_reagent} ---")
            
            reagent_mask = data_array[:, 3] == current_reagent
            reagent_data = data_array[reagent_mask]

            X_data = np.array(reagent_data[:, 0], dtype=float).reshape(-1, 1) # D50
            y_data = np.array(reagent_data[:, 1], dtype=float)                # AE Power (mV^2)
            trial_labels = reagent_data[:, 2]

            print(f"Dataset created with {len(X_data)} data points for {current_reagent}")

            print("--- Training Gaussian Process Regressor ---")
            
            # カーネル定義: GitHub実装に合わせ、初期値1.0, バウンドなし
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
            
            gpr.fit(X_data, y_data)
            
            model_filename = f"gpr_model_{current_reagent}_{EXPERIMENT}.joblib"
            model_path = os.path.join(MODEL_DIR, model_filename)
            joblib.dump(gpr, model_path)
            print(f"Model saved to: {model_path}")
            
            r_squared = gpr.score(X_data, y_data)
            
            X_plot = np.linspace(X_data.min() * 0.9, X_data.max() * 1.1, 500).reshape(-1, 1)
            y_mean, y_std = gpr.predict(X_plot, return_std=True)
            average_variance_of_prediction = np.mean(y_std**2)
            
            all_metrics.append({
                'reagent': current_reagent,
                'r_squared': r_squared,
                'average_variance': average_variance_of_prediction
            })

            print(f"R-squared (R2) for {current_reagent}: {r_squared:.4f}")
            print(f"Average variance for {current_reagent}: {average_variance_of_prediction:.4f}")

            # Plotting
            plt.figure(figsize=(8, 6))
            markers = {'1st': 'o', '2nd': 'x', '3rd': '^'}
            colors = {'1st': 'black', '2nd': 'red', '3rd': 'blue'}
            
            for trial_type in np.unique(trial_labels):
                mask = trial_labels == trial_type
                plt.scatter(X_data[mask], y_data[mask], c=colors.get(trial_type, 'black'), 
                            marker=markers.get(trial_type, 'o'), label=trial_type)

            plt.plot(X_plot, y_mean, 'k-', label='GPR Mean')
            plt.fill_between(X_plot.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                             alpha=0.2, color='blue', label='95% Confidence Interval')
            
            # ラベルは mV^2
            plt.title(f'GPR Results for {current_reagent} ({EXPERIMENT})')
            plt.xlabel(r'$D_{50} (\rm{\mu m})$')
            plt.ylabel(r'Total Power Spectrum ($\rm{mV}^2$)')
            plt.legend()
            plt.grid(True)
            plt.xlim(left=max(0, X_data.min() * 0.8))
            plt.ylim(bottom=0)

            output_dir = 'results'
            os.makedirs(output_dir, exist_ok=True)
            
            trial_str = TARGET_TRIAL or 'all'
            plot_filename = os.path.join(output_dir, f'gpr_plot_{current_reagent}_{trial_str}.png')
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            plt.close()

        if all_metrics:
            output_dir = 'results'
            os.makedirs(output_dir, exist_ok=True)
            reagent_str = TARGET_REAGENT or 'all'
            trial_str = TARGET_TRIAL or 'all'
            
            if reagent_str == 'all':
                csv_filename = os.path.join(output_dir, f'gpr_metrics_by_reagent_{trial_str}.csv')
            else:
                csv_filename = os.path.join(output_dir, f'gpr_metrics_{reagent_str}_{trial_str}.csv')

            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(csv_filename, index=False)
            print(f"\nAll metrics saved to {csv_filename}")