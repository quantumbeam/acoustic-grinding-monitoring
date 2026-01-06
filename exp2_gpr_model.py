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

# スタイル設定 (LaTeXなし環境用に 'no-latex' を追加)
# これによりTeXがインストールされていなくても動作します
plt.style.use(['science', 'ieee', 'no-latex'])

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
    
    for file_path in tqdm(required_files, desc="Checking AE files"):
        new_power = calculate_fft_power(file_path)
        
        if new_power is not None:
            relative_path = os.path.relpath(file_path)
            cached_power = ae_cache.get(relative_path)
            
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GPR model on AE and PSD data.')
    parser.add_argument('--reagent', type=str, default='all',
                        choices=['NaCl', 'Citricacid', 'Ajinomoto', 'all'],
                        help="Specify the reagent to process. Default is 'all'.")
    parser.add_argument('--trial', type=str, default='all',
                        choices=['1st', '2nd', '3rd', 'all'],
                        help="Specify the trial to process. Default is 'all'.")
    args = parser.parse_args()
    
    TARGET_REAGENT = None if args.reagent == 'all' else args.reagent
    TARGET_TRIAL = None if args.trial == 'all' else args.trial
    EXPERIMENT = 'exp2'
    CACHE_FILE = 'ae_power_cache.json'
    MODEL_DIR = 'results'

    # --- フォントサイズ設定 (GitHub実装参照) ---
    font_size = 24
    label_font_size = 32
    tick_font_size = 24
    legend_font_size = 18

    # フォントファミリを標準的なsans-serifに設定（LaTeXなしの場合の文字化け防止）
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': label_font_size,
        'xtick.labelsize': tick_font_size,
        'ytick.labelsize': tick_font_size,
        'legend.fontsize': legend_font_size,
        'font.family': 'sans-serif', 
        'mathtext.fontset': 'dejavusans' # 数式フォントも標準的なものへ
    })

    print("--- Scanning for required files ---")
    ae_base_path = os.path.join('ae_data', EXPERIMENT)
    psd_base_path = os.path.join('powder_size_distribution_data', EXPERIMENT)
    
    reagent_pattern = TARGET_REAGENT if TARGET_REAGENT else '*'
    trial_pattern = TARGET_TRIAL if TARGET_TRIAL else '*'
    
    all_psd_files = []
    psd_reagent_dirs = glob.glob(os.path.join(psd_base_path, reagent_pattern))
    for psd_reagent_dir in psd_reagent_dirs:
        psd_trial_dirs = glob.glob(os.path.join(psd_reagent_dir, trial_pattern))
        for psd_trial_dir in psd_trial_dirs:
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, '*.csv')))

    required_ae_files = set()
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
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

    ae_cache = update_ae_cache(CACHE_FILE, list(required_ae_files))

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

        ae_power_timeseries_v2 = [ae_cache.get(os.path.relpath(f)) for f in ae_files_for_session]
        ae_power_timeseries_v2 = [p for p in ae_power_timeseries_v2 if p is not None]

        if len(ae_power_timeseries_v2) < 4:
            continue
        
        ae_power_timeseries_mv2 = np.array(ae_power_timeseries_v2) * 1e6

        smoothed_ae_power = moving_average(ae_power_timeseries_mv2)
        final_ae_power = smoothed_ae_power[-1]

        collected_data.append((d50, final_ae_power, trial, reagent))

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

            X_data = np.array(reagent_data[:, 0], dtype=float).reshape(-1, 1)
            y_data = np.array(reagent_data[:, 1], dtype=float)
            trial_labels = reagent_data[:, 2]

            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
            gpr.fit(X_data, y_data)
            
            model_filename = f"gpr_model_{current_reagent}_{EXPERIMENT}.joblib"
            model_path = os.path.join(MODEL_DIR, model_filename)
            joblib.dump(gpr, model_path)
            
            r_squared = gpr.score(X_data, y_data)
            
            X_plot = np.linspace(X_data.min() * 0.9, X_data.max() * 1.1, 500).reshape(-1, 1)
            y_mean, y_std = gpr.predict(X_plot, return_std=True)
            average_variance = np.mean(y_std**2)
            
            all_metrics.append({
                'reagent': current_reagent,
                'r_squared': r_squared,
                'average_variance': average_variance
            })

            print(f"R-squared (R2): {r_squared:.4f}")

            # Plotting with SciencePlots style
            # (12, 8)に設定
            plt.figure(figsize=(12, 8))
            
            markers = {'1st': 'o', '2nd': 'x', '3rd': '^'}
            colors = {'1st': 'black', '2nd': 'red', '3rd': 'blue'}
            
            for trial_type in np.unique(trial_labels):
                mask = trial_labels == trial_type
                plt.scatter(X_data[mask], y_data[mask], c=colors.get(trial_type, 'black'), 
                            marker=markers.get(trial_type, 'o'), label=trial_type, s=100)

            plt.plot(X_plot, y_mean, 'k-', label='GPR Mean')
            plt.fill_between(X_plot.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                             alpha=0.2, color='blue', label='95% Confidence Interval')
            
            # 論文用途向けにμ(ミュー)の表示を調整
            plt.xlabel(r'$D_{50} (\mathrm{\mu m})$')
            plt.ylabel(r'Total Power Spectrum ($\mathrm{mV}^2$)')
            plt.legend()
            
            plt.xlim(left=max(0, X_data.min() * 0.8))
            plt.ylim(bottom=0)

            trial_str = TARGET_TRIAL or 'all'
            plot_filename = os.path.join(MODEL_DIR, f'gpr_plot_{current_reagent}_{trial_str}.png')
            plt.savefig(plot_filename, dpi=300)
            print(f"Plot saved to {plot_filename}")
            plt.close()

        if all_metrics:
            reagent_str = TARGET_REAGENT or 'all'
            trial_str = TARGET_TRIAL or 'all'
            csv_filename = os.path.join(MODEL_DIR, f'gpr_metrics_{reagent_str}_{trial_str}.csv')
            pd.DataFrame(all_metrics).to_csv(csv_filename, index=False)