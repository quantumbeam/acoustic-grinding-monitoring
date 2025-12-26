import pandas as pd
import numpy as np
import os
import glob
import re
import argparse
from fft_processing import calculate_fft_power
from tqdm import tqdm

# For GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt

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

    # --- Data Collection ---
    print(f"--- Creating dataset for {EXPERIMENT} ---")
    if TARGET_REAGENT or TARGET_TRIAL:
        print(f"Filtering for Reagent: {TARGET_REAGENT or 'All'}, Trial: {TARGET_TRIAL or 'All'}")
    
    collected_data = []

    # Base paths
    ae_base_path = os.path.join('ae_data', EXPERIMENT)
    psd_base_path = os.path.join('powder_size_distribution_data', EXPERIMENT)

    # First, collect all files to be processed to use with tqdm
    all_psd_files = []
    reagent_pattern = TARGET_REAGENT if TARGET_REAGENT else '*'
    trial_pattern = TARGET_TRIAL if TARGET_TRIAL else '*'
    
    psd_reagent_dirs = glob.glob(os.path.join(psd_base_path, reagent_pattern))
    for psd_reagent_dir in psd_reagent_dirs:
        psd_trial_dirs = glob.glob(os.path.join(psd_reagent_dir, trial_pattern))
        for psd_trial_dir in psd_trial_dirs:
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, '*.csv')))

    print(f"Found {len(all_psd_files)} total files to process...")

    # Process files with a progress bar
    for psd_file in tqdm(all_psd_files, desc="Matching data"):
        # Parse reagent and trial from the file path
        path_parts = psd_file.split(os.sep)
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
        ae_files_for_session = glob.glob(ae_search_pattern)
        ae_files_for_session.sort() 

        if not ae_files_for_session:
            continue

        ae_power_timeseries = [calculate_fft_power(f) for f in ae_files_for_session]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]

        if len(ae_power_timeseries) < 4:
            continue

        smoothed_ae_power = moving_average(np.array(ae_power_timeseries))
        final_ae_power = smoothed_ae_power[-1]

        collected_data.append((d50, final_ae_power, trial, reagent))

    # --- GPR Modeling ---
    if not collected_data:
        print("\nNo matched data points found. Cannot train GPR model.")
    else:
        print(f"\nSuccessfully collected {len(collected_data)} data points.")
        data_array = np.array(collected_data, dtype=object)
        
        # Determine which reagents to process based on collected data
        unique_reagents_in_data = np.unique(data_array[:, 3])
        
        all_metrics = []

        for current_reagent in unique_reagents_in_data:
            print(f"\n--- Processing GPR for Reagent: {current_reagent} ---")
            
            # Filter data for the current reagent
            reagent_mask = data_array[:, 3] == current_reagent
            reagent_data = data_array[reagent_mask]

            X_data = np.array(reagent_data[:, 0], dtype=float).reshape(-1, 1)
            y_data = np.array(reagent_data[:, 1], dtype=float)
            trial_labels = reagent_data[:, 2]

            print(f"Dataset created with {len(X_data)} data points for {current_reagent}")

            # --- Define and Train the GPR Model ---
            print("--- Training Gaussian Process Regressor ---")
            kernel = 1.0 * RBF(length_scale=np.std(X_data), length_scale_bounds=(1e-2, 1e5)) \
                + WhiteKernel(noise_level=np.std(y_data)/2, noise_level_bounds=(1e-10, 1e5))
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0, normalize_y=True)
            gpr.fit(X_data, y_data)
            
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

            # --- Visualize the Results ---
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
            
            plt.title(f'GPR Results for {current_reagent} ({EXPERIMENT})')
            plt.xlabel('D50 (μm)')
            plt.ylabel('Total Power Spectrum (a.u.)')
            plt.legend()
            plt.grid(True)
            plt.xlim(left=max(0, X_data.min() * 0.8))
            plt.ylim(bottom=0)

            # --- Save Artifacts ---
            output_dir = 'results'
            os.makedirs(output_dir, exist_ok=True)
            
            trial_str = TARGET_TRIAL or 'all'
            plot_filename = os.path.join(output_dir, f'gpr_plot_{current_reagent}_{trial_str}.png')
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            plt.close() # Close the figure to free memory

        # --- Save All Metrics to a Single CSV ---
        if all_metrics:
            output_dir = 'results'
            os.makedirs(output_dir, exist_ok=True)
            reagent_str = TARGET_REAGENT or 'all'
            trial_str = TARGET_TRIAL or 'all'
            
            # Create a filename that reflects the content
            if reagent_str == 'all':
                csv_filename = os.path.join(output_dir, f'gpr_metrics_by_reagent_{trial_str}.csv')
            else:
                csv_filename = os.path.join(output_dir, f'gpr_metrics_{reagent_str}_{trial_str}.csv')

            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(csv_filename, index=False)
            print(f"\nAll metrics saved to {csv_filename}")
