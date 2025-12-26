import pandas as pd
import numpy as np
import os
import glob
import re
import argparse
from fft_processing import calculate_fft_power

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
    
    # This will hold our final (D50, AE Power, Trial) tuples
    collected_data = []

    # Base paths
    ae_base_path = os.path.join('ae_data', EXPERIMENT)
    psd_base_path = os.path.join('powder_size_distribution_data', EXPERIMENT)

    # Build glob patterns for the outer loops
    reagent_pattern = TARGET_REAGENT if TARGET_REAGENT else '*'
    trial_pattern = TARGET_TRIAL if TARGET_TRIAL else '*'

    # 1. Iterate through PSD files, which are the ground truth for each session.
    psd_reagent_dirs = glob.glob(os.path.join(psd_base_path, reagent_pattern))
    for psd_reagent_dir in psd_reagent_dirs:
        reagent = os.path.basename(psd_reagent_dir)
        
        psd_trial_dirs = glob.glob(os.path.join(psd_reagent_dir, trial_pattern))
        for psd_trial_dir in psd_trial_dirs:
            trial = os.path.basename(psd_trial_dir)
            
            psd_files = glob.glob(os.path.join(psd_trial_dir, '*.csv'))

            for psd_file in psd_files:
                print(f"\nProcessing session file: {os.path.basename(psd_file)}")
                
                # 2. Get the ground truth D50 for this session
                d50 = get_d50(psd_file)
                if d50 is None:
                    print("  - D50 not found, skipping.")
                    continue

                # 3. Parse the session info (e.g., 'grind5min') from the PSD filename
                match = re.search(r'(grind\d+min)', os.path.basename(psd_file))
                if not match:
                    print("  - Could not parse grind duration from filename, skipping.")
                    continue
                grind_duration_key = match.group(1)

                # 4. Find all corresponding AE files for this specific session
                ae_session_path = os.path.join(ae_base_path, reagent, trial)
                ae_search_pattern = os.path.join(ae_session_path, f"*{grind_duration_key}*.csv")
                ae_files_for_session = glob.glob(ae_search_pattern)
                ae_files_for_session.sort() # Sort chronologically

                if not ae_files_for_session:
                    print(f"  - No matching AE files found for pattern: {ae_search_pattern}")
                    continue

                # 5. Create and process the time series for this session
                ae_power_timeseries = [calculate_fft_power(f) for f in ae_files_for_session]
                ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]

                if len(ae_power_timeseries) < 4:
                    print(f"  - Not enough valid AE data points ({len(ae_power_timeseries)}) for moving average, skipping.")
                    continue

                smoothed_ae_power = moving_average(np.array(ae_power_timeseries))
                
                # 6. The final point of the smoothed series is our feature
                final_ae_power = smoothed_ae_power[-1]

                # Append as (D50, AE Power, Trial) to match paper's axes
                collected_data.append((d50, final_ae_power, trial))
                print(f"  - Matched: D50={d50:.2f} -> Final Smoothed AE Power={final_ae_power:.2f}")

    # --- GPR Modeling ---
    if not collected_data:
        print("\nNo matched data points found. Cannot train GPR model.")
    else:
        # Unpack the collected data
        data_array = np.array(collected_data, dtype=object)
        X_data = np.array(data_array[:, 0], dtype=float).reshape(-1, 1) # D50
        y_data = np.array(data_array[:, 1], dtype=float)                # AE Power
        trial_labels = data_array[:, 2]                                 # Trial ('1st', '2nd', etc.)

        print(f"\n--- Dataset created with {len(X_data)} data points ---")

        # --- 2. Define and Train the GPR Model ---
        print("--- Training Gaussian Process Regressor ---")
        
        # Adjust kernel parameters based on the new X and Y scales
        kernel = 1.0 * RBF(length_scale=np.std(X_data), length_scale_bounds=(1e-2, 1e5)) \
            + WhiteKernel(noise_level=np.std(y_data)/2, noise_level_bounds=(1e-10, 1e5))

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0, normalize_y=True)
        gpr.fit(X_data, y_data)
        
        print("GPR model training complete.")
        print(f"Learned Kernel: {gpr.kernel_}")
        print(f"Log-Marginal-Likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta)}")

        # Calculate R-squared (R2) score
        r_squared = gpr.score(X_data, y_data)
        print(f"R-squared (R2) score: {r_squared:.4f}")

        # Predict with the model to get variance across the plot range
        X_plot = np.linspace(X_data.min() * 0.9, X_data.max() * 1.1, 500).reshape(-1, 1)
        y_mean, y_std = gpr.predict(X_plot, return_std=True)
        average_variance_of_prediction = np.mean(y_std**2)
        print(f"Average variance of predictions: {average_variance_of_prediction:.4f}")

        # --- 3. Visualize the Results ---
        print("--- Plotting results ---")

        # Create a directory for the results
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)

        # Define the base filename for outputs based on reagent and trial
        reagent_str = TARGET_REAGENT or 'all'
        trial_str = TARGET_TRIAL or 'all'
        output_base_filename = f"{reagent_str}_{trial_str}"

        # --- 4. Save Metrics to CSV ---
        metrics_df = pd.DataFrame({
            'r_squared': [r_squared],
            'average_variance': [average_variance_of_prediction]
        })
        csv_filename = os.path.join(output_dir, f'gpr_metrics_{output_base_filename}.csv')
        metrics_df.to_csv(csv_filename, index=False)
        print(f"Metrics saved to {csv_filename}")

        plt.figure(figsize=(8, 6))
        
        # Plot data points with different markers for each trial
        markers = {'1st': 'o', '2nd': 'x', '3rd': '^'}
        colors = {'1st': 'black', '2nd': 'red', '3rd': 'blue'}
        
        for trial_type in np.unique(trial_labels):
            mask = trial_labels == trial_type
            plt.scatter(X_data[mask], y_data[mask], 
                        c=colors.get(trial_type, 'black'), 
                        marker=markers.get(trial_type, 'o'), 
                        label=trial_type)

        # Plot the GPR prediction and confidence interval
        plt.plot(X_plot, y_mean, 'k-', label='Gaussian Process Regression (Mean)')
        plt.fill_between(X_plot.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                         alpha=0.2, color='blue', label='95% Confidence Interval')
        
        title_reagent = TARGET_REAGENT or 'All Reagents'
        plt.title(f'GPR Results for {title_reagent} ({EXPERIMENT})')
        plt.xlabel('D50 (μm)')
        plt.ylabel('Total Power Spectrum (a.u.)')
        plt.legend()
        plt.grid(True)
        plt.xlim(left=max(0, X_data.min() * 0.8))
        plt.ylim(bottom=0)

        plot_filename = os.path.join(output_dir, f'gpr_plot_{output_base_filename}.png')
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        
        plt.show()
