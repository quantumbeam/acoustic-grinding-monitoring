import pandas as pd
import numpy as np
import os
import glob
from fft_processing import calculate_fft_power

# For GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt

def get_d50(file_path):
    """
    Parses a powder size distribution CSV to find the D50 value.

    Args:
        file_path (str): Path to the powder size distribution CSV file.

    Returns:
        float: The D50 value, or None if not found.
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

def create_dataset(experiment='exp2'):
    """
    Creates a matched dataset of (AE Power, D50) for a given experiment.

    Args:
        experiment (str): The experiment to process ('exp2', 'exp3', etc.).

    Returns:
        tuple: A tuple containing two NumPy arrays (X_data, y_data) for AE power and D50.
    """
    ae_powers = []
    d50_values = []

    # Define base paths
    ae_base_path = os.path.join('ae_data', experiment)
    psd_base_path = os.path.join('powder_size_distribution_data', experiment)

    # Find all AE files for the experiment
    # Using glob to find all csv files recursively
    ae_files = glob.glob(os.path.join(ae_base_path, '**', '*.csv'), recursive=True)
    
    if not ae_files:
        print(f"Warning: No AE files found for experiment '{experiment}'")
        return np.array([]), np.array([])

    for ae_file in ae_files:
        # Calculate AE power
        # Note: We need to un-ignore ae_data if .gitignore is active
        power = calculate_fft_power(ae_file)
        if power is None:
            continue

        # Find the corresponding powder size distribution file
        # The relative path from the experiment directory should be the same
        relative_path = os.path.relpath(ae_file, ae_base_path)
        psd_file = os.path.join(psd_base_path, relative_path)

        if not os.path.exists(psd_file):
            print(f"Warning: Corresponding PSD file not found for {ae_file}")
            continue

        # Extract D50 value
        d50 = get_d50(psd_file)
        if d50 is None:
            continue
            
        # Add the matched pair to our dataset
        ae_powers.append(power)
        d50_values.append(d50)
        print(f"Processed {os.path.basename(ae_file)}: AE Power={power:.2f}, D50={d50:.2f}")

    return np.array(ae_powers).reshape(-1, 1), np.array(d50_values)


if __name__ == '__main__':
    # --- 1. Create the dataset for exp2 ---
    print("--- Creating dataset for exp2 ---")
    # This will fail if ae_data is in .gitignore, so we need to handle that.
    # For now, we assume it's accessible.
    X_data, y_data = create_dataset(experiment='exp2')

    if X_data.shape[0] == 0:
        print("\nDataset could not be created. Exiting.")
        print("This might be because 'ae_data' is in your .gitignore file.")
    else:
        print(f"\n--- Dataset created with {X_data.shape[0]} data points ---")

        # --- 2. Define and Train the GPR Model ---
        print("--- Training Gaussian Process Regressor ---")
        
        # A common kernel for GPR is a combination of RBF (for the main structure)
        # and WhiteKernel (to model noise).
        kernel = 1.0 * RBF(length_scale=1e5, length_scale_bounds=(1e2, 1e8)) \
            + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+2))

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)

        # Fit the model
        gpr.fit(X_data, y_data)
        print("GPR model training complete.")
        print(f"Learned Kernel: {gpr.kernel_}")

        # --- 3. Visualize the Results ---
        print("--- Plotting results ---")
        
        # Generate points for plotting the prediction
        X_plot = np.linspace(X_data.min(), X_data.max(), 500).reshape(-1, 1)
        
        # Get mean and standard deviation from the GPR model
        y_mean, y_std = gpr.predict(X_plot, return_std=True)

        plt.figure(figsize=(10, 6))
        # Plot the original data points
        plt.scatter(X_data, y_data, c='red', label='Observations')
        # Plot the GPR prediction
        plt.plot(X_plot, y_mean, 'b-', label='GPR Prediction')
        # Plot the confidence interval (95%)
        plt.fill_between(X_plot.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                         alpha=0.2, color='blue', label='95% Confidence Interval')
        
        plt.title('GPR Model of AE Power vs. D50 Particle Size (exp2)')
        plt.xlabel('Total AE Power (100kHz - 1MHz)')
        plt.ylabel('D50 Particle Size (μm)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot to a file
        plot_filename = 'gpr_exp2_results.png'
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        
        plt.show()
