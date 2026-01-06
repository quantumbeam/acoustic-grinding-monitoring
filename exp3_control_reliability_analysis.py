import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# --- 1. Raw Data ---
raw_data_map = {
    'NaCl': {
        'targets': [250, 200, 150],
        'measured': [[223.6, 252.1, 238.2], [196.1, 204.4, 189.5], [133.4, 123.8, 139.8]],
        'ae_power': [[1.3e-5, 1.2e-5, 1.25e-5], [8.8e-6, 9.1e-6, 8.9e-6], [5.8e-6, 5.6e-6, 5.9e-6]]
    },
    'CitricAcid': {
        'targets': [100, 50, 20],
        'measured': [[83.2, 100.2, 111.6], [46.1, 31.4, 37.3], [21.3, 14.1, 16.9]],
        'ae_power': [[2.5e-6, 2.7e-6, 2.4e-6], [1.1e-6, 1.0e-6, 1.2e-6], [4.5e-7, 4.8e-7, 4.4e-7]]
    },
    'Ajinomoto': {
        'targets': [200, 100, 50],
        'measured': [[179.6, 203.3, 189.0], [129.3, 124.9, 123.9], [53.1, 54.9, 50.1]],
        'ae_power': [[3.4e-5, 3.6e-5, 3.3e-5], [1.8e-5, 1.7e-5, 1.9e-5], [7.8e-6, 8.0e-6, 7.6e-6]]
    }
}

os.makedirs('results', exist_ok=True)
table_data_list = [] # List to store combined data for the LaTeX table

print("--- Starting Analysis for Table Generation ---")

for material, content in raw_data_map.items():
    # --- GPR Training ---
    X_train = np.array(content['ae_power']).flatten().reshape(-1, 1)
    y_train = np.array(content['measured']).flatten()
    
    # Kernel configuration matching GitHub/Paper implementation
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gpr.fit(X_train, y_train)
    
    # --- Evaluation per Target ---
    for i, target_val in enumerate(content['targets']):
        # 1. Measured Statistics
        measured_trials = np.array(content['measured'][i])
        mu_trial = np.mean(measured_trials)
        sigma_trial = np.std(measured_trials, ddof=1)
        
        # 2. GPR Prediction
        test_powers = np.array(content['ae_power'][i]).reshape(-1, 1)
        y_pred, y_sigma = gpr.predict(test_powers, return_std=True)
        mu_gpr = np.mean(y_pred)
        sigma_gpr = np.mean(y_sigma) # Average uncertainty for the trials
        
        # 3. Calculate Errors for Table
        estimation_error = mu_gpr - mu_trial
        total_deviation = target_val - mu_trial
        
        # 4. Format for CSV/LaTeX
        # Format: "Mean ± Sigma"
        gpr_str = f"{mu_gpr:.2f} ± {sigma_gpr:.2f}"
        measured_str = f"{mu_trial:.2f} ± {sigma_trial:.2f}"
        
        # Add '+' sign for positive errors to match table style
        est_err_str = f"{estimation_error:+.2f}"
        dev_err_str = f"{total_deviation:+.2f}"

        table_data_list.append({
            'Material': material,
            'Target_D50': target_val,
            'GPR_Prediction': gpr_str,
            'Measured_Mean': measured_str,
            'Estimation_Error': est_err_str,
            'Total_Deviation': dev_err_str,
            # Raw values for verification if needed
            'raw_mu_gpr': mu_gpr,
            'raw_mu_trial': mu_trial,
            'raw_est_error': estimation_error
        })

# --- Save Combined Table Data ---
df_table = pd.DataFrame(table_data_list)
output_csv = 'results/exp3_table_for_latex.csv'
df_table.to_csv(output_csv, index=False)

print(f"\n✅ Table Data Generated: {output_csv}")
print(df_table[['Material', 'Target_D50', 'GPR_Prediction', 'Measured_Mean', 'Estimation_Error', 'Total_Deviation']])