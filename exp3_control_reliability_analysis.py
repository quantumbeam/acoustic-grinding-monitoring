import pandas as pd
import numpy as np
import os
import joblib
import re
import glob
from natsort import natsorted
from fft_processing import calculate_fft_power


# --- Copied from exp2_gpr_model.py ---
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

# --- Settings consistent with training script ---
RESULTS_DIR = 'results'
EXPERIMENT = 'exp3'
AE_SCALE_TO_MV2 = 1e6

PSD_BASE_PATH = os.path.join('powder_size_distribution_data', EXPERIMENT)
AE_BASE_PATH = os.path.join('ae_data', EXPERIMENT)

os.makedirs(RESULTS_DIR, exist_ok=True)
table_data_list = []

# Map raw material naming to model naming used in training script
MODEL_NAME_MAP = {
    'NaCl': 'NaCl',
    'Citricacid': 'Citricacid',
    'Ajinomoto': 'Ajinomoto',
}

print("--- Starting Analysis for Table Generation (joblib-loaded GPR) ---")

materials = natsorted([os.path.basename(d) for d in glob.glob(os.path.join(PSD_BASE_PATH, '*')) if os.path.isdir(d)])

for material in materials:
    print(f"Processing material: {material}")
    
    model_key = MODEL_NAME_MAP.get(material, material)
    model_path = os.path.join(RESULTS_DIR, f"gpr_model_{model_key}_exp2.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found for material '{material}'. Expected: {model_path}. Check EXPERIMENT setting and model naming."
        )

    # --- Load trained estimator: AE -> D50 ---
    gpr = joblib.load(model_path)
    
    # --- Dynamically find targets ---
    target_files_sample = glob.glob(os.path.join(PSD_BASE_PATH, material, '1st', '*.csv'))
    targets = []
    for f in target_files_sample:
        match = re.search(r'_for_?(\d+)um', os.path.basename(f))
        if match:
            targets.append(int(match.group(1)))
    
    targets = natsorted(list(set(targets)))
    
    for target_val in targets:
        print(f"  Processing target: {target_val} um")
        measured_trials = []
        ae_power_trials = []

        for trial in ['1st', '2nd', '3rd']:
            # 1. Find PSD file and get D50
            psd_dir = os.path.join(PSD_BASE_PATH, material, trial)
            all_psd_files = glob.glob(os.path.join(psd_dir, '*.csv'))
            psd_file = None
            for f in all_psd_files:
                if re.search(f'_for_?{target_val}um', os.path.basename(f)):
                    psd_file = f
                    break
            
            if not psd_file:
                print(f"    [Warning] No PSD file for {material} {trial} target {target_val}um")
                continue
            
            d50_value = get_d50(psd_file)
            if d50_value:
                measured_trials.append(d50_value)
            
            # 2. Find latest AE file and calculate power
            ae_dir = os.path.join(AE_BASE_PATH, material, trial)
            
            all_ae_files = glob.glob(os.path.join(ae_dir, '*.csv'))
            ae_files_for_target = []
            for f in all_ae_files:
                if re.search(f'_for_?{target_val}um', os.path.basename(f)):
                    ae_files_for_target.append(f)

            latest_ae_file = None
            latest_timestamp = ''
            
            for f in ae_files_for_target:
                match = re.search(r'(\d{8}_\d{6})', os.path.basename(f))
                if match:
                    timestamp = match.group(1)
                    if timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_ae_file = f
            
            if not latest_ae_file:
                print(f"    [Warning] No AE file for {material} {trial} target {target_val}um")
                continue
            
            ae_power = calculate_fft_power(latest_ae_file)
            if ae_power:
                ae_power_trials.append(ae_power)

        if not measured_trials or not ae_power_trials:
            print(f"    [Skipping] Incomplete data for target {target_val}um")
            continue

        # --- Evaluation per Target ---
        # 1) Measured statistics (laser diffraction D50)
        measured_trials = np.array(measured_trials, dtype=float)
        mu_trial = float(np.mean(measured_trials))
        sigma_trial = float(np.std(measured_trials, ddof=1))

        # 2) GPR prediction from AE
        test_powers_raw = np.array(ae_power_trials, dtype=float).reshape(-1, 1)
        test_powers_mV2 = test_powers_raw * AE_SCALE_TO_MV2

        y_pred, y_sigma = gpr.predict(test_powers_mV2, return_std=True)

        mu_gpr = float(np.mean(y_pred))
        sigma_gpr = float(np.mean(y_sigma))

        # 3) Errors
        estimation_error = mu_gpr - mu_trial
        total_deviation = target_val - mu_trial

        # 4) Formatting for CSV/LaTeX
        gpr_str = f"{mu_gpr:.2f} ± {sigma_gpr:.2f}"
        measured_str = f"{mu_trial:.2f} ± {sigma_trial:.2f}"
        est_err_str = f"{estimation_error:+.2f}"
        dev_err_str = f"{total_deviation:+.2f}"

        table_data_list.append({
            'Material': material,
            'Target_D50': target_val,
            'GPR_Prediction': gpr_str,
            'Measured_Mean': measured_str,
            'Estimation_Error': est_err_str,
            'Total_Deviation': dev_err_str,
            'raw_mu_gpr': mu_gpr,
            'raw_sigma_gpr_mean': sigma_gpr,
            'raw_mu_trial': mu_trial,
            'raw_sigma_trial': sigma_trial,
            'raw_est_error': estimation_error,
            'model_path': model_path
        })

# --- Save Combined Table Data ---
df_table = pd.DataFrame(table_data_list)
output_csv = os.path.join(RESULTS_DIR, 'exp3_table_for_latex.csv')
df_table.to_csv(output_csv, index=False)

print(f"\nTable Data Generated: {output_csv}")
if not df_table.empty:
    print(df_table[['Material', 'Target_D50', 'GPR_Prediction', 'Measured_Mean', 'Estimation_Error', 'Total_Deviation']])
else:
    print("No data was processed.")