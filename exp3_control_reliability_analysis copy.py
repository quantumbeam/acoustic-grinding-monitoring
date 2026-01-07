import pandas as pd
import numpy as np
import os
import joblib
import re
import glob
from fft_processing import calculate_fft_power

# ============================================================
# Helper Functions
# ============================================================
def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None

def natural_keys(text):
    """
    Sorts in human order (e.g. 1, 2, 10 instead of 1, 10, 2).
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def parse_timestamp_key(filename: str):
    """
    Robust sort key for filename like: 20251217_164537NaCl_grind_for_200um.csv
    """
    base = os.path.basename(filename)
    m = re.match(r'(\d{8})_(\d{6})', base)
    if m:
        return (m.group(1), m.group(2), base)
    return ("", "", base)

def safe_z(value, mu, sigma, eps=1e-12):
    """z = (value - mu) / sigma"""
    v = safe_float(value)
    m = safe_float(mu)
    s = safe_float(sigma)
    if v is None or m is None or s is None:
        return None
    if abs(s) < eps:
        return None
    return float((v - m) / s)

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

# ============================================================
# Configuration
# ============================================================
RESULTS_DIR = 'results'
EXPERIMENT = 'exp3'
PSD_BASE_PATH = os.path.join('powder_size_distribution_data', EXPERIMENT)
AE_BASE_PATH = os.path.join('ae_data', EXPERIMENT)
AE_SCALE_TO_MV2 = 1e6

# Models
# Method A uses Forward Model (Target -> AE)
MODEL_A_TAG = "particle2ae" 
# Method B uses Inverse Model (AE -> Estimated D50)
MODEL_B_TAG = "ae2particle" 

MODEL_NAME_MAP = {
    'NaCl': 'NaCl',
    'Citricacid': 'Citricacid',
    'Ajinomoto': 'Ajinomoto',
}

# Control Parameters
K_SIGMA_AE = 0.0  # Method A threshold parameter

os.makedirs(RESULTS_DIR, exist_ok=True)

print("--- Unified Evaluation: Method A (Threshold) & Method B (Estimation) ---")

# Get materials and sort them naturally
materials = [os.path.basename(d) for d in glob.glob(os.path.join(PSD_BASE_PATH, '*')) if os.path.isdir(d)]
materials.sort(key=natural_keys)

rows = []

for material in materials:
    model_key = MODEL_NAME_MAP.get(material, material)
    
    # 1. Load Forward Model for Method A (Target -> AE Threshold)
    model_a_path = os.path.join(RESULTS_DIR, f"gpr_model_{MODEL_A_TAG}_{model_key}_exp2.joblib")
    if not os.path.exists(model_a_path):
        print(f"Warning: Method A model not found for {material}")
        continue
    gpr_A = joblib.load(model_a_path)

    # 2. Load Inverse Model for Method B (AE -> D50 Estimation)
    model_b_path = os.path.join(RESULTS_DIR, f"gpr_model_{MODEL_B_TAG}_{model_key}_exp2.joblib")
    if not os.path.exists(model_b_path):
        print(f"Warning: Method B model not found for {material}")
        continue
    gpr_B = joblib.load(model_b_path)

    # Identify Targets
    target_files_sample = glob.glob(os.path.join(PSD_BASE_PATH, material, '1st', '*.csv'))
    targets = []
    for f in target_files_sample:
        m = re.search(r'_for_?(\d+)um', os.path.basename(f))
        if m:
            targets.append(int(m.group(1)))
    targets = list(set(targets))
    targets.sort()

    for target_val in targets:
        # --- Method A Preparation: Calculate AE Threshold ---
        x_target = np.array([[float(target_val)]], dtype=float)
        ae_mu, ae_std = gpr_A.predict(x_target, return_std=True)
        A_ae_th = float(ae_mu[0]) - K_SIGMA_AE * float(ae_std[0])

        for trial in ['1st', '2nd', '3rd']:
            # --- Common Data: Measured D50 (Ground Truth) ---
            psd_dir = os.path.join(PSD_BASE_PATH, material, trial)
            psd_candidates = glob.glob(os.path.join(psd_dir, '*.csv'))
            psd_file = None
            for f in psd_candidates:
                if re.search(f'_for_?{target_val}um', os.path.basename(f)):
                    psd_file = f
                    break
            
            measured_d50 = get_d50(psd_file) if psd_file else None
            measured_d50_f = safe_float(measured_d50)

            # --- Common Data: AE Signal (Stop Point) ---
            ae_dir = os.path.join(AE_BASE_PATH, material, trial)
            ae_candidates = glob.glob(os.path.join(ae_dir, '*.csv'))
            ae_files = [f for f in ae_candidates if re.search(f'_for_?{target_val}um', os.path.basename(f))]
            ae_files.sort(key=parse_timestamp_key)

            last_file = ae_files[-1] if len(ae_files) >= 1 else None
            second_last_file = ae_files[-2] if len(ae_files) >= 2 else None

            def process_ae(file_path):
                if file_path is None:
                    return None, None
                p = calculate_fft_power(file_path)
                if p is None or not np.isfinite(p):
                    return None, None
                return os.path.basename(file_path), float(p) * AE_SCALE_TO_MV2

            name_last, ae_last = process_ae(last_file)
            name_second, ae_second = process_ae(second_last_file)

            # --- Method A Evaluation (Threshold Control) ---
            # Did AE cross the threshold?
            A_is_expected_cross = False
            if ae_last is not None and ae_second is not None:
                if (ae_last < A_ae_th) and (ae_second >= A_ae_th):
                    A_is_expected_cross = True
            
            # Metric A: Total Deviation (Target - Measured)
            # "How far is the result from the target?"
            A_total_deviation = None
            if measured_d50_f is not None:
                A_total_deviation = float(target_val) - measured_d50_f


            # --- Method B Evaluation (Estimation Control) ---
            # What does the AE imply about D50?
            B_d50hat = None
            B_d50hat_sigma = None
            if ae_last is not None:
                y, s = gpr_B.predict(np.array([[ae_last]], dtype=float), return_std=True)
                B_d50hat = float(y[0])
                B_d50hat_sigma = float(s[0])
            
            # Metric B: Estimation Error (Predicted - Measured)
            # "Did the AE accurately reflect the particle size?"
            B_estimation_error = None
            if B_d50hat is not None and measured_d50_f is not None:
                B_estimation_error = B_d50hat - measured_d50_f

            
            # --- Store Result ---
            rows.append({
                "Material": material,
                "Trial": trial,
                "Target_D50": target_val,
                
                # Common
                "Common_Measured_D50": measured_d50_f,
                "Common_AE_Last_mV2": ae_last,
                
                # Method A Specifics
                "A_AE_Threshold": A_ae_th,
                "A_Is_ExpectedCross": A_is_expected_cross,
                "A_Total_Deviation": A_total_deviation,
                
                # Method B Specifics
                "B_Predicted_D50": B_d50hat,
                "B_Predicted_Sigma": B_d50hat_sigma,
                "B_Estimation_Error": B_estimation_error
            })

# Save Detailed Data
df = pd.DataFrame(rows)
out_detail = os.path.join(RESULTS_DIR, "exp3_evaluation_detail.csv")
df.to_csv(out_detail, index=False)
print(f"Saved detailed results to: {out_detail}")

if df.empty:
    print("No data processed.")
    exit()

# ============================================================
# Summary Aggregation (For Paper Table)
# ============================================================
summary_rows = []

for (mat, tgt), g in df.groupby(["Material", "Target_D50"]):
    # 1. Measured Stats (Common)
    mu_trial = g["Common_Measured_D50"].mean()
    sigma_trial = g["Common_Measured_D50"].std(ddof=1)
    
    # 2. Method B Stats (Prediction)
    mu_gpr = g["B_Predicted_D50"].mean()
    mean_sigma_gpr = g["B_Predicted_Sigma"].mean()
    mean_est_error = g["B_Estimation_Error"].mean()

    # 3. Method A Stats (Performance)
    mean_total_dev = g["A_Total_Deviation"].mean()
    
    # Format strings for Table
    meas_str = f"{mu_trial:.2f} ± {sigma_trial:.2f}" if pd.notnull(mu_trial) else "N/A"
    gpr_str = f"{mu_gpr:.2f} ± {mean_sigma_gpr:.2f}" if pd.notnull(mu_gpr) else "N/A"

    summary_rows.append({
        "Material": mat,
        "Target_D50": tgt,
        
        # Columns matching your LaTeX table request
        "B_GPR_Prediction": gpr_str,
        "Common_Measured_Mean": meas_str,
        "B_Estimation_Error": mean_est_error,
        "A_Total_Deviation": mean_total_dev,
        
        # Raw numeric values for plotting if needed
        "num_B_mu_GPR": mu_gpr,
        "num_B_sigma_GPR": mean_sigma_gpr,
        "num_Common_mu_trial": mu_trial,
        "num_Common_sigma_trial": sigma_trial
    })

df_summary = pd.DataFrame(summary_rows)
out_summary = os.path.join(RESULTS_DIR, "exp3_evaluation_summary_for_table.csv")
df_summary.to_csv(out_summary, index=False)

print("\n--- Summary Table Preview (Top 10) ---")
print(df_summary[["Material", "Target_D50", "B_GPR_Prediction", "Common_Measured_Mean", "B_Estimation_Error", "A_Total_Deviation"]].head(10).to_string(index=False))
print(f"\nSummary table saved to: {out_summary}")