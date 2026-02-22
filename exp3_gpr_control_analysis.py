import pandas as pd
import numpy as np
import os
import joblib
import re
import glob
import json
import sys
import subprocess
import argparse
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

def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

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
RESULTS_DIR = 'results/paper_plots'
MODEL_DIR_CANDIDATES = [
    RESULTS_DIR,
    os.path.join('model_comparison', 'gpr'),
]
EXP2_GPR_DIR = os.path.join('model_comparison', 'gpr')
EXP2_DATASET_PATH = os.path.join(EXP2_GPR_DIR, 'exp2_gpr_dataset_raw.csv')
EXP2_COMMON_AE_SUMMARY_PATH = os.path.join(EXP2_GPR_DIR, 'exp2_common_ae_last_summary.csv')
EXPERIMENT = 'exp3'
PSD_BASE_PATH = os.path.join('data/powder_size_distribution', EXPERIMENT)
AE_BASE_PATH = os.path.join('data/ae', EXPERIMENT)
AE_SCALE_TO_MV2 = 1e6
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, 'ae_power_cache.json')

# Models
# Method P2AE uses Forward Model (Target -> AE)
MODEL_A_TAG = "particle2ae" 
# Method AE2P uses Inverse Model (AE -> Estimated D50)
MODEL_B_TAG = "ae2particle" 

MODEL_NAME_MAP = {
    'NaCl': 'NaCl',
    'Citricacid': 'Citricacid',
    'MSG': 'MSG',
}

# Control Parameters
K_SIGMA_AE = 0.0  # Method A threshold parameter

os.makedirs(RESULTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser(
    description="Unified exp3 evaluation with optional auto-training of exp2 GPR models."
)
parser.add_argument(
    "--force-retrain-exp2-models",
    action="store_true",
    help="Force rerun model_comparison/exp2_gpr_model.py before exp3 evaluation.",
)
args = parser.parse_args()

print("--- Unified Evaluation: Method P2AE (Threshold) & Method AE2P (Estimation) ---")

def load_ae_power_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

def save_ae_power_cache(cache_path, cache_data):
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, sort_keys=True)
    except OSError:
        pass

AE_POWER_CACHE = load_ae_power_cache(CACHE_FILE)
CACHE_DIRTY = False

def get_cached_ae_power(file_path):
    global CACHE_DIRTY
    if file_path is None:
        return None
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        return None
    cached = AE_POWER_CACHE.get(file_path)
    if isinstance(cached, dict) and cached.get("mtime") == mtime:
        cached_power = cached.get("power")
        if cached_power is not None and np.isfinite(cached_power):
            return float(cached_power)
    p = calculate_fft_power(file_path)
    if p is None or not np.isfinite(p):
        return None
    AE_POWER_CACHE[file_path] = {"mtime": mtime, "power": float(p)}
    CACHE_DIRTY = True
    return float(p)


def find_model_path(model_tag: str, model_key: str, experiment: str = "exp2"):
    filename = f"gpr_model_{model_tag}_{model_key}_{experiment}.joblib"
    for base_dir in MODEL_DIR_CANDIDATES:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            return path
    return None


def ensure_exp2_gpr_models(required_model_keys, force_retrain=False):
    if force_retrain:
        trainer_script = os.path.join(SCRIPT_DIR, "model_comparison", "exp2_gpr_model.py")
        print("Force retrain enabled. Running exp2 GPR training script...")
        subprocess.run([sys.executable, trainer_script], check=True)

    missing = []
    for model_key in sorted(set(required_model_keys)):
        for model_tag in (MODEL_A_TAG, MODEL_B_TAG):
            if find_model_path(model_tag, model_key, experiment="exp2") is None:
                missing.append((model_tag, model_key))

    if not missing:
        return

    trainer_script = os.path.join(SCRIPT_DIR, "model_comparison", "exp2_gpr_model.py")
    print("Missing exp2 GPR models detected. Training automatically...")
    subprocess.run([sys.executable, trainer_script], check=True)

    unresolved = []
    for model_tag, model_key in missing:
        if find_model_path(model_tag, model_key, experiment="exp2") is None:
            unresolved.append((model_tag, model_key))
    if unresolved:
        unresolved_str = ", ".join(f"{tag}:{key}" for tag, key in unresolved)
        raise FileNotFoundError(f"Failed to prepare required exp2 GPR models: {unresolved_str}")


def refresh_exp2_common_ae_last_summary():
    os.makedirs(EXP2_GPR_DIR, exist_ok=True)

    need_trainer = False
    if not os.path.exists(EXP2_DATASET_PATH):
        need_trainer = True
    else:
        try:
            exp2_df_preview = pd.read_csv(EXP2_DATASET_PATH, nrows=5)
            if "Common_AE_Last_mV2" not in exp2_df_preview.columns:
                need_trainer = True
        except Exception:
            need_trainer = True

    if need_trainer:
        trainer_script = os.path.join(SCRIPT_DIR, "model_comparison", "exp2_gpr_model.py")
        print("Preparing exp2 dataset/summary CSV via exp2 GPR training script...")
        subprocess.run([sys.executable, trainer_script], check=True)

    exp2_df = pd.read_csv(EXP2_DATASET_PATH)
    if "Common_AE_Last_mV2" not in exp2_df.columns:
        raise KeyError(f"Missing column 'Common_AE_Last_mV2' in {EXP2_DATASET_PATH}")
    exp2_df["Common_AE_Last_mV2"] = pd.to_numeric(exp2_df["Common_AE_Last_mV2"], errors="coerce")
    exp2_df["grind_min"] = pd.to_numeric(exp2_df["grind_min"], errors="coerce")

    exp2_summary_df = (
        exp2_df.groupby(["reagent", "grind_min"], dropna=False)["Common_AE_Last_mV2"]
        .mean()
        .round(2)
        .reset_index(name="num_Common_AE_Last_mV2_mu_trial")
        .sort_values(["reagent", "grind_min"], kind="stable")
    )
    exp2_summary_df.to_csv(EXP2_COMMON_AE_SUMMARY_PATH, index=False)
    print(f"Saved exp2 common AE-last summary: {EXP2_COMMON_AE_SUMMARY_PATH}")

# Get materials and sort them naturally
materials = [os.path.basename(d) for d in glob.glob(os.path.join(PSD_BASE_PATH, '*')) if os.path.isdir(d)]
materials.sort(key=natural_keys)
materials = [m for m in materials if not m.endswith("_gpr")]

required_model_keys = [MODEL_NAME_MAP.get(m, m) for m in materials]
ensure_exp2_gpr_models(required_model_keys, force_retrain=args.force_retrain_exp2_models)
refresh_exp2_common_ae_last_summary()

rows = []

for material in materials:
    model_key = MODEL_NAME_MAP.get(material, material)
    
    # 1. Load Forward Model for Method P2AE (Target -> AE Threshold)
    model_a_path = find_model_path(MODEL_A_TAG, model_key, experiment="exp2")
    if model_a_path is None:
        print(f"Warning: Method P2AE model not found for {material}")
        continue
    gpr_A = joblib.load(model_a_path)

    # 2. Load Inverse Model for Method AE2P (AE -> D50 Estimation)
    model_b_path = find_model_path(MODEL_B_TAG, model_key, experiment="exp2")
    if model_b_path is None:
        print(f"Warning: Method AE2P model not found for {material}")
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
        # --- Method P2AE Preparation: Calculate AE Threshold ---
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

            ae_values = {}
            ae_series = []
            for f in ae_files:
                p = get_cached_ae_power(f)
                if p is None or not np.isfinite(p):
                    continue
                v = float(p) * AE_SCALE_TO_MV2
                ae_values[f] = v
                ae_series.append(v)
            name_last = os.path.basename(last_file) if last_file else None
            name_second = os.path.basename(second_last_file) if second_last_file else None
            ae_last = ae_values.get(last_file)
            ae_second = ae_values.get(second_last_file)
            ae_smoothed_last = None
            if len(ae_series) >= 4:
                smoothed = moving_average(np.array(ae_series, dtype=float), window_size=4)
                if smoothed.size > 0 and np.isfinite(smoothed[-1]):
                    ae_smoothed_last = float(smoothed[-1])

            # --- Method P2AE Evaluation (Threshold Control) ---
            # Did AE cross the threshold?
            A_is_expected_cross = False
            if ae_last is not None and ae_second is not None:
                if (ae_last < A_ae_th) and (ae_second >= A_ae_th):
                    A_is_expected_cross = True
            
            # Metric A: Total Deviation (Target - Measured)
            # "How far is the result from the target?"
            A_total_deviation = None
            A_total_deviation_percent = None
            if measured_d50_f is not None:
                A_total_deviation = float(target_val) - measured_d50_f
                if target_val != 0:
                    A_total_deviation_percent = (A_total_deviation / float(target_val)) * 100.0


            # --- Method AE2P Evaluation (Estimation Control) ---
            # What does the AE imply about D50?
            B_d50hat = None
            B_d50hat_sigma = None
            ae_for_estimation = ae_smoothed_last if ae_smoothed_last is not None else ae_last
            if ae_for_estimation is not None:
                y, s = gpr_B.predict(np.array([[ae_for_estimation]], dtype=float), return_std=True)
                B_d50hat = float(y[0])
                B_d50hat_sigma = float(s[0])
            
            # Metric B: Estimation Error (Predicted - Measured)
            # "Did the AE accurately reflect the particle size?"
            B_estimation_error = None
            B_estimation_error_percent = None
            B_estimation_error_in_range = None
            if B_d50hat is not None and measured_d50_f is not None:
                B_estimation_error =  measured_d50_f - B_d50hat
                if measured_d50_f != 0:
                    B_estimation_error_percent = (B_estimation_error / measured_d50_f) * 100.0
                if B_d50hat_sigma is not None:
                    B_estimation_error_in_range = abs(B_estimation_error) <= B_d50hat_sigma

            
            # --- Store Result ---
            rows.append({
                "Material": material,
                "Trial": trial,
                "Target_D50": target_val,
                
                # Common
                "Common_Measured_D50": measured_d50_f,
                "Common_AE_Last_mV2": ae_last,
                
                # Method P2AE Specifics
                "P2AE_AE_Threshold": A_ae_th,
                "P2AE_Is_ExpectedCross": A_is_expected_cross,
                "P2AE_Total_Deviation": A_total_deviation,
                "P2AE_Total_Deviation_Percent": A_total_deviation_percent,
                
                # Method AE2P Specifics
                "AE2P_Predicted_D50": B_d50hat,
                "AE2P_Predicted_Sigma": B_d50hat_sigma,
                "AE2P_Estimation_Error": B_estimation_error,
                "AE2P_Estimation_Error_Percent": B_estimation_error_percent,
                "AE2P_Est_Error_In_GPR_Range": B_estimation_error_in_range
            })

# Save Detailed Data
df = pd.DataFrame(rows)
summary_cols = [
    "Material",
    "Target_D50",
    "Common_Measured_Mean",
    "P2AE_Total_Deviation",
    "P2AE_Total_Deviation_Percent",
    "AE2P_GPR_Prediction",
    "AE2P_Estimation_Error",
    "AE2P_Estimation_Error_Percent",
    "AE2P_Est_Error_In_GPR_Range",
    "num_AE2P_mu_GPR",
    "num_AE2P_sigma_GPR",
    "num_Common_AE_Last_mV2_mu_trial",
    "num_Common_mu_trial",
    "num_Common_sigma_trial",
]
if not df.empty:
    df["Common_Measured_Mean"] = df["Common_Measured_D50"]
    df["AE2P_GPR_Prediction"] = df.apply(
        lambda row: (
            f"{row['AE2P_Predicted_D50']:.2f} ± {row['AE2P_Predicted_Sigma']:.2f}"
            if pd.notnull(row["AE2P_Predicted_D50"]) and pd.notnull(row["AE2P_Predicted_Sigma"])
            else "N/A"
        ),
        axis=1,
    )
    df["num_AE2P_mu_GPR"] = df["AE2P_Predicted_D50"]
    df["num_AE2P_sigma_GPR"] = df["AE2P_Predicted_Sigma"]
    df["num_Common_mu_trial"] = df["Common_Measured_D50"]
    df["num_Common_sigma_trial"] = np.nan

detail_cols = [
    "Material",
    "Trial",
    "Target_D50",
]
detail_cols = [c for c in detail_cols if c in df.columns]
ordered_cols = detail_cols + [c for c in summary_cols if c in df.columns]
seen = set()
ordered_cols = [c for c in ordered_cols if not (c in seen or seen.add(c))]
df = df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]
out_detail = os.path.join(RESULTS_DIR, "exp3_evaluation_detail.csv")
df.to_csv(out_detail, index=False)
print(f"Saved detailed results to: {out_detail}")

if CACHE_DIRTY:
    save_ae_power_cache(CACHE_FILE, AE_POWER_CACHE)

if df.empty:
    print("No data processed.")
    exit()

# ============================================================
# Summary Aggregation (For Paper Table)
# ============================================================
summary_rows = []

for (mat, tgt), g in df.groupby(["Material", "Target_D50"]):
    # 1. Measured Stats (Common)
    mu_ae_last = g["Common_AE_Last_mV2"].mean()
    mu_trial = g["Common_Measured_D50"].mean()
    sigma_trial = g["Common_Measured_D50"].std(ddof=1)
    
    # 2. Method AE2P Stats (Prediction)
    mu_gpr = g["AE2P_Predicted_D50"].mean()
    mean_sigma_gpr = g["AE2P_Predicted_Sigma"].mean()
    mean_est_error = g["AE2P_Estimation_Error"].mean()
    mean_est_error_pct = g["AE2P_Estimation_Error_Percent"].mean()
    ae2p_in_gpr_range = None
    if pd.notnull(mean_est_error) and pd.notnull(mean_sigma_gpr):
        ae2p_in_gpr_range = abs(mean_est_error) <= mean_sigma_gpr

    # 3. Method P2AE Stats (Performance)
    mean_total_dev = g["P2AE_Total_Deviation"].mean()
    mean_total_dev_pct = g["P2AE_Total_Deviation_Percent"].mean()
    
    # Format strings for Table
    meas_str = f"{mu_trial:.2f} ± {sigma_trial:.2f}" if pd.notnull(mu_trial) else "N/A"
    gpr_str = f"{mu_gpr:.2f} ± {mean_sigma_gpr:.2f}" if pd.notnull(mu_gpr) else "N/A"

    summary_rows.append({
        "Material": mat,
        "Target_D50": tgt,
        
        # Columns matching your LaTeX table request
        "AE2P_GPR_Prediction": gpr_str,
        "Common_Measured_Mean": meas_str,
        "AE2P_Estimation_Error": mean_est_error,
        "AE2P_Estimation_Error_Percent": mean_est_error_pct,
        "AE2P_Est_Error_In_GPR_Range": ae2p_in_gpr_range,
        "P2AE_Total_Deviation": mean_total_dev,
        "P2AE_Total_Deviation_Percent": mean_total_dev_pct,
        
        # Raw numeric values for plotting if needed
        "num_AE2P_mu_GPR": mu_gpr,
        "num_AE2P_sigma_GPR": mean_sigma_gpr,
        "num_Common_AE_Last_mV2_mu_trial": round(float(mu_ae_last), 2) if pd.notnull(mu_ae_last) else np.nan,
        "num_Common_mu_trial": mu_trial,
        "num_Common_sigma_trial": sigma_trial
    })

df_summary = pd.DataFrame(summary_rows)
summary_cols = [c for c in summary_cols if c in df_summary.columns]
df_summary = df_summary[summary_cols + [c for c in df_summary.columns if c not in summary_cols]]
out_summary = os.path.join(RESULTS_DIR, "exp3_evaluation_summary_for_table.csv")
df_summary.to_csv(out_summary, index=False)

print("\n--- Summary Table Preview (Top 10) ---")
print(df_summary[["Material", "Target_D50", "AE2P_GPR_Prediction", "Common_Measured_Mean", "AE2P_Estimation_Error", "AE2P_Estimation_Error_Percent", "AE2P_Est_Error_In_GPR_Range", "P2AE_Total_Deviation"]].head(10).to_string(index=False))
print(f"\nSummary table saved to: {out_summary}")
