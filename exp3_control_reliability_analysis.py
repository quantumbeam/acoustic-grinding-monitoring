import pandas as pd
import numpy as np
import os
import joblib
import re
import glob
from fft_processing import calculate_fft_power
import matplotlib.pyplot as plt

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
RESULTS_DIR = 'results'
EXPERIMENT = 'exp3'
PSD_BASE_PATH = os.path.join('powder_size_distribution_data', EXPERIMENT)
AE_BASE_PATH = os.path.join('ae_data', EXPERIMENT)
AE_SCALE_TO_MV2 = 1e6

# Models
# Method P2AE uses Forward Model (Target -> AE)
MODEL_A_TAG = "particle2ae" 
# Method AE2P uses Inverse Model (AE -> Estimated D50)
MODEL_B_TAG = "ae2particle" 

MODEL_NAME_MAP = {
    'NaCl': 'NaCl',
    'Citricacid': 'Citricacid',
    'Ajinomoto': 'Ajinomoto',
}
PLOT_LABEL_MAP = {
    'Ajinomoto': 'MSG'
}

# Control Parameters
K_SIGMA_AE = 0.0  # Method A threshold parameter

os.makedirs(RESULTS_DIR, exist_ok=True)

print("--- Unified Evaluation: Method P2AE (Threshold) & Method AE2P (Estimation) ---")

# Get materials and sort them naturally
materials = [os.path.basename(d) for d in glob.glob(os.path.join(PSD_BASE_PATH, '*')) if os.path.isdir(d)]
materials.sort(key=natural_keys)

rows = []

for material in materials:
    model_key = MODEL_NAME_MAP.get(material, material)
    
    # 1. Load Forward Model for Method P2AE (Target -> AE Threshold)
    model_a_path = os.path.join(RESULTS_DIR, f"gpr_model_{MODEL_A_TAG}_{model_key}_exp2.joblib")
    if not os.path.exists(model_a_path):
        print(f"Warning: Method P2AE model not found for {material}")
        continue
    gpr_A = joblib.load(model_a_path)

    # 2. Load Inverse Model for Method AE2P (AE -> D50 Estimation)
    model_b_path = os.path.join(RESULTS_DIR, f"gpr_model_{MODEL_B_TAG}_{model_key}_exp2.joblib")
    if not os.path.exists(model_b_path):
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

            def process_ae(file_path):
                if file_path is None:
                    return None, None
                p = calculate_fft_power(file_path)
                if p is None or not np.isfinite(p):
                    return None, None
                return os.path.basename(file_path), float(p) * AE_SCALE_TO_MV2

            name_last, ae_last = process_ae(last_file)
            name_second, ae_second = process_ae(second_last_file)
            ae_series = []
            for f in ae_files:
                _, v = process_ae(f)
                if v is not None and np.isfinite(v):
                    ae_series.append(v)
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

if df.empty:
    print("No data processed.")
    exit()

# ------------------------------------------------------------
# Detail Plot: Measured - (Predicted + Sigma)
# ------------------------------------------------------------
plot_df = df.copy()
plot_df["AE2P_Abs_Error"] = (
    plot_df["Common_Measured_D50"] - plot_df["AE2P_Predicted_D50"]
).abs()
plot_df["AE2P_Upper_Error"] = (
    plot_df["Common_Measured_D50"]
    - (plot_df["AE2P_Predicted_D50"] + plot_df["AE2P_Predicted_Sigma"])
)
plot_df["AE2P_Est_Error_In_GPR_Range"] = (
    plot_df["AE2P_Abs_Error"]
    <= plot_df["AE2P_Predicted_Sigma"]
)
plot_df = plot_df.dropna(subset=["AE2P_Upper_Error", "Material", "Trial", "Target_D50"])
if not plot_df.empty:
    plot_df["label"] = plot_df.apply(
        lambda row: (
            f"{PLOT_LABEL_MAP.get(row['Material'], row['Material'])} "
            f"{row['Trial']} {row['Target_D50']}"
        ),
        axis=1,
    )
    export_cols = [
        "Material",
        "Trial",
        "Target_D50",
        "Common_Measured_D50",
        "AE2P_Predicted_D50",
        "AE2P_Predicted_Sigma",
        "AE2P_Abs_Error",
        "AE2P_Est_Error_In_GPR_Range",
        "AE2P_Upper_Error",
        "label",
    ]
    export_cols = [c for c in export_cols if c in plot_df.columns]
    discussion_dir = os.path.join(RESULTS_DIR, "discussion")
    os.makedirs(discussion_dir, exist_ok=True)
    export_path = os.path.join(discussion_dir, "exp3_detail_ae2p_upper_error_points.csv")
    plot_df[export_cols].to_csv(export_path, index=False)
    print(f"Saved detail plot data to: {export_path}")
    x_pos = np.arange(len(plot_df), dtype=float)

    plt.figure(figsize=(14, 8))
    mask_true = plot_df["AE2P_Est_Error_In_GPR_Range"] == True
    mask_false = plot_df["AE2P_Est_Error_In_GPR_Range"] == False
    plt.scatter(
        x_pos[mask_true],
        plot_df.loc[mask_true, "AE2P_Upper_Error"].values,
        s=80,
        c="black",
        marker="o",
        label="True"
    )
    plt.scatter(
        x_pos[mask_false],
        plot_df.loc[mask_false, "AE2P_Upper_Error"].values,
        s=80,
        c="black",
        marker="^",
        label="False"
    )
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    max_abs = float(np.nanmax(np.abs(plot_df["AE2P_Upper_Error"].values)))
    if np.isfinite(max_abs) and max_abs > 0.0:
        plt.ylim(-max_abs * 1.1, max_abs * 1.1)
    plt.xticks(x_pos, plot_df["label"].tolist(), rotation=45, ha="right")
    plt.xlabel("Material / Trial / Target_D50")
    plt.ylabel("Measured D50 - (Predicted D50 + Sigma)")
    plt.title("AE2P Upper Error by Experiment")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(discussion_dir, "exp3_detail_ae2p_upper_error.png")
    plot_pdf_path = os.path.join(discussion_dir, "exp3_detail_ae2p_upper_error.pdf")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_pdf_path)
    plt.close()
    print(f"Saved detail plot to: {plot_path}")

# ============================================================
# Summary Aggregation (For Paper Table)
# ============================================================
summary_rows = []

for (mat, tgt), g in df.groupby(["Material", "Target_D50"]):
    # 1. Measured Stats (Common)
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
