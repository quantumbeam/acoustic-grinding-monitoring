import pandas as pd
import numpy as np
import os
import joblib
import re
import glob
from natsort import natsorted
from fft_processing import calculate_fft_power


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def parse_timestamp_key(filename: str):
    """
    Robust sort key for filename like:
      20251217_164537NaCl_grind_for_200um.csv
    """
    base = os.path.basename(filename)
    m = re.match(r'(\d{8})_(\d{6})', base)
    if m:
        return (m.group(1), m.group(2), base)
    return ("", "", base)


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


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
RESULTS_DIR = 'results'
EXPERIMENT = 'exp3'
PSD_BASE_PATH = os.path.join('powder_size_distribution_data', EXPERIMENT)
AE_BASE_PATH = os.path.join('ae_data', EXPERIMENT)

AE_SCALE_TO_MV2 = 1e6

FORWARD_DIRECTION_TAG = "particle2ae"
K_SIGMA = 0.0  # mean only (your current rule)

MODEL_NAME_MAP = {
    'NaCl': 'NaCl',
    'Citricacid': 'Citricacid',
    'Ajinomoto': 'Ajinomoto',
}

os.makedirs(RESULTS_DIR, exist_ok=True)

print("--- Check last-2 crossing vs predicted threshold (particle2ae) ---")

materials = natsorted([
    os.path.basename(d)
    for d in glob.glob(os.path.join(PSD_BASE_PATH, '*'))
    if os.path.isdir(d)
])

rows = []

for material in materials:
    model_key = MODEL_NAME_MAP.get(material, material)

    forward_model_path = os.path.join(
        RESULTS_DIR,
        f"gpr_model_{FORWARD_DIRECTION_TAG}_{model_key}_exp2.joblib"
    )
    if not os.path.exists(forward_model_path):
        raise FileNotFoundError(f"Forward model not found: {forward_model_path}")

    gpr_forward = joblib.load(forward_model_path)

    # Targets from naming in exp3 PSD
    target_files_sample = glob.glob(os.path.join(PSD_BASE_PATH, material, '1st', '*.csv'))
    targets = []
    for f in target_files_sample:
        m = re.search(r'_for_?(\d+)um', os.path.basename(f))
        if m:
            targets.append(int(m.group(1)))
    targets = natsorted(list(set(targets)))

    for target_val in targets:
        # Predict AE threshold from forward model (mV^2)
        x_target = np.array([[float(target_val)]], dtype=float)
        ae_mu, ae_std = gpr_forward.predict(x_target, return_std=True)
        ae_mu = float(ae_mu[0])
        ae_std = float(ae_std[0])
        ae_th = ae_mu - K_SIGMA * ae_std

        for trial in ['1st', '2nd', '3rd']:
            # PSD D50 (reference)
            psd_dir = os.path.join(PSD_BASE_PATH, material, trial)
            psd_candidates = glob.glob(os.path.join(psd_dir, '*.csv'))
            psd_file = None
            for f in psd_candidates:
                if re.search(f'_for_?{target_val}um', os.path.basename(f)):
                    psd_file = f
                    break
            measured_d50 = get_d50(psd_file) if psd_file else None

            # AE files for this target
            ae_dir = os.path.join(AE_BASE_PATH, material, trial)
            ae_candidates = glob.glob(os.path.join(ae_dir, '*.csv'))
            ae_files = [f for f in ae_candidates if re.search(f'_for_?{target_val}um', os.path.basename(f))]
            ae_files.sort(key=parse_timestamp_key)

            last_file = ae_files[-1] if len(ae_files) >= 1 else None
            second_last_file = ae_files[-2] if len(ae_files) >= 2 else None

            def compute_ae_mV2(path):
                if path is None:
                    return (None, None)
                p = calculate_fft_power(path)
                if p is None or not np.isfinite(p):
                    return (os.path.basename(path), None)
                return (os.path.basename(path), float(p) * AE_SCALE_TO_MV2)

            last_name, last_ae = compute_ae_mV2(last_file)
            second_name, second_ae = compute_ae_mV2(second_last_file)

            last_ae_f = safe_float(last_ae)
            second_ae_f = safe_float(second_ae)

            last_below = (last_ae_f is not None and last_ae_f < ae_th)
            second_below = (second_ae_f is not None and second_ae_f < ae_th)

            # The "expected" behavior for threshold-stop:
            # second_last >= th and last < th
            crossed_between_last2 = None
            if (second_ae_f is not None) and (last_ae_f is not None):
                crossed_between_last2 = (second_ae_f >= ae_th) and (last_ae_f < ae_th)

            # Case classification (useful for diagnostics)
            # - "ExpectedCross": second >= th, last < th
            # - "BothBelow": second < th, last < th (kept measuring after crossing / noise)
            # - "BothAbove": second >= th, last >= th (did not reach threshold or threshold mismatch)
            # - "Rebound": second < th, last >= th (unlikely; noise / ordering / labeling issue)
            case = "Missing"
            if (second_ae_f is not None) and (last_ae_f is not None):
                if (second_ae_f >= ae_th) and (last_ae_f < ae_th):
                    case = "ExpectedCross"
                elif (second_ae_f < ae_th) and (last_ae_f < ae_th):
                    case = "BothBelow"
                elif (second_ae_f >= ae_th) and (last_ae_f >= ae_th):
                    case = "BothAbove"
                elif (second_ae_f < ae_th) and (last_ae_f >= ae_th):
                    case = "Rebound"

            rows.append({
                "Material": material,
                "Trial": trial,
                "Target_D50": int(target_val),
                "Measured_D50": safe_float(measured_d50),

                "Forward_Model_Path": forward_model_path,
                "K_SIGMA": float(K_SIGMA),

                "AE_th_mV2": float(ae_th),
                "AE_th_mu_mV2": float(ae_mu),
                "AE_th_sigma_mV2": float(ae_std),

                "N_AE_files_for_target": int(len(ae_files)),

                "AE_second_last_file": second_name,
                "AE_second_last_mV2": second_ae_f,
                "AE_second_last_below": second_below if second_ae_f is not None else None,

                "AE_last_file": last_name,
                "AE_last_mV2": last_ae_f,
                "AE_last_below": last_below if last_ae_f is not None else None,

                "Crossed_between_last2": crossed_between_last2,
                "Case": case,
            })

df = pd.DataFrame(rows)

# Save detailed rows
out_detail = os.path.join(RESULTS_DIR, "exp3_threshold_stop_check_last2_detail.csv")
df.to_csv(out_detail, index=False)
print(f"\nSaved detail: {out_detail}")

# Aggregation: how often the expected pattern occurs
if not df.empty:
    # overall summary
    summary = (
        df.groupby(["Material", "Target_D50", "Case"])
          .size()
          .reset_index(name="count")
    )
    out_summary = os.path.join(RESULTS_DIR, "exp3_threshold_stop_check_last2_summary.csv")
    summary.to_csv(out_summary, index=False)
    print(f"Saved summary: {out_summary}")

    # Also: expected-cross rate per Material/Target
    rate_rows = []
    for (mat, tgt), g in df.groupby(["Material", "Target_D50"]):
        n = len(g)
        n_exp = int(np.sum(g["Case"] == "ExpectedCross"))
        rate_rows.append({"Material": mat, "Target_D50": tgt, "N": n, "N_ExpectedCross": n_exp, "Rate_ExpectedCross": n_exp / n if n else np.nan})
    rate_df = pd.DataFrame(rate_rows)
    out_rate = os.path.join(RESULTS_DIR, "exp3_threshold_stop_check_last2_rate.csv")
    rate_df.to_csv(out_rate, index=False)
    print(f"Saved rate: {out_rate}")

    print("\nTop of rate table:")
    print(rate_df.sort_values(["Material", "Target_D50"]).head(30))
else:
    print("No data was processed.")
