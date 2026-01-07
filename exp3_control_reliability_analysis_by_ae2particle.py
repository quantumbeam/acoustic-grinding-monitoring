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


def safe_z(value, ref, sigma, eps=1e-12):
    """z=(value-ref)/sigma; None if not computable."""
    v = safe_float(value)
    r = safe_float(ref)
    s = safe_float(sigma)
    if v is None or r is None or s is None:
        return None
    if abs(s) < eps:
        return None
    return float((v - r) / s)


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

# Load ae2particle models trained on exp2
MODEL_DIRECTION_TAG = "ae2particle"

MODEL_NAME_MAP = {
    'NaCl': 'NaCl',
    'Citricacid': 'Citricacid',
    'Ajinomoto': 'Ajinomoto',
}

# Achieved rule using uncertainty (optional, but recommended to export)
# achieved_conservative: d50hat + K_SIGMA_D50*sigma <= target
# achieved_optimistic:   d50hat - K_SIGMA_D50*sigma <= target
K_SIGMA_D50 = 1.0  # set 0.0 if you want mean-only; keep 1.0 for analysis

os.makedirs(RESULTS_DIR, exist_ok=True)

print("--- Check last-2 crossing for AE->D50 estimation (ae2particle) + z-scores ---")

materials = natsorted([
    os.path.basename(d)
    for d in glob.glob(os.path.join(PSD_BASE_PATH, '*'))
    if os.path.isdir(d)
])

rows = []

for material in materials:
    model_key = MODEL_NAME_MAP.get(material, material)

    model_path = os.path.join(
        RESULTS_DIR,
        f"gpr_model_{MODEL_DIRECTION_TAG}_{model_key}_exp2.joblib"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    gpr = joblib.load(model_path)

    # Targets from naming in exp3 PSD
    target_files_sample = glob.glob(os.path.join(PSD_BASE_PATH, material, '1st', '*.csv'))
    targets = []
    for f in target_files_sample:
        m = re.search(r'_for_?(\d+)um', os.path.basename(f))
        if m:
            targets.append(int(m.group(1)))
    targets = natsorted(list(set(targets)))

    for target_val in targets:
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

            last_ae = safe_float(last_ae)
            second_ae = safe_float(second_ae)

            # Predict D50_hat from AE (if available)
            d50hat_last = None
            d50hat_second = None
            sigma_last = None
            sigma_second = None

            if last_ae is not None:
                y, s = gpr.predict(np.array([[last_ae]], dtype=float), return_std=True)
                d50hat_last = float(y[0])
                sigma_last = float(s[0])

            if second_ae is not None:
                y, s = gpr.predict(np.array([[second_ae]], dtype=float), return_std=True)
                d50hat_second = float(y[0])
                sigma_second = float(s[0])

            # --- Achieved definitions ---
            # mean-only (your current)
            achieved_last_mean = (d50hat_last is not None and d50hat_last <= target_val)
            achieved_second_mean = (d50hat_second is not None and d50hat_second <= target_val)

            # conservative (require upper confidence bound <= target)
            achieved_last_cons = (
                d50hat_last is not None and sigma_last is not None and (d50hat_last + K_SIGMA_D50 * sigma_last) <= target_val
            )
            achieved_second_cons = (
                d50hat_second is not None and sigma_second is not None and (d50hat_second + K_SIGMA_D50 * sigma_second) <= target_val
            )

            # optimistic (require lower confidence bound <= target)
            achieved_last_opt = (
                d50hat_last is not None and sigma_last is not None and (d50hat_last - K_SIGMA_D50 * sigma_last) <= target_val
            )
            achieved_second_opt = (
                d50hat_second is not None and sigma_second is not None and (d50hat_second - K_SIGMA_D50 * sigma_second) <= target_val
            )

            # Expected pattern for "stop when achieved" (mean-only) to match your prior summary
            crossed_between_last2 = None
            if (d50hat_second is not None) and (d50hat_last is not None):
                crossed_between_last2 = (d50hat_second > target_val) and (d50hat_last <= target_val)

            # Case classification (mean-only; parallel to threshold check)
            case = "Missing"
            if (d50hat_second is not None) and (d50hat_last is not None):
                if (d50hat_second > target_val) and (d50hat_last <= target_val):
                    case = "ExpectedCross"
                elif (d50hat_second <= target_val) and (d50hat_last <= target_val):
                    case = "BothAchieved"
                elif (d50hat_second > target_val) and (d50hat_last > target_val):
                    case = "BothNot"
                elif (d50hat_second <= target_val) and (d50hat_last > target_val):
                    case = "Rebound"

            # --- New: z relative to TARGET (how many sigma above/below target) ---
            # z>0 means predicted is above target (not achieved), z<0 achieved with margin
            z_last = safe_z(d50hat_last, target_val, sigma_last)
            z_second = safe_z(d50hat_second, target_val, sigma_second)

            rows.append({
                "Material": material,
                "Trial": trial,
                "Target_D50": int(target_val),
                "Measured_D50": safe_float(measured_d50),

                "Model_Path": model_path,
                "K_SIGMA_D50": float(K_SIGMA_D50),

                "N_AE_files_for_target": int(len(ae_files)),

                "AE_second_last_file": second_name,
                "AE_second_last_mV2": second_ae,
                "D50hat_second_last": safe_float(d50hat_second),
                "D50hat_second_last_sigma": safe_float(sigma_second),

                "Achieved_second_last_mean": achieved_second_mean if d50hat_second is not None else None,
                "Achieved_second_last_conservative": achieved_second_cons if d50hat_second is not None else None,
                "Achieved_second_last_optimistic": achieved_second_opt if d50hat_second is not None else None,

                "AE_last_file": last_name,
                "AE_last_mV2": last_ae,
                "D50hat_last": safe_float(d50hat_last),
                "D50hat_last_sigma": safe_float(sigma_last),

                "Achieved_last_mean": achieved_last_mean if d50hat_last is not None else None,
                "Achieved_last_conservative": achieved_last_cons if d50hat_last is not None else None,
                "Achieved_last_optimistic": achieved_last_opt if d50hat_last is not None else None,

                "Crossed_between_last2": crossed_between_last2,
                "Case": case,

                # standardized diagnostics
                "z_second_last": z_second,
                "z_last": z_last,
                "k_equiv_second_last": z_second,  # same quantity; kept for naming symmetry
                "k_equiv_last": z_last,
            })

df = pd.DataFrame(rows)

# Save detail
out_detail = os.path.join(RESULTS_DIR, "exp3_ae2particle_stop_check_last2_detail_with_z.csv")
df.to_csv(out_detail, index=False)
print(f"\nSaved detail: {out_detail}")

# Save summary (same as threshold summary)
# Material,Target_D50,Case,count
if not df.empty:
    summary = (
        df.groupby(["Material", "Target_D50", "Case"])
          .size()
          .reset_index(name="count")
    )
    out_summary = os.path.join(RESULTS_DIR, "exp3_ae2particle_stop_check_last2_summary.csv")
    summary.to_csv(out_summary, index=False)
    print(f"Saved summary: {out_summary}")

    # Rate table with z medians (parallel to your threshold "rate_with_z")
    rate_rows = []
    for (mat, tgt), g in df.groupby(["Material", "Target_D50"]):
        n = len(g)
        n_exp = int(np.sum(g["Case"] == "ExpectedCross"))
        rate_rows.append({
            "Material": mat,
            "Target_D50": tgt,
            "N": n,
            "N_ExpectedCross": n_exp,
            "Rate_ExpectedCross": n_exp / n if n else np.nan,
            "median_z_last": float(np.nanmedian(pd.to_numeric(g["z_last"], errors="coerce"))),
            "median_k_equiv_last": float(np.nanmedian(pd.to_numeric(g["k_equiv_last"], errors="coerce"))),
        })
    rate_df = pd.DataFrame(rate_rows)
    out_rate = os.path.join(RESULTS_DIR, "exp3_ae2particle_stop_check_last2_rate_with_z.csv")
    rate_df.to_csv(out_rate, index=False)
    print(f"Saved rate: {out_rate}")

    print("\nTop of rate table:")
    print(rate_df.sort_values(["Material", "Target_D50"]).head(30))
else:
    print("No data was processed.")
