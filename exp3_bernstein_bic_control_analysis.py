import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from fft_processing import calculate_fft_power
from monotone_bernstein import load_model


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]


def parse_timestamp_key(filename: str):
    base = os.path.basename(filename)
    m = re.match(r"(\d{8})_(\d{6})", base)
    if m:
        return (m.group(1), m.group(2), base)
    return ("", "", base)


def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def get_d50(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Dx (50)"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        return float(parts[1])
    except (IOError, ValueError) as e:
        print(f"Could not read or parse D50 from {file_path}: {e}")
    return None


def load_ae_power_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_ae_power_cache(cache_path, cache_data):
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, sort_keys=True)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Unified exp3 evaluation using monotone Bernstein BIC models (P2AE + AE2P)."
    )
    parser.add_argument("--model-dir", type=str, default="results/aic_bic")
    parser.add_argument("--output-dir", type=str, default="results/aic_bic")
    parser.add_argument("--experiment", type=str, default="exp3_bernstein_bic")
    parser.add_argument("--ae-scale-to-mv2", type=float, default=1e6)
    parser.add_argument("--moving-avg-window", type=int, default=4)
    args = parser.parse_args()

    results_dir = os.path.abspath(args.output_dir)
    experiment = args.experiment
    psd_base_path = os.path.join("data", "powder_size_distribution", experiment)
    ae_base_path = os.path.join("data", "ae", experiment)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, "ae_power_cache.json")
    ae_power_cache = load_ae_power_cache(cache_file)
    cache_dirty = False

    os.makedirs(results_dir, exist_ok=True)

    model_name_map = {
        "NaCl": "NaCl",
        "Citricacid": "Citricacid",
        "MSG": "MSG",
    }

    def get_cached_ae_power(file_path):
        nonlocal cache_dirty
        if file_path is None:
            return None
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            return None
        cached = ae_power_cache.get(file_path)
        if isinstance(cached, dict) and cached.get("mtime") == mtime:
            cached_power = cached.get("power")
            if cached_power is not None and np.isfinite(cached_power):
                return float(cached_power)
        p = calculate_fft_power(file_path)
        if p is None or not np.isfinite(p):
            return None
        ae_power_cache[file_path] = {"mtime": mtime, "power": float(p)}
        cache_dirty = True
        return float(p)

    print("--- Unified Evaluation: Bernstein BIC (P2AE + AE2P) ---")

    materials = [os.path.basename(d) for d in glob.glob(os.path.join(psd_base_path, "*")) if os.path.isdir(d)]
    materials.sort(key=natural_keys)

    rows = []

    for material in materials:
        model_key = model_name_map.get(material, material)

        model_a_path = os.path.join(
            args.model_dir,
            f"monotone_bernstein_model_bic_particle2ae_{model_key}_exp2.joblib",
        )
        if not os.path.exists(model_a_path):
            print(f"Warning: BIC P2AE model not found for {material}: {model_a_path}")
            continue
        model_a, _ = load_model(model_a_path)

        model_b_path = os.path.join(
            args.model_dir,
            f"monotone_bernstein_model_bic_ae2particle_{model_key}_exp2.joblib",
        )
        if not os.path.exists(model_b_path):
            print(f"Warning: BIC AE2P model not found for {material}: {model_b_path}")
            continue
        model_b, _ = load_model(model_b_path)

        target_files_sample = glob.glob(os.path.join(psd_base_path, material, "1st", "*.csv"))
        targets = []
        for f in target_files_sample:
            m = re.search(r"_for_?(\d+)um", os.path.basename(f))
            if m:
                targets.append(int(m.group(1)))
        targets = sorted(set(targets))

        for target_val in targets:
            x_target = np.array([float(target_val)], dtype=float)
            ae_th = float(model_a.predict(x_target)[0])

            for trial in ["1st", "2nd", "3rd"]:
                psd_dir = os.path.join(psd_base_path, material, trial)
                psd_candidates = glob.glob(os.path.join(psd_dir, "*.csv"))
                psd_file = None
                for f in psd_candidates:
                    if re.search(f"_for_?{target_val}um", os.path.basename(f)):
                        psd_file = f
                        break

                measured_d50 = get_d50(psd_file) if psd_file else None
                measured_d50_f = safe_float(measured_d50)

                ae_dir = os.path.join(ae_base_path, material, trial)
                ae_candidates = glob.glob(os.path.join(ae_dir, "*.csv"))
                ae_files = [f for f in ae_candidates if re.search(f"_for_?{target_val}um", os.path.basename(f))]
                ae_files.sort(key=parse_timestamp_key)

                last_file = ae_files[-1] if len(ae_files) >= 1 else None
                second_last_file = ae_files[-2] if len(ae_files) >= 2 else None

                ae_values = {}
                ae_series = []
                for f in ae_files:
                    p = get_cached_ae_power(f)
                    if p is None or not np.isfinite(p):
                        continue
                    v = float(p) * float(args.ae_scale_to_mv2)
                    ae_values[f] = v
                    ae_series.append(v)

                ae_last = ae_values.get(last_file)
                ae_second = ae_values.get(second_last_file)
                ae_smoothed_last = None
                if len(ae_series) >= int(args.moving_avg_window):
                    smoothed = moving_average(np.array(ae_series, dtype=float), window_size=int(args.moving_avg_window))
                    if smoothed.size > 0 and np.isfinite(smoothed[-1]):
                        ae_smoothed_last = float(smoothed[-1])

                is_expected_cross = False
                if ae_last is not None and ae_second is not None:
                    if (ae_last < ae_th) and (ae_second >= ae_th):
                        is_expected_cross = True

                total_deviation = None
                total_deviation_percent = None
                if measured_d50_f is not None:
                    total_deviation = float(target_val) - measured_d50_f
                    if target_val != 0:
                        total_deviation_percent = (total_deviation / float(target_val)) * 100.0

                d50_hat = None
                d50_hat_sigma = np.nan
                ae_for_estimation = ae_smoothed_last if ae_smoothed_last is not None else ae_last
                if ae_for_estimation is not None:
                    d50_hat = float(model_b.predict(np.array([ae_for_estimation], dtype=float))[0])

                estimation_error = None
                estimation_error_percent = None
                estimation_error_in_range = None
                if d50_hat is not None and measured_d50_f is not None:
                    estimation_error = measured_d50_f - d50_hat
                    if measured_d50_f != 0:
                        estimation_error_percent = (estimation_error / measured_d50_f) * 100.0

                rows.append(
                    {
                        "Material": material,
                        "Trial": trial,
                        "Target_D50": target_val,
                        "Common_Measured_D50": measured_d50_f,
                        "Common_AE_Last_mV2": ae_last,
                        "P2AE_AE_Threshold": ae_th,
                        "P2AE_Is_ExpectedCross": is_expected_cross,
                        "P2AE_Total_Deviation": total_deviation,
                        "P2AE_Total_Deviation_Percent": total_deviation_percent,
                        "AE2P_Predicted_D50": d50_hat,
                        "AE2P_Predicted_Sigma": d50_hat_sigma,
                        "AE2P_Estimation_Error": estimation_error,
                        "AE2P_Estimation_Error_Percent": estimation_error_percent,
                        "AE2P_Est_Error_In_GPR_Range": estimation_error_in_range,
                    }
                )

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
                f"{row['AE2P_Predicted_D50']:.2f} ± N/A"
                if pd.notnull(row["AE2P_Predicted_D50"])
                else "N/A"
            ),
            axis=1,
        )
        df["num_AE2P_mu_GPR"] = df["AE2P_Predicted_D50"]
        df["num_AE2P_sigma_GPR"] = df["AE2P_Predicted_Sigma"]
        df["num_Common_mu_trial"] = df["Common_Measured_D50"]
        df["num_Common_sigma_trial"] = np.nan

    detail_cols = ["Material", "Trial", "Target_D50"]
    detail_cols = [c for c in detail_cols if c in df.columns]
    ordered_cols = detail_cols + [c for c in summary_cols if c in df.columns]
    seen = set()
    ordered_cols = [c for c in ordered_cols if not (c in seen or seen.add(c))]
    if not df.empty:
        df = df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]

    out_detail = os.path.join(results_dir, "exp3_bernstein_bic_evaluation_detail.csv")
    df.to_csv(out_detail, index=False)
    print(f"Saved detailed results to: {out_detail}")

    if cache_dirty:
        save_ae_power_cache(cache_file, ae_power_cache)

    if df.empty:
        print("No data processed.")
        return

    summary_rows = []
    for (mat, tgt), g in df.groupby(["Material", "Target_D50"]):
        mu_trial = g["Common_Measured_D50"].mean()
        sigma_trial = g["Common_Measured_D50"].std(ddof=1)

        mu_pred = g["AE2P_Predicted_D50"].mean()
        mean_sigma_pred = np.nan
        mean_est_error = g["AE2P_Estimation_Error"].mean()
        mean_est_error_pct = g["AE2P_Estimation_Error_Percent"].mean()
        in_range = None

        mean_total_dev = g["P2AE_Total_Deviation"].mean()
        mean_total_dev_pct = g["P2AE_Total_Deviation_Percent"].mean()

        meas_str = f"{mu_trial:.2f} ± {sigma_trial:.2f}" if pd.notnull(mu_trial) else "N/A"
        pred_str = f"{mu_pred:.2f} ± N/A" if pd.notnull(mu_pred) else "N/A"

        summary_rows.append(
            {
                "Material": mat,
                "Target_D50": tgt,
                "AE2P_GPR_Prediction": pred_str,
                "Common_Measured_Mean": meas_str,
                "AE2P_Estimation_Error": mean_est_error,
                "AE2P_Estimation_Error_Percent": mean_est_error_pct,
                "AE2P_Est_Error_In_GPR_Range": in_range,
                "P2AE_Total_Deviation": mean_total_dev,
                "P2AE_Total_Deviation_Percent": mean_total_dev_pct,
                "num_AE2P_mu_GPR": mu_pred,
                "num_AE2P_sigma_GPR": mean_sigma_pred,
                "num_Common_mu_trial": mu_trial,
                "num_Common_sigma_trial": sigma_trial,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_cols_existing = [c for c in summary_cols if c in df_summary.columns]
    df_summary = df_summary[summary_cols_existing + [c for c in df_summary.columns if c not in summary_cols_existing]]

    out_summary = os.path.join(results_dir, "exp3_bernstein_bic_evaluation_summary_for_table.csv")
    df_summary.to_csv(out_summary, index=False)

    print("\n--- Summary Table Preview (Top 10) ---")
    print(
        df_summary[
            [
                "Material",
                "Target_D50",
                "AE2P_GPR_Prediction",
                "Common_Measured_Mean",
                "AE2P_Estimation_Error",
                "AE2P_Estimation_Error_Percent",
                "AE2P_Est_Error_In_GPR_Range",
                "P2AE_Total_Deviation",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nSummary table saved to: {out_summary}")


if __name__ == "__main__":
    main()
