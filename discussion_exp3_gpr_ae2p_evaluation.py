import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from math import erf, sqrt

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from fft_processing import calculate_fft_power
from monotone_bernstein import load_model

# Match figure typography to other discussion plots
plt.rcParams.update(
    {
        "font.size": 24,
        "axes.labelsize": 32,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 18,
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    }
)

RESULTS_DIR = os.path.join("results", "paper_plots")
DISCUSSION_DIR = os.path.join(RESULTS_DIR, "discussion")
SI_DIR = os.path.join("results", "SI_plots")
DISCUSSION_MODEL_DIR = RESULTS_DIR
INPUT_CSV_DEFAULT = os.path.join(DISCUSSION_DIR, "exp3_gpr_evaluation_detail.csv")
SI_DETAIL_CSV_DEFAULT = os.path.join(SI_DIR, "exp3_gpr_evaluation_detail.csv")
M_BOUNDARY = 1.96
AE_SCALE_TO_MV2 = 1e6
MOVING_AVG_WINDOW = 4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "ae_power_cache.json")
MONO_BERNSTEIN_BIC_TRAINER = os.path.join(
    SCRIPT_DIR, "exp2_monotone_bernstein_model_bic.py"
)

PLOT_LABEL_MAP = {
    "MSG": "MSG",
}
EXP2_TRAIN_REAGENTS = ["NaCl", "Citricacid", "MSG"]


def norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


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


def load_ae_power_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_ae_power_cache(cache_path: str, cache_data: dict) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, sort_keys=True)
    except OSError:
        pass


def get_cached_ae_power(cache: dict, file_path: str) -> tuple[float | None, bool]:
    key = norm_path(file_path)
    cached = cache.get(key)

    # Backward compatibility: cache value may be float or {"power": float, ...}
    if isinstance(cached, dict):
        p = safe_float(cached.get("power"))
        if p is not None:
            return float(p), False
    else:
        p = safe_float(cached)
        if p is not None:
            return float(p), False

    p = calculate_fft_power(file_path)
    if p is None or not np.isfinite(p):
        return None, False

    cache[key] = float(p)
    return float(p), True


def fit_residual_gp(x_data: np.ndarray, residual: np.ndarray, n_restarts: int = 10):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
    )
    gp.fit(np.asarray(x_data, dtype=float).reshape(-1, 1), np.asarray(residual, dtype=float).reshape(-1))
    return gp


def build_exp2_dataset(cache: dict) -> tuple[np.ndarray, bool]:
    psd_base_path = os.path.join("data", "powder_size_distribution", "exp2")
    ae_base_path = os.path.join("data", "ae", "exp2")

    all_psd_files = []
    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, "*")):
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, "*")):
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, "*.csv")))

    collected_data = []
    cache_dirty = False

    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue

        reagent = path_parts[-3]
        trial = path_parts[-2]
        d50 = get_d50(psd_file)
        if d50 is None:
            continue

        match = re.search(r"(grind\d+min)", os.path.basename(psd_file))
        if not match:
            continue
        grind_key = match.group(1)

        ae_session_path = os.path.join(ae_base_path, reagent, trial)
        ae_files = sorted(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))
        if not ae_files:
            continue

        ae_power_timeseries = []
        for f in ae_files:
            p, updated = get_cached_ae_power(cache, f)
            cache_dirty = cache_dirty or updated
            if p is not None:
                ae_power_timeseries.append(float(p))
        if len(ae_power_timeseries) < MOVING_AVG_WINDOW:
            continue

        ae_power_mV2 = np.asarray(ae_power_timeseries, dtype=float) * AE_SCALE_TO_MV2
        smoothed = moving_average(ae_power_mV2, window_size=MOVING_AVG_WINDOW)
        if smoothed.size == 0:
            continue

        final_ae_power = float(smoothed[-1])
        collected_data.append((float(d50), final_ae_power, trial, reagent))

    if not collected_data:
        return np.array([], dtype=object), cache_dirty
    return np.array(collected_data, dtype=object), cache_dirty


def train_exp2_monotone_bernstein_bic_models(model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(MONO_BERNSTEIN_BIC_TRAINER):
        raise FileNotFoundError(f"Trainer script not found: {MONO_BERNSTEIN_BIC_TRAINER}")

    print("Training exp2 monotone Bernstein BIC models (ae2p + p2ae)...")
    subprocess.run([sys.executable, MONO_BERNSTEIN_BIC_TRAINER], check=True)

    default_model_dir = RESULTS_DIR
    if norm_path(model_dir) == norm_path(default_model_dir):
        return

    for reagent in EXP2_TRAIN_REAGENTS:
        for direction in ("ae2particle", "particle2ae"):
            filename = f"monotone_bernstein_model_bic_{direction}_{reagent}_exp2.joblib"
            src = os.path.join(default_model_dir, filename)
            dst = os.path.join(model_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)


def load_bic_ae2p_mean_models(model_dir: str) -> dict[str, object]:
    models = {}
    for reagent in EXP2_TRAIN_REAGENTS:
        model_path = os.path.join(model_dir, f"monotone_bernstein_model_bic_ae2particle_{reagent}_exp2.joblib")
        reg, _extra = load_model(model_path)
        models[reagent] = reg
    return models


def build_ae2p_residual_gp_models_from_bic(model_dir: str) -> dict[str, GaussianProcessRegressor]:
    cache = load_ae_power_cache(CACHE_FILE)
    data_array, cache_dirty = build_exp2_dataset(cache)
    if cache_dirty:
        save_ae_power_cache(CACHE_FILE, cache)
    if data_array.size == 0:
        raise RuntimeError("No EXP2 matched data points found for AE2P residual GP fitting.")

    bic_mean_models = load_bic_ae2p_mean_models(model_dir)
    residual_gp_models: dict[str, GaussianProcessRegressor] = {}

    for reagent in EXP2_TRAIN_REAGENTS:
        mask = data_array[:, 3] == reagent
        reagent_data = data_array[mask]
        if reagent_data.size == 0:
            continue
        d50_vals = np.asarray(reagent_data[:, 0], dtype=float)
        ae_vals = np.asarray(reagent_data[:, 1], dtype=float)
        y_mean = bic_mean_models[reagent].predict(ae_vals)
        residual = d50_vals - np.asarray(y_mean, dtype=float)
        residual_gp_models[reagent] = fit_residual_gp(ae_vals, residual)

    return residual_gp_models


def predict_ae2p_bic_mean_residual_sigma(
    bic_mean_model,
    residual_gp: GaussianProcessRegressor,
    ae_value_mV2: float,
) -> tuple[float | None, float | None]:
    x = np.array([float(ae_value_mV2)], dtype=float)
    try:
        mu = bic_mean_model.predict(x)
        _, std = residual_gp.predict(x.reshape(-1, 1), return_std=True)
    except Exception:
        return None, None
    return safe_float(mu[0]), safe_float(std[0])


def export_paper_plots_ae2p_compat(detail_df: pd.DataFrame) -> None:
    """Export AE2P-only compatibility CSVs under results/paper_plots."""
    if detail_df.empty:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    detail_export = detail_df.copy()
    detail_export["Common_Measured_Mean"] = detail_export["Common_Measured_D50"]
    detail_export["P2AE_Total_Deviation"] = np.nan
    detail_export["P2AE_Total_Deviation_Percent"] = np.nan
    detail_export["AE2P_GPR_Prediction"] = detail_export.apply(
        lambda row: (
            f"{row['AE2P_Predicted_D50']:.2f} ± {row['AE2P_Predicted_Sigma']:.2f}"
            if pd.notnull(row["AE2P_Predicted_D50"]) and pd.notnull(row["AE2P_Predicted_Sigma"])
            else "N/A"
        ),
        axis=1,
    )
    detail_export["num_AE2P_mu_GPR"] = detail_export["AE2P_Predicted_D50"]
    detail_export["num_AE2P_sigma_GPR"] = detail_export["AE2P_Predicted_Sigma"]
    detail_export["num_Common_AE_Last_mV2_mu_trial"] = detail_export["Common_AE_Last_mV2"]
    detail_export["num_Common_mu_trial"] = detail_export["Common_Measured_D50"]
    detail_export["num_Common_sigma_trial"] = np.nan

    detail_cols = [
        "Material",
        "Trial",
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
        "Common_Measured_D50",
        "Common_AE_Last_mV2",
        "AE2P_Predicted_D50",
        "AE2P_Predicted_Sigma",
    ]
    detail_cols = [c for c in detail_cols if c in detail_export.columns]
    detail_export = detail_export[detail_cols + [c for c in detail_export.columns if c not in detail_cols]]
    detail_out = os.path.join(RESULTS_DIR, "exp3_evaluation_detail.csv")
    detail_export.to_csv(detail_out, index=False)
    print(f"Saved AE2P-only compatibility detail to: {detail_out}")

    summary_rows = []
    for (mat, tgt), g in detail_df.groupby(["Material", "Target_D50"]):
        mu_ae_last = g["Common_AE_Last_mV2"].mean()
        mu_trial = g["Common_Measured_D50"].mean()
        sigma_trial = g["Common_Measured_D50"].std(ddof=1)

        mu_gpr = g["AE2P_Predicted_D50"].mean()
        mean_sigma_gpr = g["AE2P_Predicted_Sigma"].mean()
        mean_est_error = g["AE2P_Estimation_Error"].mean()
        mean_est_error_pct = g["AE2P_Estimation_Error_Percent"].mean()
        ae2p_in_gpr_range = None
        if pd.notnull(mean_est_error) and pd.notnull(mean_sigma_gpr):
            ae2p_in_gpr_range = abs(mean_est_error) <= mean_sigma_gpr

        meas_str = f"{mu_trial:.2f} ± {sigma_trial:.2f}" if pd.notnull(mu_trial) else "N/A"
        gpr_str = f"{mu_gpr:.2f} ± {mean_sigma_gpr:.2f}" if pd.notnull(mu_gpr) else "N/A"

        summary_rows.append(
            {
                "Material": mat,
                "Target_D50": tgt,
                "Common_Measured_Mean": meas_str,
                "P2AE_Total_Deviation": np.nan,
                "P2AE_Total_Deviation_Percent": np.nan,
                "AE2P_GPR_Prediction": gpr_str,
                "AE2P_Estimation_Error": mean_est_error,
                "AE2P_Estimation_Error_Percent": mean_est_error_pct,
                "AE2P_Est_Error_In_GPR_Range": ae2p_in_gpr_range,
                "num_AE2P_mu_GPR": mu_gpr,
                "num_AE2P_sigma_GPR": mean_sigma_gpr,
                "num_Common_AE_Last_mV2_mu_trial": round(float(mu_ae_last), 2)
                if pd.notnull(mu_ae_last)
                else np.nan,
                "num_Common_mu_trial": mu_trial,
                "num_Common_sigma_trial": sigma_trial,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
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
    summary_cols = [c for c in summary_cols if c in summary_df.columns]
    summary_df = summary_df[summary_cols + [c for c in summary_df.columns if c not in summary_cols]]
    summary_out = os.path.join(RESULTS_DIR, "exp3_evaluation_summary_for_table.csv")
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved AE2P-only compatibility summary to: {summary_out}")


def collect_exp3_materials() -> list[str]:
    psd_base_path = os.path.join("data", "powder_size_distribution", "exp3")
    materials = [
        os.path.basename(d)
        for d in glob.glob(os.path.join(psd_base_path, "*"))
        if os.path.isdir(d)
    ]
    materials.sort(key=natural_keys)
    return [m for m in materials if not m.endswith("_gpr")]


def ensure_gpr_models(model_dir: str, force_retrain: bool = False) -> None:
    expected = []
    for reagent in EXP2_TRAIN_REAGENTS:
        for direction in ("ae2particle", "particle2ae"):
            expected.append(
                os.path.join(
                    model_dir,
                    f"monotone_bernstein_model_bic_{direction}_{reagent}_exp2.joblib",
                )
            )

    missing = [p for p in expected if not os.path.exists(p)]
    if force_retrain or missing:
        if missing:
            print("Missing monotone Bernstein BIC model files were detected.")
        train_exp2_monotone_bernstein_bic_models(model_dir)


def build_exp3_gpr_detail(model_dir: str, output_csv: str) -> None:
    psd_base_path = os.path.join("data", "powder_size_distribution", "exp3")
    ae_base_path = os.path.join("data", "ae", "exp3")

    cache = load_ae_power_cache(CACHE_FILE)
    cache_dirty = False
    rows = []
    bic_mean_models = load_bic_ae2p_mean_models(model_dir)
    residual_gp_models = build_ae2p_residual_gp_models_from_bic(model_dir)

    materials = collect_exp3_materials()
    for material in materials:
        bic_mean_model = bic_mean_models.get(material)
        residual_gp_model = residual_gp_models.get(material)
        if bic_mean_model is None or residual_gp_model is None:
            print(f"Warning: missing BIC mean/residual GP model for {material}, skipping.")
            continue

        target_files_sample = glob.glob(os.path.join(psd_base_path, material, "1st", "*.csv"))
        targets = []
        for f in target_files_sample:
            m = re.search(r"_for_?(\d+)um", os.path.basename(f))
            if m:
                targets.append(int(m.group(1)))
        targets = sorted(set(targets))

        for target_val in targets:
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
                ae_values = {}
                ae_series = []
                for f in ae_files:
                    p, updated = get_cached_ae_power(cache, f)
                    cache_dirty = cache_dirty or updated
                    if p is None:
                        continue
                    v = float(p) * AE_SCALE_TO_MV2
                    ae_values[f] = v
                    ae_series.append(v)

                ae_last = ae_values.get(last_file)
                ae_smoothed_last = None
                if len(ae_series) >= MOVING_AVG_WINDOW:
                    smoothed = moving_average(np.array(ae_series, dtype=float), window_size=MOVING_AVG_WINDOW)
                    if smoothed.size > 0 and np.isfinite(smoothed[-1]):
                        ae_smoothed_last = float(smoothed[-1])

                d50_hat = None
                d50_hat_sigma = None
                ae_for_estimation = ae_smoothed_last if ae_smoothed_last is not None else ae_last
                if ae_for_estimation is not None:
                    d50_hat, d50_hat_sigma = predict_ae2p_bic_mean_residual_sigma(
                        bic_mean_model,
                        residual_gp_model,
                        ae_for_estimation,
                    )

                estimation_error = None
                estimation_error_percent = None
                estimation_error_in_range = None
                estimation_error_pm = None
                estimation_error_mp = None
                z_pred_pm = None
                z_pred_mp = None
                estimation_error_percent_pm = None
                estimation_error_percent_mp = None
                if d50_hat is not None and measured_d50_f is not None:
                    estimation_error_pm = d50_hat - measured_d50_f
                    estimation_error_mp = measured_d50_f - d50_hat
                    estimation_error = estimation_error_mp
                    if measured_d50_f != 0:
                        estimation_error_percent_pm = (estimation_error_pm / measured_d50_f) * 100.0
                        estimation_error_percent_mp = (estimation_error_mp / measured_d50_f) * 100.0
                        estimation_error_percent = estimation_error_percent_mp
                    if d50_hat_sigma is not None and abs(float(d50_hat_sigma)) > 1e-12:
                        z_pred_pm = estimation_error_pm / d50_hat_sigma
                        z_pred_mp = estimation_error_mp / d50_hat_sigma
                        estimation_error_in_range = abs(estimation_error) <= d50_hat_sigma

                rows.append(
                    {
                        "Material": material,
                        "Trial": trial,
                        "Target_D50": target_val,
                        "Common_Measured_D50": measured_d50_f,
                        "Common_AE_Last_mV2": ae_last,
                        "AE2P_Predicted_D50": d50_hat,
                        "AE2P_Predicted_Sigma": d50_hat_sigma,
                        "AE2P_Estimation_Error": estimation_error,
                        "AE2P_Estimation_Error_Percent": estimation_error_percent,
                        "AE2P_Estimation_Error_PM": estimation_error_pm,
                        "AE2P_Estimation_Error_MP": estimation_error_mp,
                        "AE2P_z_pred_PM": z_pred_pm,
                        "AE2P_z_pred_MP": z_pred_mp,
                        "AE2P_Estimation_Error_Percent_PM": estimation_error_percent_pm,
                        "AE2P_Estimation_Error_Percent_MP": estimation_error_percent_mp,
                        "AE2P_Est_Error_In_GPR_Range": estimation_error_in_range,
                    }
                )

    detail_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    detail_df.to_csv(output_csv, index=False)
    os.makedirs(SI_DIR, exist_ok=True)
    detail_df.to_csv(SI_DETAIL_CSV_DEFAULT, index=False)
    export_paper_plots_ae2p_compat(detail_df)
    if cache_dirty:
        save_ae_power_cache(CACHE_FILE, cache)
    print(f"Saved discussion exp3 detail to: {output_csv}")
    print(f"Saved SI exp3 detail to: {SI_DETAIL_CSV_DEFAULT}")


def ensure_detail_csv(input_csv: str, model_dir: str, force_retrain_models: bool, force_rebuild_detail: bool) -> None:
    ensure_gpr_models(model_dir=model_dir, force_retrain=force_retrain_models)
    if force_rebuild_detail or (not os.path.exists(input_csv)):
        print("Preparing discussion exp3 detailed evaluation CSV...")
        build_exp3_gpr_detail(model_dir=model_dir, output_csv=input_csv)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Discussion AE2P error plot using BIC-selected monotone Bernstein mean + residual GP uncertainty "
            "with automatic detail generation."
        )
    )
    parser.add_argument("--input-csv", type=str, default=INPUT_CSV_DEFAULT)
    parser.add_argument("--model-dir", type=str, default=DISCUSSION_MODEL_DIR)
    parser.add_argument("--force-retrain-models", action="store_true")
    parser.add_argument("--force-rebuild-detail", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    input_csv = args.input_csv
    model_dir = args.model_dir

    if not args.skip_prepare:
        ensure_detail_csv(
            input_csv=input_csv,
            model_dir=model_dir,
            force_retrain_models=args.force_retrain_models,
            force_rebuild_detail=args.force_rebuild_detail,
        )

    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        print("Run with preparation enabled (default) to generate models and detail CSV.")
        return

    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"Input file is empty: {input_csv}")
        return

    required_cols = [
        "Material",
        "Trial",
        "Target_D50",
        "Common_Measured_D50",
        "AE2P_Predicted_D50",
        "AE2P_Predicted_Sigma",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing required columns in {input_csv}: {missing}")
        return

    export_paper_plots_ae2p_compat(df)

    plot_df = df.copy()
    plot_df["AE2P_Measured_Error"] = (
        plot_df["Common_Measured_D50"] - plot_df["AE2P_Predicted_D50"]
    )
    plot_df["AE2P_m"] = plot_df["AE2P_Measured_Error"] / plot_df["AE2P_Predicted_Sigma"]
    plot_df["AE2P_p"] = plot_df["AE2P_m"].apply(
        lambda v: normal_cdf(v) if np.isfinite(v) else np.nan
    )
    plot_df = plot_df.dropna(
        subset=["AE2P_m", "AE2P_p", "Material", "Trial", "Target_D50"]
    )
    plot_df["AE2P_p_ge_threshold"] = plot_df["AE2P_m"].abs() <= M_BOUNDARY

    if plot_df.empty:
        print("No valid AE2P rows to export/plot.")
        return

    plot_df["label"] = plot_df.apply(
        lambda row: (
            f"{PLOT_LABEL_MAP.get(row['Material'], row['Material'])} "
            f"{row['Trial']} {row['Target_D50']} $\\mu$m"
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
        "AE2P_Measured_Error",
        "AE2P_m",
        "AE2P_p",
        "AE2P_p_ge_threshold",
        "label",
    ]
    export_cols = [c for c in export_cols if c in plot_df.columns]

    os.makedirs(DISCUSSION_DIR, exist_ok=True)
    export_path = os.path.join(DISCUSSION_DIR, "ae2p_upper_error_points.csv")
    export_df = plot_df[export_cols].copy()
    export_df["AE2P_Predicted_95PI"] = export_df["AE2P_Predicted_Sigma"] * M_BOUNDARY
    export_df = export_df.drop(columns=["AE2P_Predicted_Sigma"])
    export_df.to_csv(export_path, index=False)
    print(f"Saved detail plot data to: {export_path}")

    plot_df = plot_df.reset_index(drop=True)
    plot_df["x_label"] = plot_df.apply(
        lambda row: (
            f"{PLOT_LABEL_MAP.get(row['Material'], row['Material'])} "
            f"{row['Target_D50']} $\\mu$m"
        ),
        axis=1,
    )
    unique_labels = list(dict.fromkeys(plot_df["x_label"].tolist()))
    x_lookup = {label: idx for idx, label in enumerate(unique_labels)}
    x_vals = plot_df["x_label"].map(x_lookup).astype(float).values

    plt.figure(figsize=(14, 8))
    mask_true = plot_df["AE2P_p_ge_threshold"] == True
    mask_false = plot_df["AE2P_p_ge_threshold"] == False
    plt.scatter(
        x_vals[mask_true],
        plot_df.loc[mask_true, "AE2P_m"].values,
        s=80,
        c="black",
        marker="o",
        label="Within prediction range",
    )
    plt.scatter(
        x_vals[mask_false],
        plot_df.loc[mask_false, "AE2P_m"].values,
        s=80,
        c="black",
        marker="^",
        label="Outside prediction range",
    )
    plt.axhline(M_BOUNDARY, color="black", linewidth=1.0, alpha=0.6, linestyle="--")
    plt.axhline(-M_BOUNDARY, color="black", linewidth=1.0, alpha=0.6, linestyle="--")

    max_abs = float(np.nanmax(np.abs(plot_df["AE2P_m"].values)))
    max_abs = max(max_abs, M_BOUNDARY)
    if np.isfinite(max_abs) and max_abs > 0.0:
        plt.ylim(-max_abs * 1.1, max_abs * 1.1)

    plt.xticks(
        np.arange(len(unique_labels), dtype=float),
        unique_labels,
        rotation=45,
        ha="right",
        fontsize=plt.rcParams["xtick.labelsize"],
    )
    plt.xlabel("")
    plt.ylabel("Normalized Error")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(DISCUSSION_DIR, "ae2p_error.png")
    plot_pdf_path = os.path.join(DISCUSSION_DIR, "ae2p_error.pdf")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_pdf_path)
    plt.close()
    print(f"Saved detail plot to: {plot_path}")
    print(f"Saved detail plot to: {plot_pdf_path}")


if __name__ == "__main__":
    main()
