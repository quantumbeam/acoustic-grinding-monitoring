import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Monotone-GP Threshold Pseudo-Stopping Visualization for exp3.

Applies monotone-GP forward models (particle2ae, trained on exp2) to exp3 AE time series,
identifies pseudo-stop cycles via threshold crossing, and outputs per-(Material, Target)
overlay plots + summary CSV.

Output: model_comparison/monotone_gp/
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from fft_processing import calculate_fft_power
from monotone_svgp import load_model_npz

plt.style.use(["science", "ieee", "no-latex"])
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

EXPERIMENT = "exp3"
PSD_BASE_PATH = os.path.join("data/powder_size_distribution", EXPERIMENT)
AE_BASE_PATH = os.path.join("data/ae", EXPERIMENT)
AE_SCALE_TO_MV2 = 1e6
MOVING_AVG_WINDOW = 4
OUTPUT_DIR = os.path.join("model_comparison", "monotone_gp")
MODEL_DIR = os.path.join("model_comparison", "monotone_gp")
K_SIGMA_AE = 0.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "ae_power_cache.json")

MODEL_NAME_MAP = {"NaCl": "NaCl", "Citricacid": "Citricacid", "MSG": "MSG"}
TRIAL_COLORS = {"1st": "black", "2nd": "red", "3rd": "blue"}
TRIAL_MARKERS = {"1st": "o", "2nd": "x", "3rd": "^"}


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
    except (IOError, ValueError):
        return None
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


def main():
    global CACHE_DIRTY

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- Monotone-GP Threshold Pseudo-Stopping Analysis (exp3) ---")
    print(f"    K_SIGMA_AE = {K_SIGMA_AE}")

    materials = [
        os.path.basename(d)
        for d in glob.glob(os.path.join(PSD_BASE_PATH, "*"))
        if os.path.isdir(d)
    ]
    materials.sort(key=natural_keys)

    rows = []

    for material in materials:
        model_key = MODEL_NAME_MAP.get(material, material)
        model_path = os.path.join(
            MODEL_DIR,
            f"monotone_gp_model_particle2ae_{model_key}_exp2.npz",
        )
        if not os.path.exists(model_path):
            print(f"Warning: monotone-GP model not found for {material}: {model_path}")
            continue

        reg, _ = load_model_npz(model_path)
        print(f"\n[{material}] Monotone-GP model loaded.")

        target_files_sample = glob.glob(os.path.join(PSD_BASE_PATH, material, "1st", "*.csv"))
        targets = []
        for f in target_files_sample:
            m = re.search(r"_for_?(\d+)um", os.path.basename(f))
            if m:
                targets.append(int(m.group(1)))
        targets = sorted(set(targets))

        for target_val in targets:
            ae_mu, ae_var = reg.predict_f(np.array([[float(target_val)]], dtype=float))
            theta_sigma = float(np.sqrt(max(ae_var[0], 0.0)))
            theta_ae = float(ae_mu[0]) - K_SIGMA_AE * theta_sigma

            print(
                f"  Target={target_val} um -> theta_AE={theta_ae:.4f} mV² "
                f"(mu={float(ae_mu[0]):.4f}, sigma={theta_sigma:.4f})"
            )

            trial_data = {}

            for trial in ["1st", "2nd", "3rd"]:
                psd_dir = os.path.join(PSD_BASE_PATH, material, trial)
                psd_candidates = glob.glob(os.path.join(psd_dir, "*.csv"))
                psd_file = None
                for f in psd_candidates:
                    if re.search(f"_for_?{target_val}um", os.path.basename(f)):
                        psd_file = f
                        break
                measured_d50 = get_d50(psd_file) if psd_file else None

                ae_dir = os.path.join(AE_BASE_PATH, material, trial)
                ae_candidates = glob.glob(os.path.join(ae_dir, "*.csv"))
                ae_files = [
                    f for f in ae_candidates if re.search(f"_for_?{target_val}um", os.path.basename(f))
                ]
                ae_files.sort(key=parse_timestamp_key)

                raw_series = []
                for f in ae_files:
                    p = get_cached_ae_power(f)
                    if p is None or not np.isfinite(p):
                        continue
                    raw_series.append(float(p) * AE_SCALE_TO_MV2)

                n_cycles = len(raw_series)
                if n_cycles < MOVING_AVG_WINDOW:
                    rows.append(
                        {
                            "Material": material,
                            "Target_D50": target_val,
                            "Trial": trial,
                            "theta_monotone_gp": theta_ae,
                            "theta_monotone_gp_sigma": theta_sigma,
                            "reachable": False,
                            "cross_type": "unreach",
                            "k_star": None,
                            "S_AE_at_k_star": None,
                            "min_S_AE": None,
                            "margin": None,
                            "n_cycles": n_cycles,
                            "Measured_D50": measured_d50,
                        }
                    )
                    continue

                smoothed = moving_average(np.array(raw_series), window_size=MOVING_AVG_WINDOW)

                i_star = None
                for i, s in enumerate(smoothed):
                    if s <= theta_ae:
                        i_star = i
                        break

                reachable = i_star is not None
                k_star = i_star + (MOVING_AVG_WINDOW - 1) if reachable else None
                s_at_k = float(smoothed[i_star]) if reachable else None
                min_s = float(np.min(smoothed))
                margin = min_s - theta_ae

                if not reachable:
                    cross_type = "unreach"
                elif len(smoothed) >= 2 and smoothed[-2] > theta_ae and smoothed[-1] <= theta_ae:
                    cross_type = "last_cross"
                elif i_star == 0:
                    cross_type = "already_below"
                else:
                    cross_type = "mid_cross"

                rows.append(
                    {
                        "Material": material,
                        "Target_D50": target_val,
                        "Trial": trial,
                        "theta_monotone_gp": theta_ae,
                        "theta_monotone_gp_sigma": theta_sigma,
                        "reachable": reachable,
                        "cross_type": cross_type,
                        "k_star": k_star,
                        "S_AE_at_k_star": s_at_k,
                        "min_S_AE": min_s,
                        "margin": margin,
                        "n_cycles": n_cycles,
                        "Measured_D50": measured_d50,
                    }
                )

                trial_data[trial] = {
                    "raw": np.array(raw_series),
                    "smoothed": smoothed,
                    "k_star": k_star,
                    "reachable": reachable,
                }

            if trial_data:
                plt.figure(figsize=(12, 8))
                plotted_any = False
                for trial in ["1st", "2nd", "3rd"]:
                    if trial not in trial_data:
                        continue
                    d = trial_data[trial]
                    sm = d["smoothed"]
                    x = np.arange(MOVING_AVG_WINDOW - 1, MOVING_AVG_WINDOW - 1 + len(sm))
                    plt.plot(
                        x,
                        sm,
                        color=TRIAL_COLORS[trial],
                        marker=TRIAL_MARKERS[trial],
                        markersize=7,
                        linewidth=2.0,
                        label=f"{trial}",
                    )
                    if d["reachable"]:
                        k = d["k_star"]
                        yk = sm[k - (MOVING_AVG_WINDOW - 1)]
                        plt.scatter([k], [yk], color=TRIAL_COLORS[trial], s=120, zorder=5)
                    plotted_any = True

                if plotted_any:
                    plt.axhline(theta_ae, color="gray", linestyle="--", linewidth=2.0, label="Threshold")
                    plt.xlabel("Grinding cycle index")
                    plt.ylabel(r"Smoothed AE power ($\mathrm{mV}^2$)")
                    plt.title(f"{material}, Target={target_val} um")
                    plt.legend(loc="best")
                    plt.tight_layout()

                    out_png = os.path.join(OUTPUT_DIR, f"exp3_monotone_gp_pseudostop_{material}_{target_val}um.png")
                    plt.savefig(out_png, dpi=300)
                plt.close()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "exp3_monotone_gp_pseudostop_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"\nSaved summary: {out_csv}")

    if CACHE_DIRTY:
        save_ae_power_cache(CACHE_FILE, AE_POWER_CACHE)
        print(f"Updated AE cache: {CACHE_FILE}")


if __name__ == "__main__":
    main()
