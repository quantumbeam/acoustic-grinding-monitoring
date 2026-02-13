import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Monotone Bernstein Threshold Pseudo-Stopping Visualization for exp3.

Applies monotone Bernstein forward models (particle2ae, trained on exp2) to exp3 AE time series,
identifies pseudo-stop cycles via threshold crossing, and produces per-(Material, Target)
overlay plots plus a summary CSV.
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
from monotone_bernstein import load_model
from monotone_bernstein import threshold_from_target_d50

# ============================================================
# Style
# ============================================================
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

# ============================================================
# Constants
# ============================================================
EXPERIMENT = "exp3"
PSD_BASE_PATH = os.path.join("data/powder_size_distribution", EXPERIMENT)
AE_BASE_PATH = os.path.join("data/ae", EXPERIMENT)
AE_SCALE_TO_MV2 = 1e6
MOVING_AVG_WINDOW = 4
OUTPUT_DIR = os.path.join("model_comparison", "monotone_bernstein")
MODEL_DIR = os.path.join("model_comparison", "monotone_bernstein")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "ae_power_cache.json")

MODEL_NAME_MAP = {
    "NaCl": "NaCl",
    "Citricacid": "Citricacid",
    "MSG": "MSG",
}

TRIAL_COLORS = {"1st": "black", "2nd": "red", "3rd": "blue"}
TRIAL_MARKERS = {"1st": "o", "2nd": "x", "3rd": "^"}


# ============================================================
# Helper Functions
# ============================================================
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


# ============================================================
# AE Cache
# ============================================================
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


# ============================================================
# Main
# ============================================================
def main():
    global CACHE_DIRTY

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- Monotone Bernstein Threshold Pseudo-Stopping Analysis (exp3) ---")

    materials = [
        os.path.basename(d) for d in glob.glob(os.path.join(PSD_BASE_PATH, "*")) if os.path.isdir(d)
    ]
    materials.sort(key=natural_keys)

    rows = []

    for material in materials:
        model_key = MODEL_NAME_MAP.get(material, material)

        model_path = os.path.join(
            MODEL_DIR,
            f"monotone_bernstein_model_particle2ae_{model_key}_exp2.joblib",
        )
        if not os.path.exists(model_path):
            print(f"Warning: monotone_bernstein model not found for {material}: {model_path}")
            continue
        model, _ = load_model(model_path)

        print(f"\n[{material}] monotone_bernstein model loaded.")

        target_files_sample = glob.glob(os.path.join(PSD_BASE_PATH, material, "1st", "*.csv"))
        targets = []
        for f in target_files_sample:
            m = re.search(r"_for_?(\d+)um", os.path.basename(f))
            if m:
                targets.append(int(m.group(1)))
        targets = sorted(set(targets))

        for target_val in targets:
            theta_ae = threshold_from_target_d50(model, float(target_val))
            print(f"  Target={target_val} um -> theta_AE={theta_ae:.4f} mV²")

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
                ae_files = [f for f in ae_candidates if re.search(f"_for_?{target_val}um", os.path.basename(f))]
                ae_files.sort(key=parse_timestamp_key)

                raw_series = []
                for f in ae_files:
                    p = get_cached_ae_power(f)
                    if p is None or not np.isfinite(p):
                        continue
                    raw_series.append(float(p) * AE_SCALE_TO_MV2)

                n_cycles = len(raw_series)

                if n_cycles < MOVING_AVG_WINDOW:
                    print(f"    {trial}: insufficient length ({n_cycles} < {MOVING_AVG_WINDOW}), skipping")
                    rows.append(
                        {
                            "Material": material,
                            "Target_D50": target_val,
                            "Trial": trial,
                            "theta_bernstein": theta_ae,
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
                else:
                    cross_type = "earlier_cross"

                trial_data[trial] = {
                    "smoothed": smoothed,
                    "k_star": k_star,
                    "reachable": reachable,
                }

                rows.append(
                    {
                        "Material": material,
                        "Target_D50": target_val,
                        "Trial": trial,
                        "theta_bernstein": theta_ae,
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

                status = f"k*={k_star}" if reachable else "unreach"
                print(f"    {trial}: n={n_cycles}, {status}")

            if not trial_data:
                continue

            fig, ax = plt.subplots(figsize=(12, 8))

            for trial in ["1st", "2nd", "3rd"]:
                td = trial_data.get(trial)
                if td is None:
                    continue
                smoothed = td["smoothed"]
                x_idx = np.arange(MOVING_AVG_WINDOW - 1, MOVING_AVG_WINDOW - 1 + len(smoothed))

                ax.plot(
                    x_idx,
                    smoothed,
                    color=TRIAL_COLORS[trial],
                    marker=TRIAL_MARKERS[trial],
                    markersize=8,
                    linewidth=1.5,
                    label=trial,
                )

                if td["reachable"]:
                    ax.axvline(
                        td["k_star"],
                        color=TRIAL_COLORS[trial],
                        linestyle="-",
                        linewidth=2,
                        alpha=0.6,
                    )
                else:
                    ax.annotate(
                        "unreach",
                        xy=(x_idx[-1], smoothed[-1]),
                        fontsize=14,
                        color=TRIAL_COLORS[trial],
                        ha="left",
                        va="bottom",
                    )

            ax.axhline(
                theta_ae,
                color="green",
                linestyle="--",
                linewidth=2,
                label=r"$\theta_{\mathrm{AE}}$" + f" = {theta_ae:.2f}",
            )

            ax.set_xlabel("Cycle index")
            ax.set_ylabel(r"Smoothed AE power ($\mathrm{mV}^2$)")
            ax.set_title(f"{material}, Target = {target_val} " + r"$\mathrm{\mu m}$")
            ax.legend(loc="best")

            base_name = f"exp3_monotone_bernstein_pseudostop_{material}_{target_val}um"
            fig.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {base_name}.png")

    if CACHE_DIRTY:
        save_ae_power_cache(CACHE_FILE, AE_POWER_CACHE)
        print("\nAE power cache updated.")

    if rows:
        df = pd.DataFrame(rows)
        col_order = [
            "Material",
            "Target_D50",
            "Trial",
            "theta_bernstein",
            "reachable",
            "cross_type",
            "k_star",
            "S_AE_at_k_star",
            "min_S_AE",
            "margin",
            "n_cycles",
            "Measured_D50",
        ]
        df = df[[c for c in col_order if c in df.columns]]
        csv_path = os.path.join(OUTPUT_DIR, "exp3_monotone_bernstein_pseudostop_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSummary CSV saved to: {csv_path}")
        print(df.to_string(index=False))
    else:
        print("\nNo data processed.")


if __name__ == "__main__":
    main()
