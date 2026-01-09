#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot last-5 AE points before stopping vs A_AE_Threshold for all materials/targets/trials.

- Uses scienceplots style (science, ieee, no-latex) for publication-ready figures.
- Saves both .pdf and .pdf to results/discussion/plots_last5/

Assumptions:
- fft_processing.py provides calculate_fft_power(path)->float
- AE files: ae_data/exp3/<Material>/<Trial>/*.csv
- Filenames include target tag like "..._for_100um..." (flexible matching)
- results/exp3_evaluation_detail.csv has: Material, Target_D50, Trial, A_AE_Threshold
"""

import os
import re
import glob
import math
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# scienceplots: pip install SciencePlots
import scienceplots  # noqa: F401

from fft_processing import calculate_fft_power  # must exist in your local project


# ----------------------
# Global plot style (TeX-free)
# ----------------------
plt.style.use(["science", "ieee", "no-latex"])

# Slightly tune defaults (optional but helps consistency)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": False,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
})


# ----------------------
# User config
# ----------------------
RESULTS_DIR = Path("results")
DETAIL_CSV = RESULTS_DIR / "exp3_evaluation_detail.csv"

AE_BASE_DIR = Path("ae_data")
EXP_TAG = "exp3"

AE_SCALE_TO_MV2 = 1e6   # convert to mV^2 if calculate_fft_power returns V^2
LAST_N = 5

OUT_DIR = RESULTS_DIR / "discussion" / "plots_last5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map CSV material names to folder names (adjust to your directory names)
MATERIAL_DIR_MAP = {
    "NaCl": "NaCl",
    "Citric Acid": "Citric_Acid",  # if your folder has spaces, set "Citric Acid": "Citric Acid"
    "Ajinomoto": "Ajinomoto",
    "MSG": "Ajinomoto",
}


# ----------------------
# Helpers
# ----------------------
def safe_material_dir_name(material: str) -> str:
    if material in MATERIAL_DIR_MAP:
        return MATERIAL_DIR_MAP[material]
    return material.replace(" ", "_")


def parse_sort_key(path: str):
    """Sort by timestamp prefix if present: YYYYMMDD_HHMMSS in basename."""
    base = os.path.basename(path)
    m = re.search(r"(\d{8})_(\d{6})", base)
    if m:
        return (m.group(1), m.group(2), base)
    return ("", "", base)


def find_ae_files(material: str, trial: str, target_um: int):
    """Find AE files for given material/trial/target. Be permissive about naming."""
    mat_dir = safe_material_dir_name(material)
    ae_dir = AE_BASE_DIR / EXP_TAG / mat_dir / trial
    if not ae_dir.is_dir():
        return []

    patterns = [
        str(ae_dir / f"*for*{target_um}um*.csv"),
        str(ae_dir / f"*{target_um}um*.csv"),
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    files = sorted(list(set(files)), key=parse_sort_key)
    return files


def compute_last_n_ae_powers(files, last_n=5):
    """
    Compute AE power (mV^2) for last_n files.
    Skips invalid values but preserves chronological order within the tail window.
    """
    tail = files[-last_n:] if len(files) >= last_n else files[:]
    vals = []
    kept_files = []

    for f in tail:
        p = calculate_fft_power(f)
        if p is None:
            continue
        if isinstance(p, float) and (math.isnan(p) or math.isinf(p)):
            continue

        vals.append(float(p) * AE_SCALE_TO_MV2)
        kept_files.append(os.path.basename(f))

    return np.array(vals, dtype=float), kept_files


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)


# ----------------------
# Main
# ----------------------
df = pd.read_csv(DETAIL_CSV)

required_cols = {"Material", "Target_D50", "Trial", "A_AE_Threshold"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {DETAIL_CSV}: {missing}")

# Iterate per (material, target)
group_keys = (
    df[["Material", "Target_D50"]]
    .drop_duplicates()
    .sort_values(["Material", "Target_D50"])
)

print(f"Found {len(group_keys)} (Material, Target) groups.")

for _, r in group_keys.iterrows():
    material = str(r["Material"])
    target = int(r["Target_D50"])

    sub = df[(df["Material"] == material) & (df["Target_D50"] == target)]
    if sub.empty:
        continue

    thr_vals = sub["A_AE_Threshold"].dropna().unique()
    if len(thr_vals) == 0:
        print(f"[WARN] No threshold for {material} target {target}. Skipping.")
        continue
    if len(thr_vals) > 1:
        print(f"[WARN] Multiple thresholds for {material} target {target}: {thr_vals}. Using the first.")
    threshold = float(thr_vals[0])

    trials = sorted(sub["Trial"].dropna().unique().tolist())

    # Figure: one per (material,target)
    fig, ax = plt.subplots(figsize=(3.45, 2.35))  # ieee-ish single-column friendly

    plotted_any = False
    max_len = 0

    for trial in trials:
        files = find_ae_files(material, trial, target)
        if not files:
            print(f"[WARN] No AE files found for {material} target {target} trial {trial}")
            continue

        vals, _kept = compute_last_n_ae_powers(files, last_n=LAST_N)
        if len(vals) == 0:
            print(f"[WARN] No valid AE powers computed for {material} target {target} trial {trial}")
            continue

        x = np.arange(1, len(vals) + 1)  # 1..N (older -> newer within last window)
        ax.plot(x, vals, marker="o", linewidth=1.2, markersize=3.0, label=str(trial))
        plotted_any = True
        max_len = max(max_len, len(vals))

    # Threshold line
    ax.axhline(threshold, linestyle="--", linewidth=1.2, label="Threshold")

    # Labels (TeX-free)
    ax.set_xlabel(f"Last {LAST_N} cycles (older → newer)")
    ax.set_ylabel("AE power (mV^2)")

    # X ticks: only show available indices
    if max_len > 0:
        ax.set_xticks(np.arange(1, max_len + 1))

    # Legend: keep compact
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout(pad=0.25)

    if plotted_any:
        stem = f"last{LAST_N}_{sanitize(material)}_{target}um"
        out_png = OUT_DIR / f"{stem}.png"
        out_pdf = OUT_DIR / f"{stem}.pdf"
        fig.savefig(out_png)
        fig.savefig(out_pdf)
        print(f"[OK] Saved: {out_png} and {out_pdf}")
    else:
        print(f"[WARN] Nothing plotted for {material} target {target}")

    plt.close(fig)

print("Done.")
