#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a LaTeX-ready CSV for Supplementary Table (Strategy B: AE-to-particle estimation)
by reading the existing summary CSV.

Input:
  results/exp3_evaluation_summary_for_table.csv

Output:
  results/exp3_strategyB_table.csv

Table columns:
  Material, Target, Measured mean, Estimated mean, Estimation error, z_last

Definition:
  Estimation error = Estimated_mean - Measured_mean
  z_last = (Estimated_mean - Target) / sigma_pred
    where sigma_pred is the mean predicted sigma at stopping (num_B_sigma_GPR)
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ----------------------
# User config
# ----------------------
RESULTS_DIR = Path("results")
IN_SUMMARY = RESULTS_DIR / "exp3_evaluation_summary_for_table.csv"
OUT_TABLE = RESULTS_DIR / "discussion_model_prediction_uncertainty.csv"

# Material name mapping for the paper (edit if needed)
MAT_DISPLAY = {
    "NaCl": "NaCl",
    "Citricacid": "Citric Acid",
    "Citric Acid": "Citric Acid",
    "Ajinomoto": "MSG",
    "MSG": "MSG",
}


def main() -> None:
    if not IN_SUMMARY.exists():
        raise FileNotFoundError(f"Input summary CSV not found: {IN_SUMMARY}")

    df = pd.read_csv(IN_SUMMARY)

    # Required columns to compute the table
    required = {
        "Material",
        "Target_D50",
        "num_Common_mu_trial",   # measured mean (numeric)
        "num_B_mu_GPR",          # estimated mean (numeric)
        "num_B_sigma_GPR",       # estimated sigma (numeric)
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {IN_SUMMARY}: {sorted(missing)}")

    # Build table
    out = pd.DataFrame()
    out["Material"] = df["Material"].map(lambda x: MAT_DISPLAY.get(str(x), str(x)))
    out["Target"] = df["Target_D50"].astype(int)

    out["Measured mean"] = pd.to_numeric(df["num_Common_mu_trial"], errors="coerce")
    out["Estimated mean"] = pd.to_numeric(df["num_B_mu_GPR"], errors="coerce")

    # Estimation error: explicitly match your LaTeX definition (Estimated - Measured)
    out["Estimation error"] = out["Estimated mean"] - out["Measured mean"]

    sigma = pd.to_numeric(df["num_B_sigma_GPR"], errors="coerce")
    # z_last = (Estimated - Target) / sigma
    # Protect against 0/NaN sigma
    out["z_last"] = np.where(
        (sigma.notna()) & (sigma.abs() > 1e-12) & out["Estimated mean"].notna(),
        (out["Estimated mean"] - out["Target"]) / sigma,
        np.nan,
    )

    # Optional rounding to resemble your LaTeX example
    out["Measured mean"] = out["Measured mean"].round(2)
    out["Estimated mean"] = out["Estimated mean"].round(1)
    out["Estimation error"] = out["Estimation error"].round(2)
    out["z_last"] = out["z_last"].round(2)

    # Sort in the same order as the LaTeX table (NaCl -> Citric Acid -> MSG, and targets descending or as you prefer)
    material_order = {"NaCl": 0, "Citric Acid": 1, "MSG": 2}
    out["_mat_order"] = out["Material"].map(lambda x: material_order.get(x, 999))
    out = out.sort_values(["_mat_order", "Target"], ascending=[True, False]).drop(columns=["_mat_order"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_TABLE, index=False)

    print(f"[OK] Saved: {OUT_TABLE}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
