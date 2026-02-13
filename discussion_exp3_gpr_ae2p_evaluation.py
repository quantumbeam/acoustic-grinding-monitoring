import os
from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Match figure typography to other discussion plots
plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 32,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 18,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
})

RESULTS_DIR = "results"
INPUT_CSV = os.path.join(RESULTS_DIR, "exp3_evaluation_detail.csv")
DISCUSSION_DIR = os.path.join(RESULTS_DIR, "discussion")
M_BOUNDARY = 1.96

PLOT_LABEL_MAP = {
    "MSG": "MSG",
}


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input file not found: {INPUT_CSV}")
        print("Run exp3_gpr_control_analysis.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        print(f"Input file is empty: {INPUT_CSV}")
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
        print(f"Missing required columns in {INPUT_CSV}: {missing}")
        return

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
