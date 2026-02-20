import argparse
import os

import pandas as pd


DEFAULT_INPUT = os.path.join(
    "results",
    "SI_plots",
    "model_validation",
    "D50",
    "exp2_monotone_bernstein_bic_selection_summary.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join("results", "SI_plots", "model_validation", "table_s2")

TARGET_REAGENTS = ["NaCl", "Citricacid", "MSG"]
MATERIAL_LABEL = {
    "NaCl": "NaCl",
    "Citricacid": "Citric acid",
    "MSG": "MSG",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Table S2 source CSVs (degree selection and IC values) "
            "from monotone Bernstein BIC selection summary."
        )
    )
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    needed_cols = [
        "reagent",
        "direction",
        "aic_best_degree",
        "bic_best_degree",
        "cv_best_degree",
        "aic_best_value",
        "bic_best_value",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    df = df[df["direction"] == "particle2ae"].copy()
    df = df[df["reagent"].isin(TARGET_REAGENTS)].copy()
    df["reagent"] = pd.Categorical(df["reagent"], categories=TARGET_REAGENTS, ordered=True)
    df = df.sort_values("reagent")

    out_base = pd.DataFrame(
        {
            "Material": df["reagent"].map(MATERIAL_LABEL),
            "Degree selected by AIC": df["aic_best_degree"].astype(int),
            "Degree selected by BIC": df["bic_best_degree"].astype(int),
            "Degree selected by CV": df["cv_best_degree"].astype(int),
        }
    )

    out_with_ic = out_base.copy()
    out_with_ic["AIC (at AIC-selected degree)"] = df["aic_best_value"].astype(float)
    out_with_ic["BIC (at BIC-selected degree)"] = df["bic_best_value"].astype(float)

    os.makedirs(args.output_dir, exist_ok=True)
    out_main_path = os.path.join(args.output_dir, "table_s2_degree_selection_main.csv")
    out_ic_path = os.path.join(args.output_dir, "table_s2_degree_selection_with_ic.csv")

    out_base.to_csv(out_main_path, index=False)
    out_with_ic.to_csv(out_ic_path, index=False)

    print(f"Saved: {out_main_path}")
    print(f"Saved: {out_ic_path}")


if __name__ == "__main__":
    main()
