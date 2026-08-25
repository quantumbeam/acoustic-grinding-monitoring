"""ESI Table S1: comparison of D50, the mean and the mode as the representative diameter.

Section 2.3 of the manuscript states that the mean and $D_{50}$ show broadly
comparable relationships with the AE feature whereas the mode is unstable, and
cites Table S1 for it.  That table was missing from the submitted ESI; this
script builds it from the evaluation already produced by
``run_12_evaluate_representative_diameters.py``.

For every size-control condition of exp3 the AE-to-particle model is evaluated
against the measured powder, once per candidate metric.  The quantity tabulated
is the relative estimation error of that model,

    100 * (measured - predicted) / measured,

taken from
``analysis_results/run_12_evaluate_representative_diameters/<metric>/evaluation_summary_for_table.csv``.
The mode is unusable for citric acid at the finest target because the dominant
peak of a multimodal distribution switches between runs, which is the behaviour
Section 2.3 describes.

Outputs (analysis_results/run_13_build_representative_diameter_table/):
    table_s1_representative_diameter.csv    the tabulated values
    table_s1_representative_diameter.tex    ready to paste into the ESI
"""

import os

import numpy as np
import pandas as pd

METRICS = ["D50", "Dmean", "Dmode"]
METRIC_HEADERS = {"D50": "$D_{50}$", "Dmean": "Mean", "Dmode": "Mode"}
MATERIAL_ORDER = ["NaCl", "Citricacid", "MSG"]
MATERIAL_LABELS = {"NaCl": "NaCl", "Citricacid": "Citric acid", "MSG": "MSG"}

INPUT_DIR = os.path.join("analysis_results", "run_12_evaluate_representative_diameters")
OUTPUT_DIR = os.path.join("analysis_results", "run_13_build_representative_diameter_table")
ERROR_COLUMN = "AE2P_Estimation_Error_Percent"


def load(metric):
    path = os.path.join(INPUT_DIR, metric, "evaluation_summary_for_table.csv")
    frame = pd.read_csv(path)
    frame["metric"] = metric
    return frame[["Material", "Target_Size", "metric", ERROR_COLUMN]]


def build():
    frames = pd.concat([load(metric) for metric in METRICS], ignore_index=True)
    table = frames.pivot_table(
        index=["Material", "Target_Size"], columns="metric", values=ERROR_COLUMN
    )[METRICS]
    table = table.reindex(
        pd.MultiIndex.from_tuples(
            [
                (material, size)
                for material in MATERIAL_ORDER
                for size in sorted(
                    frames.loc[frames["Material"] == material, "Target_Size"].unique()
                )
            ],
            names=["Material", "Target_Size"],
        )
    )
    return table


def to_latex(table):
    lines = []
    for material in MATERIAL_ORDER:
        if lines:
            lines.append("    \\addlinespace")
        block = table.loc[material]
        first = True
        for size, row in block.iterrows():
            label = MATERIAL_LABELS[material] if first else ""
            first = False
            cells = " & ".join(f"{row[m]:+.1f}" for m in METRICS)
            lines.append(f"    {label:<11s} & {int(size):>3d} & {cells} \\\\")
    lines.append("    \\midrule")
    median = table.abs().median()
    lines.append(
        "    \\multicolumn{2}{l}{Median absolute error} & "
        + " & ".join(f"{median[m]:.1f}" for m in METRICS)
        + " \\\\"
    )
    body = "\n".join(lines)

    return f"""\\begin{{table}}[htbp]
  \\centering
  \\caption{{
  Comparison of the median diameter, the mean and the mode as the representative
  particle-size metric. For each size-control condition the relative estimation error of
  the AE-to-particle model, $100\\,(\\text{{measured}}-\\text{{predicted}})/\\text{{measured}}$,
  is listed for the three candidate metrics. The mean and $D_{{50}}$ behave comparably,
  whereas the mode is unstable: for citric acid at the \\SI{{20}}{{\\micro\\meter}} target the
  dominant peak of the multimodal distribution switches between runs and the error exceeds
  three orders of magnitude.
  }}
  \\label{{tab:S1_representative_diameter}}
  \\small
  \\setlength{{\\tabcolsep}}{{8pt}}
  \\begin{{tabular}}{{llrrr}}
    \\toprule
    Material & \\makecell{{Target\\\\(\\unit{{\\micro\\meter}})}}
             & \\multicolumn{{3}}{{c}}{{Estimation error (\\unit{{\\percent}})}} \\\\
    \\cmidrule(l){{3-5}}
             & & {" & ".join(METRIC_HEADERS[m] for m in METRICS)} \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def main():
    table = build()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "table_s1_representative_diameter.csv")
    tex_path = os.path.join(OUTPUT_DIR, "table_s1_representative_diameter.tex")
    table.round(2).to_csv(csv_path)
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(to_latex(table))
    print(table.round(1).to_string())
    print("\nmedian absolute error (%):")
    print(table.abs().median().round(1).to_string())
    print(f"\nSaved: {csv_path}\nSaved: {tex_path}")


if __name__ == "__main__":
    main()
