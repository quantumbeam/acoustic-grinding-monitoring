import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Fit degree-4 polynomial models (particle2ae direction) on exp2 dataset.

For each material, fits AE_power_mV2 = a*D50^4 + b*D50^3 + c*D50^2 + d*D50 + e
and saves the model (np.poly1d coefficients) as joblib.

Output: model_comparison/poly4/
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import joblib
from sklearn.metrics import r2_score

# ============================================================
# Style
# ============================================================
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 32,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 18,
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',
})

# ============================================================
# Constants
# ============================================================
DATASET_PATH = os.path.join('model_comparison', 'gpr', 'exp2_gpr_dataset_raw.csv')
OUTPUT_DIR = os.path.join('model_comparison', 'poly4')
DEGREE = 4

TRIAL_COLORS = {'1st': 'black', '2nd': 'red', '3rd': 'blue'}
TRIAL_MARKERS = {'1st': 'o', '2nd': 'x', '3rd': '^'}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. "
            "Run exp2_gpr_model.py first."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f"--- Poly4 Model Fitting (degree={DEGREE}) ---")
    print(f"    Dataset: {len(df)} rows")

    metrics_rows = []

    for reagent in sorted(df["reagent"].unique()):
        df_r = df[df["reagent"] == reagent]
        x = df_r["d50"].to_numpy(dtype=float)
        y = df_r["ae_power_mV2"].to_numpy(dtype=float)
        trials = df_r["trial"].to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)
        x, y, trials = x[mask], y[mask], trials[mask]

        if x.size < DEGREE + 1:
            print(f"[{reagent}] Not enough data ({x.size}), skipping.")
            continue

        # Fit polynomial
        coeffs = np.polyfit(x, y, DEGREE)
        poly = np.poly1d(coeffs)

        y_pred = poly(x)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        print(f"\n[{reagent}] R²={r2:.4f}, RMSE={rmse:.2f} mV²")
        print(f"    coeffs: {coeffs}")

        # Save model
        model_path = os.path.join(
            OUTPUT_DIR, f"poly4_model_particle2ae_{reagent}_exp2.joblib"
        )
        joblib.dump({
            "experiment": "exp2",
            "direction": "particle2ae",
            "reagent": reagent,
            "method": f"poly{DEGREE}",
            "degree": DEGREE,
            "coefficients": coeffs,
            "x_range": [float(x.min()), float(x.max())],
        }, model_path)
        print(f"    Saved: {model_path}")

        metrics_rows.append({
            "reagent": reagent,
            "direction": "particle2ae",
            "degree": DEGREE,
            "r_squared": r2,
            "rmse": rmse,
            "n_samples": x.size,
            "model_path": model_path,
        })

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 8))

        for t in sorted(np.unique(trials)):
            m = trials == t
            ax.scatter(
                x[m], y[m],
                marker=TRIAL_MARKERS.get(t, 'o'),
                c=TRIAL_COLORS.get(t, 'black'),
                s=100,
                label=t,
            )

        x_plot = np.linspace(x.min() * 0.9, x.max() * 1.1, 500)
        y_plot = poly(x_plot)
        ax.plot(x_plot, y_plot, 'k--', linewidth=2,
                label=f'Poly4 ($R^2$={r2:.3f})')

        ax.set_xlabel(r'$D_{50}~(\mathrm{\mu m})$')
        ax.set_ylabel(r'Total spectral power ($\mathrm{mV}^2$)')
        ax.set_title(f'{reagent} — particle2ae')
        ax.legend(loc='best')

        base = f"exp2_poly4_particle2ae_{reagent}"
        fig.savefig(os.path.join(OUTPUT_DIR, f"{base}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Plot: {base}.png")

    # Save metrics
    if metrics_rows:
        df_m = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join(OUTPUT_DIR, "exp2_poly4_metrics.csv")
        df_m.to_csv(metrics_path, index=False)
        print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
