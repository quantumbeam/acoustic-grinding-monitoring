import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import joblib
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import r2_score

# Style setting (LaTeX-less environment)
plt.style.use(['science', 'ieee', 'no-latex'])


def load_dataset(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run exp2_gpr_model.py first to generate model_comparison/gpr/exp2_gpr_dataset_raw.csv."
        )
    return pd.read_csv(dataset_path)


def filter_dataset(df: pd.DataFrame, reagent: str, trial: str) -> pd.DataFrame:
    if reagent != "all":
        df = df[df["reagent"] == reagent]
    if trial != "all":
        df = df[df["trial"] == trial]
    return df


def fit_isotonic(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> IsotonicRegression:
    """
    Fit isotonic regression with optional sample weights.
    increasing="auto" lets the solver pick increasing/decreasing based on data.
    """
    model = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    model.fit(x, y, sample_weight=sample_weight)
    return model


def compute_weights_knn(x_sorted: np.ndarray, y_sorted: np.ndarray, k: int = 5, eps: float = 1e-12) -> np.ndarray:
    """
    Heteroscedasticity-aware weights via local variance estimated from k-nearest neighbors in x-space.
    weight_i = 1 / (Var(y in local neighborhood) + eps)

    Notes:
    - O(n^2) but n is small in your dataset, so fine.
    - Uses symmetric window in sorted order (fast & stable).
    """
    n = x_sorted.size
    k = int(max(3, min(k, n)))  # at least 3, at most n
    half = k // 2

    weights = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + k)
        lo = max(0, hi - k)

        y_local = y_sorted[lo:hi]
        if y_local.size <= 1:
            var = 0.0
        else:
            var = float(np.var(y_local, ddof=1))
        weights[i] = 1.0 / (var + eps)

    # Normalize for numerical stability (optional)
    weights /= np.mean(weights)
    return weights


def compute_weights_bin(x_sorted: np.ndarray, y_sorted: np.ndarray, n_bins: int = 6, eps: float = 1e-12) -> np.ndarray:
    """
    Bin-based weights: estimate variance of y within x-bins.
    weight(bin) = 1 / (Var(y in bin) + eps)
    """
    n = x_sorted.size
    n_bins = int(max(2, min(n_bins, n)))
    edges = np.quantile(x_sorted, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        var = float(np.var(y_sorted, ddof=1)) if n > 1 else 0.0
        w = np.full(n, 1.0 / (var + eps), dtype=float)
        w /= np.mean(w)
        return w

    bin_id = np.digitize(x_sorted, edges[1:-1], right=True)
    weights = np.zeros(n, dtype=float)
    for b in np.unique(bin_id):
        idx = np.where(bin_id == b)[0]
        yb = y_sorted[idx]
        var = float(np.var(yb, ddof=1)) if yb.size > 1 else 0.0
        w = 1.0 / (var + eps)
        weights[idx] = w

    weights /= np.mean(weights)
    return weights


def unique_xy_mean_y(x_sorted: np.ndarray, y_sorted: np.ndarray):
    """
    Return strictly increasing unique x, with y averaged over duplicate x.
    """
    unique_x, inv = np.unique(x_sorted, return_inverse=True)
    y_sum = np.zeros_like(unique_x, dtype=float)
    y_cnt = np.zeros_like(unique_x, dtype=float)
    for i, g in enumerate(inv):
        y_sum[g] += y_sorted[i]
        y_cnt[g] += 1.0
    y_unique = y_sum / np.maximum(y_cnt, 1.0)
    return unique_x, y_unique


def build_isotonic_pchip(
    iso: IsotonicRegression,
    x_sorted: np.ndarray,
    y_sorted: np.ndarray,
    extrapolate: bool = False,
):
    """
    Smooth an isotonic fit by applying PCHIP interpolation to the isotonic *outputs*.
    Steps:
      1) Build strictly increasing knot x (unique_x) from data.
      2) Evaluate isotonic at knot x to get monotone y_knots.
      3) Fit PCHIP on (x_knots, y_knots).
    """
    x_knots, _ = unique_xy_mean_y(x_sorted, y_sorted)
    if x_knots.size < 2:
        return None, None, None

    y_knots = iso.predict(x_knots)

    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=extrapolate)
    return pchip, x_knots, y_knots


def predict_pchip(
    pchip: PchipInterpolator,
    x_plot: np.ndarray,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    extrapolate: bool = False,
):
    y_smooth = pchip(x_plot)
    if not extrapolate:
        y_left = y_knots[0]
        y_right = y_knots[-1]
        y_smooth = np.where(x_plot < x_knots[0], y_left, y_smooth)
        y_smooth = np.where(x_plot > x_knots[-1], y_right, y_smooth)
    return y_smooth


def plot_models(
    x: np.ndarray,
    y: np.ndarray,
    trial_labels: np.ndarray,
    direction: str,
    reagent: str,
    outdir: str,
    model_dir: str,
    experiment: str,
    weight_method: str = "knn",
    knn_k: int = 5,
    n_bins: int = 6,
    extrapolate_pchip: bool = False,
    save_models: bool = True,
):
    if x.size < 2:
        print(f"Skip {direction} / {reagent}: not enough points.")
        return

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    trials_sorted = trial_labels[order]

    x_plot = np.linspace(x_sorted.min() * 0.9, x_sorted.max() * 1.1, 500)

    # Unweighted isotonic (for PCHIP smoothing)
    iso = fit_isotonic(x_sorted, y_sorted)
    pchip_iso, xk_iso, yk_iso = build_isotonic_pchip(
        iso, x_sorted, y_sorted, extrapolate=extrapolate_pchip
    )
    if pchip_iso is None:
        print(f"Skip {direction} / {reagent}: not enough unique x for PCHIP.")
        return []
    y_iso_smooth = predict_pchip(pchip_iso, x_plot, xk_iso, yk_iso, extrapolate=extrapolate_pchip)

    # Weighted isotonic (for PCHIP smoothing)
    if weight_method == "knn":
        w = compute_weights_knn(x_sorted, y_sorted, k=knn_k)
    elif weight_method == "bin":
        w = compute_weights_bin(x_sorted, y_sorted, n_bins=n_bins)
    else:
        raise ValueError(f"Unknown weight_method: {weight_method} (use 'knn' or 'bin')")

    wiso = fit_isotonic(x_sorted, y_sorted, sample_weight=w)
    pchip_wiso, xk_wiso, yk_wiso = build_isotonic_pchip(
        wiso, x_sorted, y_sorted, extrapolate=extrapolate_pchip
    )
    if pchip_wiso is None:
        print(f"Skip {direction} / {reagent}: not enough unique x for PCHIP.")
        return []
    y_wiso_smooth = predict_pchip(pchip_wiso, x_plot, xk_wiso, yk_wiso, extrapolate=extrapolate_pchip)

    markers = {'1st': 'o', '2nd': 'x', '3rd': '^'}
    colors = {'1st': 'black', '2nd': 'red', '3rd': 'blue'}

    plt.figure(figsize=(12, 8))
    for t in np.unique(trials_sorted):
        m = trials_sorted == t
        plt.scatter(
            x_sorted[m],
            y_sorted[m],
            marker=markers.get(t, 'o'),
            c=colors.get(t, 'black'),
            s=100,
            label=t
        )

    # Smoothed monotone curves: PCHIP applied to isotonic outputs
    plt.plot(x_plot, y_iso_smooth, 'k--', label='Isotonic + PCHIP (smooth)')
    plt.plot(x_plot, y_wiso_smooth, 'k:', label=f'Weighted isotonic + PCHIP ({weight_method})')

    if direction == "particle2ae":
        plt.xlabel(r'$D_{50}~(\mathrm{\mu m})$')
        plt.ylabel(r'Total spectral power ($\mathrm{mV}^2$)')
    else:
        plt.xlabel(r'Total spectral power ($\mathrm{mV}^2$)')
        plt.ylabel(r'$D_{50}~(\mathrm{\mu m})$')

    plt.legend()

    os.makedirs(outdir, exist_ok=True)
    base = f"{experiment}_isotonic_weighted_pchip_plot_{direction}_{reagent}"
    plt.savefig(os.path.join(outdir, f"{base}.png"), dpi=300)
    plt.close()

    print(f"Saved: {base}.png")

    metrics = []
    # Evaluate on training x (smoothed predictions)
    y_pred_iso = predict_pchip(pchip_iso, x_sorted, xk_iso, yk_iso, extrapolate=extrapolate_pchip)
    y_pred_wiso = predict_pchip(pchip_wiso, x_sorted, xk_wiso, yk_wiso, extrapolate=extrapolate_pchip)

    metrics.append({
        "method": "isotonic_pchip",
        "direction": direction,
        "reagent": reagent,
        "r_squared": float(r2_score(y_sorted, y_pred_iso)),
        "weight_method": "none",
        "model_path": os.path.join(
            model_dir, f"isotonic_pchip_model_{direction}_{reagent}_{experiment}.joblib"
        )
    })
    metrics.append({
        "method": "weighted_isotonic_pchip",
        "direction": direction,
        "reagent": reagent,
        "r_squared": float(r2_score(y_sorted, y_pred_wiso)),
        "weight_method": weight_method,
        "model_path": os.path.join(
            model_dir, f"weighted_isotonic_pchip_model_{direction}_{reagent}_{experiment}.joblib"
        )
    })

    if save_models:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump({
            "experiment": experiment,
            "direction": direction,
            "reagent": reagent,
            "method": "isotonic_pchip",
            "extrapolate": extrapolate_pchip,
            "isotonic": iso,
            "pchip": pchip_iso,
            "x_knots": xk_iso,
            "y_knots": yk_iso,
        }, metrics[0]["model_path"])
        joblib.dump({
            "experiment": experiment,
            "direction": direction,
            "reagent": reagent,
            "method": "weighted_isotonic_pchip",
            "weight_method": weight_method,
            "extrapolate": extrapolate_pchip,
            "isotonic": wiso,
            "pchip": pchip_wiso,
            "x_knots": xk_wiso,
            "y_knots": yk_wiso,
        }, metrics[1]["model_path"])

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot isotonic/weighted isotonic and their PCHIP-smoothed monotone curves for exp2 dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join("model_comparison", "gpr", "exp2_gpr_dataset_raw.csv"),
        help="Path to dataset CSV produced by exp2_gpr_model.py"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("model_comparison", "monotonic"),
        help="Directory to save plots"
    )
    parser.add_argument(
        "--reagent",
        type=str,
        default="all",
        choices=["NaCl", "Citricacid", "MSG", "all"]
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="all",
        choices=["1st", "2nd", "3rd", "all"]
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="all",
        choices=["particle2ae", "ae2particle", "all"]
    )
    parser.add_argument(
        "--weight_method",
        type=str,
        default="knn",
        choices=["knn", "bin"],
        help="Weight estimation method for weighted isotonic."
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="Number of neighbors for knn-based local variance (weight_method=knn)."
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=6,
        help="Number of bins for bin-based variance (weight_method=bin)."
    )
    parser.add_argument(
        "--extrapolate_pchip",
        action="store_true",
        help="Allow PCHIP to extrapolate outside knot range (not recommended for paper figures)."
    )
    parser.add_argument(
        "--no_save_models",
        action="store_true",
        help="Do not save isotonic+PCHIP models to joblib."
    )

    args = parser.parse_args()

    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 32,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 18,
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans'
    })

    EXPERIMENT = "exp2"
    df = load_dataset(args.dataset)
    df = filter_dataset(df, args.reagent, args.trial)

    if df.empty:
        print("No data after filtering.")
        raise SystemExit(0)

    directions = ["particle2ae", "ae2particle"] if args.direction == "all" else [args.direction]

    all_metrics = []
    for reagent in sorted(df["reagent"].unique()):
        df_r = df[df["reagent"] == reagent]
        for direction in directions:
            if direction == "particle2ae":
                x = df_r["d50"].to_numpy(dtype=float)
                y = df_r["ae_power_mV2"].to_numpy(dtype=float)
            else:
                x = df_r["ae_power_mV2"].to_numpy(dtype=float)
                y = df_r["d50"].to_numpy(dtype=float)

            trial_labels = df_r["trial"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)

            metrics = plot_models(
                x[mask],
                y[mask],
                trial_labels[mask],
                direction,
                reagent,
                args.outdir,
                args.outdir,
                EXPERIMENT,
                weight_method=args.weight_method,
                knn_k=args.knn_k,
                n_bins=args.n_bins,
                extrapolate_pchip=args.extrapolate_pchip,
                save_models=not args.no_save_models,
            )
            all_metrics.extend(metrics)

    if all_metrics:
        metrics_path = os.path.join(args.outdir, f"{EXPERIMENT}_isotonic_pchip_metrics.csv")
        pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")
