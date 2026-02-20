import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fft_processing import calculate_fft_power
from monotone_bernstein import load_model


@dataclass
class Sample:
    d50: float
    grind_min: float
    trial: str
    reagent: str
    ae_series_mV2: np.ndarray


def norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def get_d50(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Dx (50)"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        return float(parts[1])
    except (OSError, ValueError):
        return None
    return None


def moving_average(data: np.ndarray, window_size: int = 4) -> np.ndarray:
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def load_cache(cache_file: str) -> dict:
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def update_cache_policy_a(cache: dict, cache_file: str, file_paths: list[str]) -> dict:
    """Policy A: cache hit -> no recompute; compute only missing entries."""
    updated = 0
    skipped = 0
    for file_path in file_paths:
        key = norm_path(file_path)
        if key in cache:
            skipped += 1
            continue
        power = calculate_fft_power(file_path)
        if power is None:
            continue
        cache[key] = float(power)
        updated += 1
    if updated:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)
        print(f"Cache updated: {updated} new values (skipped {skipped}).")
        print(f"Cache saved to: {cache_file}")
    else:
        print(f"Cache hit for all required files (skipped {skipped}).")
    return cache


def build_samples(experiment: str, reagent: str, trial: str) -> list[Sample]:
    ae_base_path = os.path.join("data/ae", experiment)
    psd_base_path = os.path.join("data/powder_size_distribution", experiment)

    reagent_pattern = "*" if reagent == "all" else reagent
    trial_pattern = "*" if trial == "all" else trial

    all_psd_files = []
    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, reagent_pattern)):
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, trial_pattern)):
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, "*.csv")))

    required_ae_files = set()
    psd_info = []
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue
        reagent_name = path_parts[-3]
        trial_name = path_parts[-2]

        d50 = get_d50(psd_file)
        if d50 is None:
            continue

        match = re.search(r"(grind\d+min)", os.path.basename(psd_file))
        if not match:
            continue
        grind_key = match.group(1)

        grind_match = re.search(r"grind(\d+)min", grind_key)
        if not grind_match:
            continue
        grind_min = float(grind_match.group(1))

        ae_session_path = os.path.join(ae_base_path, reagent_name, trial_name)
        ae_files = sorted(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))
        if not ae_files:
            continue

        required_ae_files.update(ae_files)
        psd_info.append((reagent_name, trial_name, d50, grind_min, ae_files))

    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ae_power_cache.json")
    cache = load_cache(cache_file)
    cache = update_cache_policy_a(cache, cache_file, list(required_ae_files))

    samples = []
    for reagent_name, trial_name, d50, grind_min, ae_files in psd_info:
        ae_power_timeseries = [cache.get(norm_path(f)) for f in ae_files]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]
        if len(ae_power_timeseries) < 2:
            continue
        ae_power_mV2 = np.array(ae_power_timeseries, dtype=float) * 1e6
        samples.append(
            Sample(
                d50=float(d50),
                grind_min=grind_min,
                trial=trial_name,
                reagent=reagent_name,
                ae_series_mV2=ae_power_mV2,
            )
        )

    return samples


def build_dataset(samples: list[Sample], window_size: int) -> np.ndarray:
    rows = []
    for s in samples:
        smoothed = moving_average(s.ae_series_mV2, window_size=window_size)
        if smoothed.size == 0:
            continue
        final_ae = float(smoothed[-1])
        rows.append((s.d50, final_ae, s.grind_min, s.trial, s.reagent))
    return np.array(rows, dtype=object)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return rmse, mae, r2


def configure_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass

    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "lines.linewidth": 1.2,
            "lines.markersize": 6,
            "legend.frameon": False,
        }
    )


def select_representative_samples(
    samples: list[Sample], reagent: str, trial_pref: str, grind_pref: float
) -> list[Sample]:
    candidates = [s for s in samples if s.reagent == reagent]
    if not candidates:
        return []

    if trial_pref == "all":
        trials = sorted({s.trial for s in candidates})
        selected = []
        for t in trials:
            matches = [s for s in candidates if s.trial == t and int(s.grind_min) == int(grind_pref)]
            if matches:
                selected.append(matches[0])
                continue
            fallback = [s for s in candidates if s.trial == t]
            if fallback:
                selected.append(fallback[0])
        return selected

    preferred = [s for s in candidates if s.trial == trial_pref and int(s.grind_min) == int(grind_pref)]
    if preferred:
        return [preferred[0]]

    preferred_trial = [s for s in candidates if s.trial == trial_pref]
    if preferred_trial:
        return [preferred_trial[0]]

    return [candidates[0]]


def plot_recommended_timeseries(sample: Sample, window_size: int, out_dir: str, base_name: str):
    series = sample.ae_series_mV2
    if len(series) == 0:
        return None, None

    smoothed = moving_average(series, window_size=window_size)
    x_vals = np.arange(1, len(series) + 1)
    smooth_x = np.arange(window_size, len(series) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, series, "o-", color="black", label="Original Data")
    if len(smoothed):
        plt.plot(
            smooth_x,
            smoothed,
            "x--",
            color="red",
            label=f"{window_size}-point Moving Average",
        )
    plt.xlabel("Number of motions")
    plt.ylabel(r"Total spectral power($\mathrm{mV}^2$)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{base_name}.png")
    out_pdf = os.path.join(out_dir, f"{base_name}.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    return out_png, out_pdf


def evaluate_with_model(data_array: np.ndarray, model) -> tuple[float, float, float]:
    d50_vals = np.array(data_array[:, 0], dtype=float)
    ae_vals = np.array(data_array[:, 1], dtype=float)
    y_pred = model.predict(d50_vals)
    return compute_metrics(ae_vals, y_pred)


def evaluate_with_model_map(data_array: np.ndarray, model_map: dict[str, object]) -> tuple[float, float, float]:
    d50_vals = np.array(data_array[:, 0], dtype=float)
    ae_vals = np.array(data_array[:, 1], dtype=float)
    reagent_labels = np.array(data_array[:, 4], dtype=str)

    y_pred = np.zeros_like(ae_vals, dtype=float)
    valid_mask = np.ones_like(ae_vals, dtype=bool)
    for i, (d50, reagent) in enumerate(zip(d50_vals, reagent_labels)):
        model = model_map.get(str(reagent))
        if model is None:
            valid_mask[i] = False
            continue
        y_pred[i] = float(model.predict(np.array([d50], dtype=float))[0])

    if not np.any(valid_mask):
        return float("nan"), float("nan"), float("nan")
    return compute_metrics(ae_vals[valid_mask], y_pred[valid_mask])


def select_window(results: list[dict], rmse_tol: float) -> dict:
    min_rmse = min(row["rmse"] for row in results)
    candidates = [row for row in results if row["rmse"] <= min_rmse * (1.0 + rmse_tol)]
    return min(candidates, key=lambda r: (r["window_size"], r["mae"]))


def load_particle2ae_model(model_dir: str, experiment: str, reagent: str):
    model_path = os.path.join(
        model_dir,
        f"monotone_bernstein_model_bic_particle2ae_{reagent}_{experiment}.joblib",
    )
    if not os.path.exists(model_path):
        return None, None, None

    model, extra = load_model(model_path)
    return model, extra, model_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate moving-average window size using saved exp2 monotone Bernstein BIC models."
        )
    )
    parser.add_argument("--experiment", type=str, default="exp2")
    parser.add_argument("--reagent", type=str, default="all")
    parser.add_argument("--trial", type=str, default="all")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    parser.add_argument("--rmse-tol", type=float, default=0.0)
    parser.add_argument("--model-dir", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default=os.path.join("results", "SI", "moving_average_bernstein"))
    parser.add_argument(
        "--include-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate a combined (all reagents) score in addition to per-reagent.",
    )
    parser.add_argument(
        "--plot-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot RMSE vs window size.",
    )
    parser.add_argument(
        "--plot-timeseries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot time series with recommended moving average.",
    )
    parser.add_argument("--timeseries-trial", type=str, default="all")
    parser.add_argument("--timeseries-grind-min", type=float, default=25)
    args = parser.parse_args()

    samples = build_samples(args.experiment, args.reagent, args.trial)
    if not samples:
        print("No samples found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    reagent_names = sorted({s.reagent for s in samples})
    eval_targets = list(reagent_names)
    if args.include_all:
        eval_targets.append("all")

    summary_rows = []
    recommended_rows = []

    if args.plot_metrics or args.plot_timeseries:
        configure_plot_style()

    model_cache: dict[str, tuple[object, dict, str]] = {}
    for reagent in reagent_names:
        model, extra, model_path = load_particle2ae_model(args.model_dir, args.experiment, reagent)
        if model is not None:
            model_cache[reagent] = (model, extra if isinstance(extra, dict) else {}, model_path)

    for reagent in eval_targets:
        if reagent == "all":
            subset = samples
        else:
            subset = [s for s in samples if s.reagent == reagent]

        if len(subset) < 3:
            print(f"Skip {reagent}: not enough samples.")
            continue

        if reagent != "all" and reagent not in model_cache:
            print(f"Skip {reagent}: model not found in {args.model_dir}.")
            continue

        results = []
        for w in args.window_sizes:
            data_array = build_dataset(subset, window_size=w)
            if data_array.size == 0 or len(data_array) < 3:
                continue

            if reagent == "all":
                model_map = {k: v[0] for k, v in model_cache.items()}
                rmse, mae, r2 = evaluate_with_model_map(data_array, model_map)
                model_path = ";".join(
                    [
                        model_cache[r][2]
                        for r in sorted(model_cache.keys())
                    ]
                )
                model_selection = "bic(per-reagent)"
            else:
                model, extra, model_path = model_cache[reagent]
                rmse, mae, r2 = evaluate_with_model(data_array, model)
                model_selection = extra.get("criterion", "unknown")
            results.append(
                {
                    "reagent": reagent,
                    "window_size": w,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "n_samples": len(data_array),
                    "model_path": model_path,
                    "model_selection": model_selection,
                }
            )

        if not results:
            print(f"No results for {reagent}.")
            continue

        best = select_window(results, args.rmse_tol)
        best_row = best.copy()
        best_row["rmse_tol"] = args.rmse_tol
        recommended_rows.append(best_row)
        summary_rows.extend(results)

        print(f"{reagent}: 推奨N={best['window_size']} (RMSE={best['rmse']:.3f}, MAE={best['mae']:.3f})")

        if args.plot_metrics:
            windows = [row["window_size"] for row in results]
            rmse_vals = [row["rmse"] for row in results]
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(windows, rmse_vals, "o-", color="black")
            ax.set_ylabel("RMSE")
            ax.set_xlabel("Moving average window size")
            ax.axvline(best["window_size"], color="red", linestyle="--", alpha=0.7)
            fig.tight_layout()

            base = f"moving_average_bernstein_eval_{args.experiment}_{reagent}"
            out_png = os.path.join(args.out_dir, f"{base}.png")
            out_pdf = os.path.join(args.out_dir, f"{base}.pdf")
            fig.savefig(out_png, dpi=300)
            fig.savefig(out_pdf)
            plt.close(fig)
            print(f"Saved eval plot: {out_png}")
            print(f"Saved eval plot: {out_pdf}")

        if args.plot_timeseries and reagent != "all":
            selected_samples = select_representative_samples(
                samples,
                reagent,
                args.timeseries_trial,
                args.timeseries_grind_min,
            )
            for sample in selected_samples:
                base = (
                    f"moving_average_bernstein_recommended_{args.experiment}_"
                    f"{reagent}_{sample.trial}_{int(args.timeseries_grind_min)}min"
                )
                out_png, out_pdf = plot_recommended_timeseries(
                    sample,
                    best["window_size"],
                    args.out_dir,
                    base,
                )
                if out_png and out_pdf:
                    print(f"Saved timeseries plot: {out_png}")
                    print(f"Saved timeseries plot: {out_pdf}")

    if not summary_rows:
        print("No evaluation rows were produced.")
        return

    summary_path = os.path.join(args.out_dir, f"moving_average_bernstein_eval_{args.experiment}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved evaluation summary: {summary_path}")

    rec_path = os.path.join(args.out_dir, f"moving_average_bernstein_recommended_{args.experiment}.csv")
    pd.DataFrame(recommended_rows).to_csv(rec_path, index=False)
    print(f"Saved recommendations: {rec_path}")


if __name__ == "__main__":
    main()
