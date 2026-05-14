import argparse
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from ae_fft import calculate_fft_power
from bernstein import BernsteinMonotoneConfig, BernsteinMonotoneRegressor, save_model

plt.style.use(["science", "ieee", "no-latex"])
EXPORT_DPI = 600
RUN_OUTPUT_DIR = os.path.join("analysis_results", "run_03_train_particle_size_ae_model")


def norm_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def update_ae_cache(cache_file_path: str, required_files: list[str]) -> dict:
    required_files = sorted(required_files)
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                ae_cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            ae_cache = {}
    else:
        ae_cache = {}

    missing_files = [file_path for file_path in required_files if norm_path(file_path) not in ae_cache]
    print(
        f"[1/4] AE power cache: {len(required_files)} files required, "
        f"{len(missing_files)} files to compute.",
        flush=True,
    )

    updated_count = 0
    for i, file_path in enumerate(missing_files, start=1):
        key = norm_path(file_path)
        new_power = calculate_fft_power(file_path)
        if new_power is None:
            continue
        ae_cache[key] = float(new_power)
        updated_count += 1
        if i == 1 or i == len(missing_files) or i % 25 == 0:
            print(f"      computed AE power {i}/{len(missing_files)}", flush=True)

    if updated_count > 0:
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(ae_cache, f, indent=2)
        print(f"      cache updated: {updated_count} new values saved.", flush=True)
    else:
        print("      cache already complete.", flush=True)
    return ae_cache


def read_distribution(file_path: str) -> tuple[list[float], list[float]]:
    sizes = []
    volumes = []
    in_table = False
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not in_table:
                    if line.strip().startswith("SizeClasses"):
                        in_table = True
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    size = float(parts[0])
                    volume = float(parts[1])
                except ValueError:
                    continue
                if np.isfinite(size) and np.isfinite(volume):
                    sizes.append(size)
                    volumes.append(volume)
    except OSError:
        return [], []
    return sizes, volumes


def compute_metric_from_distribution(metric: str, sizes: list[float], volumes: list[float]) -> float | None:
    if not sizes or not volumes:
        return None
    size_arr = np.array(sizes, dtype=float)
    vol_arr = np.array(volumes, dtype=float)
    total = float(np.sum(vol_arr))
    if total <= 0.0:
        return None
    if metric == "D50":
        cum = np.cumsum(vol_arr)
        return float(np.interp(total * 0.5, cum, size_arr))
    if metric == "Dmean":
        return float(np.sum(size_arr * vol_arr) / total)
    if metric == "Dmode":
        return float(size_arr[int(np.argmax(vol_arr))])
    return None


def get_metric_value(file_path: str, metric: str) -> float | None:
    sizes, volumes = read_distribution(file_path)
    return compute_metric_from_distribution(metric, sizes, volumes)


def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def infer_monotone_direction(x_data: np.ndarray, y_data: np.ndarray) -> str:
    x = np.asarray(x_data, dtype=float).reshape(-1)
    y = np.asarray(y_data, dtype=float).reshape(-1)
    if x.size < 2:
        return "increasing"
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return "increasing"
    return "increasing" if corr >= 0.0 else "decreasing"


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_shared_dataset(experiment: str, reagent: str, trial: str, metric: str) -> np.ndarray:
    ae_base_path = os.path.join("data", "ae", experiment)
    psd_base_path = os.path.join("data", "powder_size_distribution", experiment)

    print("[1/4] Scanning time-scheduled grinding input files...", flush=True)

    reagent_pattern = reagent if reagent != "all" else "*"
    trial_pattern = trial if trial != "all" else "*"

    all_psd_files = []
    for psd_reagent_dir in glob.glob(os.path.join(psd_base_path, reagent_pattern)):
        for psd_trial_dir in glob.glob(os.path.join(psd_reagent_dir, trial_pattern)):
            all_psd_files.extend(glob.glob(os.path.join(psd_trial_dir, "*.csv")))

    required_ae_files = set()
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue
        reagent_name = path_parts[-3]
        trial_name = path_parts[-2]
        match = re.search(r"(grind\d+min)", os.path.basename(psd_file))
        if not match:
            continue
        ae_session_path = os.path.join(ae_base_path, reagent_name, trial_name)
        required_ae_files.update(glob.glob(os.path.join(ae_session_path, f"*{match.group(1)}*.csv")))

    print(
        f"      found {len(all_psd_files)} particle-size files and "
        f"{len(required_ae_files)} AE files.",
        flush=True,
    )

    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ae_power_cache.json")
    ae_cache = update_ae_cache(cache_file, list(required_ae_files))

    print("[2/4] Building matched particle-size/AE dataset...", flush=True)
    collected_data = []
    for psd_file in all_psd_files:
        path_parts = psd_file.split(os.sep)
        if len(path_parts) < 3:
            continue
        reagent_name = path_parts[-3]
        trial_name = path_parts[-2]
        particle_metric = get_metric_value(psd_file, metric)
        if particle_metric is None:
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

        ae_power_timeseries = [ae_cache.get(norm_path(f)) for f in ae_files]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]
        if len(ae_power_timeseries) < 4:
            continue

        ae_power_mv2 = np.array(ae_power_timeseries, dtype=float) * 1e6
        smoothed = moving_average(ae_power_mv2, window_size=4)
        if smoothed.size == 0:
            continue

        collected_data.append((float(particle_metric), float(smoothed[-1]), grind_min, trial_name, reagent_name))

    if not collected_data:
        raise RuntimeError("No matched data points found.")
    print(f"      matched {len(collected_data)} training data points.", flush=True)
    return np.array(collected_data, dtype=object)


def choose_hyperparams(
    x_data: np.ndarray,
    y_data: np.ndarray,
    groups: np.ndarray,
    monotone: str,
    degree_candidates: list[int],
    lambda_candidates: list[float],
    cv_mode: str,
) -> tuple[int, float, float]:
    if cv_mode == "none":
        return int(degree_candidates[0]), float(lambda_candidates[0]), float("nan")

    unique_groups = np.unique(groups)
    n_splits = min(3, unique_groups.size)
    if n_splits < 2:
        return int(degree_candidates[0]), float(lambda_candidates[0]), float("nan")

    gkf = GroupKFold(n_splits=n_splits)
    best = None
    best_rmse = float("inf")

    for degree in degree_candidates:
        for lambda_smooth in lambda_candidates:
            fold_rmse = []
            failed = False
            for tr_idx, va_idx in gkf.split(x_data, y_data, groups=groups):
                try:
                    model = BernsteinMonotoneRegressor(
                        BernsteinMonotoneConfig(
                            degree=int(degree),
                            monotone=monotone,
                            lambda_smooth=float(lambda_smooth),
                        )
                    ).fit(x_data[tr_idx], y_data[tr_idx])
                except RuntimeError:
                    failed = True
                    break
                y_pred = model.predict(x_data[va_idx])
                fold_rmse.append(float(np.sqrt(mean_squared_error(y_data[va_idx], y_pred))))

            if failed or not fold_rmse:
                continue
            mean_rmse = float(np.mean(fold_rmse))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best = (int(degree), float(lambda_smooth))

    if best is None:
        return int(degree_candidates[0]), float(lambda_candidates[0]), float("nan")
    return best[0], best[1], best_rmse


def compute_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = int(y_true.size)
    if n <= 0:
        return float("nan"), float("nan")

    rss = float(np.sum((y_true - y_pred) ** 2))
    rss = max(rss, 1e-12)
    sigma2 = rss / n

    # Gaussian i.i.d. assumption; constant terms are omitted because they cancel in comparison.
    aic = float(n * np.log(sigma2) + 2.0 * n_params)
    bic = float(n * np.log(sigma2) + np.log(max(n, 2)) * n_params)
    return aic, bic


def fit_score_row(
    x_data: np.ndarray,
    y_data: np.ndarray,
    monotone: str,
    degree: int,
    lambda_smooth: float,
) -> dict:
    try:
        reg = BernsteinMonotoneRegressor(
            BernsteinMonotoneConfig(
                degree=int(degree),
                monotone=monotone,
                lambda_smooth=float(lambda_smooth),
            )
        ).fit(x_data, y_data)
        y_pred = reg.predict(x_data)
        n_params = int(degree) + 1
        aic, bic = compute_aic_bic(y_data, y_pred, n_params=n_params)
        return {
            "degree": int(degree),
            "lambda_smooth": float(lambda_smooth),
            "valid": True,
            "n_params": n_params,
            "rmse_train": float(np.sqrt(mean_squared_error(y_data, y_pred))),
            "mae_train": float(mean_absolute_error(y_data, y_pred)),
            "r2_train": float(r2_score(y_data, y_pred)),
            "aic": float(aic),
            "bic": float(bic),
        }
    except RuntimeError:
        return {
            "degree": int(degree),
            "lambda_smooth": float(lambda_smooth),
            "valid": False,
            "n_params": int(degree) + 1,
            "rmse_train": float("nan"),
            "mae_train": float("nan"),
            "r2_train": float("nan"),
            "aic": float("nan"),
            "bic": float("nan"),
        }


def save_bic_curve_plot(
    curve_df: pd.DataFrame,
    out_path: str,
    title: str,
    cv_best_degree: int | None,
) -> None:
    valid = curve_df[curve_df["valid"]].copy().sort_values("degree")
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(valid["degree"], valid["aic"], "o-", color="black", linewidth=2, markersize=6, label="AIC")
    ax.plot(valid["degree"], valid["bic"], "s-", color="tab:blue", linewidth=2, markersize=6, label="BIC")

    idx_aic = valid["aic"].idxmin()
    idx_bic = valid["bic"].idxmin()
    ax.axvline(int(valid.loc[idx_aic, "degree"]), color="black", linestyle="--", alpha=0.5, label="Best AIC degree")
    ax.axvline(int(valid.loc[idx_bic, "degree"]), color="tab:blue", linestyle="--", alpha=0.5, label="Best BIC degree")
    if cv_best_degree is not None:
        ax.axvline(int(cv_best_degree), color="tab:green", linestyle=":", alpha=0.7, label="Best CV degree")

    ax.set_xlabel("Bernstein degree")
    ax.set_ylabel("Criterion value (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_fit_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    trial_labels: np.ndarray,
    reg: BernsteinMonotoneRegressor,
    x_label: str,
    y_label: str,
    out_path: str,
) -> None:
    markers = {"1st": "o", "2nd": "x", "3rd": "^"}
    colors = {"1st": "black", "2nd": "red", "3rd": "blue"}

    x_plot = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 400)
    y_plot = reg.predict(x_plot)

    plt.figure(figsize=(12, 8))
    for t in np.unique(trial_labels):
        m = trial_labels == t
        plt.scatter(
            x_data[m],
            y_data[m],
            marker=markers.get(t, "o"),
            c=colors.get(t, "black"),
            s=100,
            label=t,
        )
    plt.plot(x_plot, y_plot, "k-", linewidth=2.0, label="Monotone Bernstein fit")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close()


def collect_validation_targets_by_reagent(base_path: str, reagents: list[str]) -> dict[str, list[int]]:
    targets_by_reagent: dict[str, list[int]] = {}
    for reagent in reagents:
        reagent_dir = os.path.join(base_path, reagent)
        if not os.path.isdir(reagent_dir):
            continue

        targets = set()
        for csv_path in glob.glob(os.path.join(reagent_dir, "*", "*.csv")):
            m = re.search(r"_for_?(\d+)um", os.path.basename(csv_path))
            if m:
                targets.add(int(m.group(1)))

        if targets:
            targets_by_reagent[reagent] = sorted(targets)
    return targets_by_reagent


def metric_axis_label(metric: str) -> str:
    if metric == "D50":
        return r"$D_{50}~(\mathrm{\mu m})$"
    if metric == "Dmean":
        return r"$D_{\mathrm{mean}}~(\mathrm{\mu m})$"
    if metric == "Dmode":
        return r"$D_{\mathrm{mode}}~(\mathrm{\mu m})$"
    return r"Particle size $(\mathrm{\mu m})$"


def metric_suffix(metric: str) -> str:
    return "" if metric == "D50" else f"_{metric}"


def resolve_metric_dirs(
    metric: str,
    output_dir: str,
    plot_output_dir: str,
    validation_dir: str,
) -> tuple[str, str, str]:
    if metric == "D50":
        return output_dir, plot_output_dir, validation_dir

    scratch_base = os.path.join("/tmp", "acoustic-powder-monitoring", "bernstein")
    out_dir = os.path.join(scratch_base, metric)
    plot_dir = os.path.join(scratch_base, metric, "plots")
    val_dir = os.path.join(scratch_base, metric, "model_validation")

    return out_dir, plot_dir, val_dir


def run_all_metrics(args: argparse.Namespace) -> None:
    for metric in ["D50", "Dmean", "Dmode"]:
        out_dir, plot_dir, val_dir = resolve_metric_dirs(
            metric=metric,
            output_dir=args.output_dir,
            plot_output_dir=args.plot_output_dir,
            validation_dir=args.validation_dir,
        )
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--reagent",
            str(args.reagent),
            "--trial",
            str(args.trial),
            "--constraint",
            str(args.constraint),
            "--metric",
            metric,
            "--degree-candidates",
            str(args.degree_candidates),
            "--lambda-candidates",
            str(args.lambda_candidates),
            "--cv-mode",
            str(args.cv_mode),
            "--output-dir",
            out_dir,
            "--plot-output-dir",
            plot_dir,
            "--validation-dir",
            val_dir,
            "--metrics-output-root",
            str(args.metrics_output_root),
        ]
        print(f"[all-metrics] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train monotone Bernstein particle-size/AE models from time-scheduled grinding trials."
    )
    parser.add_argument("--reagent", type=str, default="all", choices=["NaCl", "Citricacid", "MSG", "all"])
    parser.add_argument("--trial", type=str, default="all", choices=["1st", "2nd", "3rd", "all"])
    parser.add_argument("--constraint", type=str, default="increasing", choices=["auto", "increasing", "decreasing"])
    parser.add_argument("--metric", type=str, default="D50", choices=["all", "D50", "Dmean", "Dmode"])
    parser.add_argument("--degree-candidates", type=str, default="2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--lambda-candidates", type=str, default="0")
    parser.add_argument("--cv-mode", type=str, default="group", choices=["none", "group"])
    parser.add_argument("--output-dir", type=str, default=RUN_OUTPUT_DIR)
    parser.add_argument("--plot-output-dir", type=str, default=RUN_OUTPUT_DIR)
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=os.path.join(RUN_OUTPUT_DIR, "model_selection"),
    )
    parser.add_argument(
        "--metrics-output-root",
        type=str,
        default=RUN_OUTPUT_DIR,
    )
    args = parser.parse_args()
    if args.metric == "all":
        run_all_metrics(args)
        return

    print("Starting particle-size/AE model training.", flush=True)
    print(
        f"Options: reagent={args.reagent}, trial={args.trial}, metric={args.metric}, "
        f"degrees={args.degree_candidates}, cv={args.cv_mode}",
        flush=True,
    )

    degree_candidates = parse_int_list(args.degree_candidates)
    lambda_candidates = parse_float_list(args.lambda_candidates)
    metric = str(args.metric)
    out_suffix = metric_suffix(metric)
    size_label = metric_axis_label(metric)

    experiment = "exp2"
    output_prefix = "training_trials"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_output_dir, exist_ok=True)
    os.makedirs(args.validation_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 24,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 14,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
        }
    )

    data_array = build_shared_dataset(experiment=experiment, reagent=args.reagent, trial=args.trial, metric=metric)
    dataset_path = os.path.join(args.output_dir, f"{output_prefix}_particle_size_ae_dataset{out_suffix}.csv")
    pd.DataFrame(data_array, columns=["particle_size", "ae_power_mV2", "grind_min", "trial", "reagent"]).to_csv(
        dataset_path, index=False
    )
    print(f"      saved dataset: {dataset_path}", flush=True)

    curve_rows = []
    summary_rows = []
    metrics_rows = []
    p2ae_models: dict[str, BernsteinMonotoneRegressor] = {}

    reagents = np.unique(data_array[:, 4])
    print(f"[3/4] Training Bernstein models for {len(reagents)} materials...", flush=True)

    for current_reagent in reagents:
        mask = data_array[:, 4] == current_reagent
        reagent_data = data_array[mask]

        size_vals = np.array(reagent_data[:, 0], dtype=float)
        ae_vals = np.array(reagent_data[:, 1], dtype=float)
        trial_labels = np.array(reagent_data[:, 3], dtype=object)

        for direction in ["particle2ae", "ae2particle"]:
            print(f"      fitting {current_reagent} / {direction}...", flush=True)
            if direction == "particle2ae":
                x_data = size_vals
                y_data = ae_vals
                x_label = size_label
                y_label = r"Total spectral power ($\mathrm{mV}^2$)"
            else:
                x_data = ae_vals
                y_data = size_vals
                x_label = r"Total spectral power ($\mathrm{mV}^2$)"
                y_label = size_label

            if args.constraint == "auto":
                monotone = infer_monotone_direction(x_data, y_data)
            else:
                monotone = args.constraint

            rows_local = []
            for degree in degree_candidates:
                for lambda_smooth in lambda_candidates:
                    row = fit_score_row(
                        x_data=x_data,
                        y_data=y_data,
                        monotone=monotone,
                        degree=int(degree),
                        lambda_smooth=float(lambda_smooth),
                    )
                    row.update(
                        {
                            "reagent": str(current_reagent),
                            "direction": direction,
                            "constraint_direction": monotone,
                        }
                    )
                    rows_local.append(row)
                    curve_rows.append(row)

            local_df = pd.DataFrame(rows_local)
            valid = local_df[local_df["valid"]].copy()
            if valid.empty:
                print(f"      skipped {current_reagent} / {direction}: no valid fits.", flush=True)
                continue

            best_aic = valid.loc[valid["aic"].idxmin()]
            best_bic = valid.loc[valid["bic"].idxmin()]

            print(f"      running group CV for {current_reagent} / {direction}...", flush=True)
            cv_best_degree, cv_best_lambda, cv_best_rmse = choose_hyperparams(
                x_data=x_data,
                y_data=y_data,
                groups=trial_labels,
                monotone=monotone,
                degree_candidates=degree_candidates,
                lambda_candidates=lambda_candidates,
                cv_mode=args.cv_mode,
            )

            # Save BIC-selected models/plots for downstream use.
            degree_best = int(best_bic["degree"])
            lambda_best = float(best_bic["lambda_smooth"])
            reg = BernsteinMonotoneRegressor(
                BernsteinMonotoneConfig(
                    degree=degree_best,
                    monotone=monotone,
                    lambda_smooth=lambda_best,
                )
            ).fit(x_data, y_data)
            model_path = os.path.join(
                args.output_dir,
                f"bernstein_model_{direction}_{current_reagent}_training_trials{out_suffix}.joblib",
            )
            save_model(
                model_path,
                reg,
                extra={
                    "experiment": experiment,
                    "direction": direction,
                    "reagent": str(current_reagent),
                    "criterion": "bic",
                },
            )
            if direction == "particle2ae":
                p2ae_models[str(current_reagent)] = reg

            y_pred_train = reg.predict(x_data)
            x_grid = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 400)
            mono_stats = reg.monotonicity_metrics(x_grid)

            metrics_rows.append(
                {
                    "direction": direction,
                    "reagent": str(current_reagent),
                    "method": "bernstein_bic",
                    "selection_criterion": "bic",
                    "metric": metric,
                    "rmse_train": float(np.sqrt(mean_squared_error(y_data, y_pred_train))),
                    "mae_train": float(mean_absolute_error(y_data, y_pred_train)),
                    "r_squared": float(r2_score(y_data, y_pred_train)),
                    "violation_rate": float(mono_stats["violation_rate"]),
                    "max_violation_derivative": float(mono_stats["max_violation_derivative"]),
                    "mean_derivative": float(mono_stats["mean_derivative"]),
                    "constraint_direction": monotone,
                    "degree": degree_best,
                    "lambda_smooth": lambda_best,
                    "n_params": int(degree_best + 1),
                    "aic_selected": float(best_bic["aic"]),
                    "bic_selected": float(best_bic["bic"]),
                    "cv_best_degree": int(cv_best_degree),
                    "cv_best_lambda": float(cv_best_lambda),
                    "cv_best_rmse": float(cv_best_rmse),
                    "model_path": model_path,
                }
            )

            plot_path = os.path.join(
                args.plot_output_dir,
                f"particle_size_ae_model_{direction}_{current_reagent}{out_suffix}.png",
            )
            save_fit_plot(
                x_data=x_data,
                y_data=y_data,
                trial_labels=trial_labels,
                reg=reg,
                x_label=x_label,
                y_label=y_label,
                out_path=plot_path,
            )

            summary_rows.append(
                {
                    "reagent": str(current_reagent),
                    "direction": direction,
                    "constraint_direction": monotone,
                    "cv_best_degree": int(cv_best_degree),
                    "cv_best_lambda": float(cv_best_lambda),
                    "cv_best_rmse": float(cv_best_rmse),
                    "aic_best_degree": int(best_aic["degree"]),
                    "aic_best_lambda": float(best_aic["lambda_smooth"]),
                    "aic_best_value": float(best_aic["aic"]),
                    "aic_best_train_rmse": float(best_aic["rmse_train"]),
                    "bic_best_degree": int(best_bic["degree"]),
                    "bic_best_lambda": float(best_bic["lambda_smooth"]),
                    "bic_best_value": float(best_bic["bic"]),
                    "bic_best_train_rmse": float(best_bic["rmse_train"]),
                    "same_degree_aic_bic": bool(int(best_aic["degree"]) == int(best_bic["degree"])),
                    "same_degree_cv_aic": bool(int(cv_best_degree) == int(best_aic["degree"])),
                    "same_degree_cv_bic": bool(int(cv_best_degree) == int(best_bic["degree"])),
                }
            )

            plot_path = os.path.join(
                args.validation_dir,
                f"bernstein_model_selection_{current_reagent}_{direction}{out_suffix}.png",
            )
            save_bic_curve_plot(
                curve_df=local_df,
                out_path=plot_path,
                title=f"{experiment} {current_reagent} {direction}",
                cv_best_degree=int(cv_best_degree),
            )

            print(
                f"[{current_reagent}][{direction}] "
                f"CV={int(cv_best_degree)}, AIC={int(best_aic['degree'])}, BIC={int(best_bic['degree'])}",
                flush=True,
            )

    print("[4/4] Saving model-selection summaries and thresholds...", flush=True)

    curve_df = pd.DataFrame(curve_rows)
    curve_path = os.path.join(args.validation_dir, f"bernstein_model_selection_curve{out_suffix}.csv")
    curve_df.to_csv(curve_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        args.validation_dir, f"bernstein_model_selection_summary{out_suffix}.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    if metric == "D50":
        metrics_dir = args.metrics_output_root
    else:
        metrics_dir = os.path.join(args.metrics_output_root, "mean_and_mode", metric)
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"training_trials_bernstein_metrics{out_suffix}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    threshold_path = None
    if metric == "D50":
        validation_psd_base = os.path.join("data", "powder_size_distribution", "exp3")
        threshold_rows = []
        targets_by_reagent = collect_validation_targets_by_reagent(
            base_path=validation_psd_base,
            reagents=sorted(p2ae_models.keys()),
        )
        for reagent, model in p2ae_models.items():
            for target_d50 in targets_by_reagent.get(reagent, []):
                ae_threshold = float(model.predict(np.array([float(target_d50)], dtype=float))[0])
                threshold_rows.append(
                    {
                        "Material": reagent,
                        "Target_D50": int(target_d50),
                        "AE_Threshold_mV2": ae_threshold,
                        "Model": "bernstein_particle2ae_training_trials",
                    }
                )

        threshold_path = os.path.join(args.output_dir, "autonomous_stopping_thresholds_from_training_models.csv")
        pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved BIC curve: {curve_path}")
    print(f"Saved BIC selection summary: {summary_path}")
    if threshold_path is not None:
        print(f"Saved autonomous stopping AE thresholds: {threshold_path}")


if __name__ == "__main__":
    main()
