import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from fft_processing import calculate_fft_power
from monotone_bernstein import BernsteinMonotoneConfig, BernsteinMonotoneRegressor, save_model

plt.style.use(["science", "ieee", "no-latex"])


def norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def update_ae_cache(cache_file_path, required_files):
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                ae_cache = json.load(f)
        except (IOError, json.JSONDecodeError):
            ae_cache = {}
    else:
        ae_cache = {}

    updated_count = 0
    for file_path in required_files:
        key = norm_path(file_path)
        if key in ae_cache:
            continue
        new_power = calculate_fft_power(file_path)
        if new_power is None:
            continue
        ae_cache[key] = float(new_power)
        updated_count += 1

    if updated_count > 0:
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(ae_cache, f, indent=2)
    return ae_cache


def get_d50(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Dx (50)"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        return float(parts[1])
    except (IOError, ValueError):
        return None
    return None


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


def parse_float_list(s: str) -> list[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def parse_int_list(s: str) -> list[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def build_shared_dataset(experiment: str, reagent: str, trial: str):
    ae_base_path = os.path.join("data/ae", experiment)
    psd_base_path = os.path.join("data/powder_size_distribution", experiment)

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

        grind_key = match.group(1)
        ae_session_path = os.path.join(ae_base_path, reagent_name, trial_name)
        required_ae_files.update(glob.glob(os.path.join(ae_session_path, f"*{grind_key}*.csv")))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, "ae_power_cache.json")
    ae_cache = update_ae_cache(cache_file, list(required_ae_files))

    collected_data = []
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

        ae_power_timeseries = [ae_cache.get(norm_path(f)) for f in ae_files]
        ae_power_timeseries = [p for p in ae_power_timeseries if p is not None]
        if len(ae_power_timeseries) < 4:
            continue

        ae_power_mv2 = np.array(ae_power_timeseries, dtype=float) * 1e6
        smoothed = moving_average(ae_power_mv2, window_size=4)
        if smoothed.size == 0:
            continue

        collected_data.append((float(d50), float(smoothed[-1]), grind_min, trial_name, reagent_name))

    if not collected_data:
        raise RuntimeError("No matched data points found.")

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
                rmse = float(np.sqrt(mean_squared_error(y_data[va_idx], y_pred)))
                fold_rmse.append(rmse)

            if failed or not fold_rmse:
                continue

            mean_rmse = float(np.mean(fold_rmse))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best = (int(degree), float(lambda_smooth))

    if best is None:
        return int(degree_candidates[0]), float(lambda_candidates[0]), float("nan")
    return best[0], best[1], best_rmse


def evaluate_degree_oof_curve(
    x_data: np.ndarray,
    y_data: np.ndarray,
    groups: np.ndarray,
    monotone: str,
    degree_candidates: list[int],
    cv_mode: str,
) -> list[dict]:
    rows = []

    unique_groups = np.unique(groups)
    n_splits = min(3, unique_groups.size)
    run_cv = cv_mode != "none" and n_splits >= 2
    gkf = GroupKFold(n_splits=n_splits) if run_cv else None

    x_min = float(np.min(x_data))
    x_max = float(np.max(x_data))
    x_grid = np.linspace(x_min, x_max, 500)
    dx = float(x_grid[1] - x_grid[0]) if x_grid.size > 1 else 1.0

    for degree in degree_candidates:
        fold_rmse = []
        fold_mae = []
        failed = False

        if run_cv and gkf is not None:
            for tr_idx, va_idx in gkf.split(x_data, y_data, groups=groups):
                try:
                    reg_cv = BernsteinMonotoneRegressor(
                        BernsteinMonotoneConfig(
                            degree=int(degree),
                            monotone=monotone,
                            lambda_smooth=0.0,
                        )
                    ).fit(x_data[tr_idx], y_data[tr_idx])
                except RuntimeError:
                    failed = True
                    break

                y_pred_va = reg_cv.predict(x_data[va_idx])
                fold_rmse.append(float(np.sqrt(mean_squared_error(y_data[va_idx], y_pred_va))))
                fold_mae.append(float(mean_absolute_error(y_data[va_idx], y_pred_va)))

        try:
            reg_full = BernsteinMonotoneRegressor(
                BernsteinMonotoneConfig(
                    degree=int(degree),
                    monotone=monotone,
                    lambda_smooth=0.0,
                )
            ).fit(x_data, y_data)
            y_grid = reg_full.predict(x_grid)
            d1 = np.gradient(y_grid, dx)
            d2 = np.gradient(d1, dx)
            curvature_l1 = float(np.sum(np.abs(d2)) * dx)
            edge_slope_abs = float(abs(d1[0]) + abs(d1[-1]))
        except RuntimeError:
            failed = True
            curvature_l1 = float("nan")
            edge_slope_abs = float("nan")

        rows.append(
            {
                "degree": int(degree),
                "lambda_smooth": 0.0,
                "n_folds": int(n_splits if run_cv else 0),
                "cv_valid": bool(run_cv and not failed and len(fold_rmse) > 0),
                "cv_rmse_mean": float(np.mean(fold_rmse)) if (not failed and fold_rmse) else float("nan"),
                "cv_rmse_std": float(np.std(fold_rmse, ddof=0)) if (not failed and fold_rmse) else float("nan"),
                "cv_mae_mean": float(np.mean(fold_mae)) if (not failed and fold_mae) else float("nan"),
                "cv_mae_std": float(np.std(fold_mae, ddof=0)) if (not failed and fold_mae) else float("nan"),
                "curvature_l1": curvature_l1,
                "edge_slope_abs": edge_slope_abs,
            }
        )

    return rows


def save_degree_cv_error_plots(
    degree_curve_df: pd.DataFrame,
    validation_dir: str,
    experiment: str,
) -> None:
    if degree_curve_df.empty:
        return

    for (reagent, direction), g in degree_curve_df.groupby(["reagent", "direction"]):
        valid = g[g["cv_valid"]].copy()
        if valid.empty:
            continue
        valid = valid.sort_values("degree")

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            valid["degree"].to_numpy(dtype=int),
            valid["cv_rmse_mean"].to_numpy(dtype=float),
            yerr=valid["cv_rmse_std"].to_numpy(dtype=float),
            fmt="o-",
            color="black",
            ecolor="gray",
            capsize=4,
            linewidth=2,
            markersize=7,
        )
        plt.xlabel("Bernstein degree")
        plt.ylabel("CV RMSE")
        plt.title(f"{experiment} {reagent} {direction}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(
            validation_dir,
            f"{experiment}_monotone_bernstein_degree_vs_cv_rmse_{reagent}_{direction}.png",
        )
        plt.savefig(plot_path, dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train monotone Bernstein models on exp2 data.")
    parser.add_argument("--reagent", type=str, default="all", choices=["NaCl", "Citricacid", "MSG", "all"])
    parser.add_argument("--trial", type=str, default="all", choices=["1st", "2nd", "3rd", "all"])
    parser.add_argument("--constraint", type=str, default="increasing", choices=["auto", "increasing", "decreasing"])
    parser.add_argument("--degree-candidates", type=str, default="5,6,7,8,9")
    parser.add_argument("--degree-validation-candidates", type=str, default="2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--lambda-candidates", type=str, default="0,1e-4,1e-3,1e-2")
    parser.add_argument("--cv-mode", type=str, default="group", choices=["none", "group"])
    parser.add_argument("--output-dir", type=str, default="model_comparison/monotone_bernstein")
    parser.add_argument("--validation-dir", type=str, default="results/model_validation")
    args = parser.parse_args()

    degree_candidates = parse_int_list(args.degree_candidates)
    degree_validation_candidates = parse_int_list(args.degree_validation_candidates)
    lambda_candidates = parse_float_list(args.lambda_candidates)

    experiment = "exp2"
    output_dir = args.output_dir
    validation_dir = args.validation_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Match publication-style readability used in GPR baseline plots.
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

    data_array = build_shared_dataset(experiment=experiment, reagent=args.reagent, trial=args.trial)

    dataset_path = os.path.join(output_dir, f"{experiment}_monotone_bernstein_dataset_raw.csv")
    pd.DataFrame(data_array, columns=["d50", "ae_power_mV2", "grind_min", "trial", "reagent"]).to_csv(
        dataset_path, index=False
    )

    all_metrics = []
    lambda_effect_rows = []
    degree_curve_rows = []
    markers = {"1st": "o", "2nd": "x", "3rd": "^"}
    colors = {"1st": "black", "2nd": "red", "3rd": "blue"}

    for current_reagent in np.unique(data_array[:, 4]):
        mask = data_array[:, 4] == current_reagent
        reagent_data = data_array[mask]

        d50_vals = np.array(reagent_data[:, 0], dtype=float)
        ae_vals = np.array(reagent_data[:, 1], dtype=float)
        trial_labels = np.array(reagent_data[:, 3], dtype=object)

        for direction in ["particle2ae", "ae2particle"]:
            if direction == "particle2ae":
                x_data = d50_vals
                y_data = ae_vals
                x_label = r"$D_{50}~(\mathrm{\mu m})$"
                y_label = r"Total spectral power ($\mathrm{mV}^2$)"
            else:
                x_data = ae_vals
                y_data = d50_vals
                x_label = r"Total spectral power ($\mathrm{mV}^2$)"
                y_label = r"$D_{50}~(\mathrm{\mu m})$"

            if args.constraint == "auto":
                monotone = infer_monotone_direction(x_data, y_data)
            else:
                monotone = args.constraint

            degree_eval_rows = evaluate_degree_oof_curve(
                x_data=x_data,
                y_data=y_data,
                groups=trial_labels,
                monotone=monotone,
                degree_candidates=degree_validation_candidates,
                cv_mode=args.cv_mode,
            )
            for row in degree_eval_rows:
                row.update(
                    {
                        "reagent": current_reagent,
                        "direction": direction,
                        "constraint_direction": monotone,
                    }
                )
                degree_curve_rows.append(row)

            cv_best_degree, cv_best_lambda, cv_best_rmse = choose_hyperparams(
                x_data=x_data,
                y_data=y_data,
                groups=trial_labels,
                monotone=monotone,
                degree_candidates=degree_candidates,
                lambda_candidates=lambda_candidates,
                cv_mode=args.cv_mode,
            )
            lambda0_degree, _, lambda0_cv_rmse = choose_hyperparams(
                x_data=x_data,
                y_data=y_data,
                groups=trial_labels,
                monotone=monotone,
                degree_candidates=degree_candidates,
                lambda_candidates=[0.0],
                cv_mode=args.cv_mode,
            )

            # Main report model policy: lambda=0 with CV-selected degree under lambda=0.
            best_degree = int(lambda0_degree)
            best_lambda = 0.0

            # Validation-only fit: full-grid CV best (may include non-zero lambda).
            reg_full_cv = BernsteinMonotoneRegressor(
                BernsteinMonotoneConfig(
                    degree=int(cv_best_degree),
                    monotone=monotone,
                    lambda_smooth=float(cv_best_lambda),
                )
            ).fit(x_data, y_data)
            reg = BernsteinMonotoneRegressor(
                BernsteinMonotoneConfig(
                    degree=best_degree,
                    monotone=monotone,
                    lambda_smooth=best_lambda,
                )
            ).fit(x_data, y_data)

            model_path = os.path.join(
                output_dir,
                f"monotone_bernstein_model_{direction}_{current_reagent}_{experiment}.joblib",
            )
            save_model(
                model_path,
                reg,
                extra={
                    "experiment": experiment,
                    "direction": direction,
                    "reagent": str(current_reagent),
                },
            )

            x_plot = np.linspace(float(np.min(x_data) * 0.9), float(np.max(x_data) * 1.1), 500)
            y_plot = reg.predict(x_plot)
            y_pred_train = reg.predict(x_data)
            y_pred_train_full = reg_full_cv.predict(x_data)

            mono_stats = reg.monotonicity_metrics(x_plot)

            all_metrics.append(
                {
                    "direction": direction,
                    "reagent": current_reagent,
                    "method": "monotone_bernstein",
                    "rmse_train": float(np.sqrt(mean_squared_error(y_data, y_pred_train))),
                    "mae_train": float(mean_absolute_error(y_data, y_pred_train)),
                    "r_squared": float(r2_score(y_data, y_pred_train)),
                    "violation_rate": mono_stats["violation_rate"],
                    "max_violation_derivative": mono_stats["max_violation_derivative"],
                    "mean_derivative": mono_stats["mean_derivative"],
                    "constraint_direction": monotone,
                    "degree": best_degree,
                    "lambda_smooth": best_lambda,
                    "cv_best_degree_full_grid": cv_best_degree,
                    "cv_best_lambda_full_grid": cv_best_lambda,
                    "cv_best_rmse_full_grid": cv_best_rmse,
                    "cv_best_degree_lambda0": lambda0_degree,
                    "cv_best_rmse_lambda0": lambda0_cv_rmse,
                    "cv_rmse_delta_lambda0_minus_full": (
                        float(lambda0_cv_rmse - cv_best_rmse)
                        if np.isfinite(lambda0_cv_rmse) and np.isfinite(cv_best_rmse)
                        else float("nan")
                    ),
                    "model_path": model_path,
                }
            )
            lambda_effect_rows.append(
                {
                    "direction": direction,
                    "reagent": current_reagent,
                    "constraint_direction": monotone,
                    "cv_best_degree_full_grid": int(cv_best_degree),
                    "cv_best_lambda_full_grid": float(cv_best_lambda),
                    "cv_best_rmse_full_grid": float(cv_best_rmse),
                    "degree_lambda0_main": int(best_degree),
                    "lambda_lambda0_main": float(best_lambda),
                    "cv_best_degree_lambda0": int(lambda0_degree),
                    "cv_best_rmse_lambda0": float(lambda0_cv_rmse),
                    "cv_rmse_delta_lambda0_minus_full": (
                        float(lambda0_cv_rmse - cv_best_rmse)
                        if np.isfinite(lambda0_cv_rmse) and np.isfinite(cv_best_rmse)
                        else float("nan")
                    ),
                    "train_rmse_full_grid": float(np.sqrt(mean_squared_error(y_data, y_pred_train_full))),
                    "train_rmse_lambda0_main": float(np.sqrt(mean_squared_error(y_data, y_pred_train))),
                    "train_rmse_delta_lambda0_minus_full": float(
                        np.sqrt(mean_squared_error(y_data, y_pred_train))
                        - np.sqrt(mean_squared_error(y_data, y_pred_train_full))
                    ),
                    "train_mae_full_grid": float(mean_absolute_error(y_data, y_pred_train_full)),
                    "train_mae_lambda0_main": float(mean_absolute_error(y_data, y_pred_train)),
                    "train_mae_delta_lambda0_minus_full": float(
                        mean_absolute_error(y_data, y_pred_train)
                        - mean_absolute_error(y_data, y_pred_train_full)
                    ),
                    "train_r2_full_grid": float(r2_score(y_data, y_pred_train_full)),
                    "train_r2_lambda0_main": float(r2_score(y_data, y_pred_train)),
                    "train_r2_delta_lambda0_minus_full": float(
                        r2_score(y_data, y_pred_train) - r2_score(y_data, y_pred_train_full)
                    ),
                }
            )

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
            plot_png = os.path.join(output_dir, f"{experiment}_monotone_bernstein_plot_{direction}_{current_reagent}.png")
            plt.savefig(plot_png, dpi=300)
            plt.close()

            print(
                f"[{current_reagent}][{direction}] "
                f"R2={all_metrics[-1]['r_squared']:.4f}, "
                f"main(degree={best_degree}, lambda={best_lambda}), "
                f"cv_best_full(degree={cv_best_degree}, lambda={cv_best_lambda}, rmse={cv_best_rmse:.4f}), "
                f"cv_lambda0(degree={lambda0_degree}, rmse={lambda0_cv_rmse:.4f}), "
                f"constraint={monotone}"
            )

    metrics_path = os.path.join(output_dir, f"{experiment}_monotone_bernstein_metrics_both_directions.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    lambda_effect_df = pd.DataFrame(lambda_effect_rows)
    lambda_effect_path = os.path.join(validation_dir, f"{experiment}_monotone_bernstein_lambda_effect_comparison.csv")
    lambda_effect_df.to_csv(lambda_effect_path, index=False)

    summary = {
        "n_models": int(len(lambda_effect_df)),
        "mean_abs_cv_rmse_delta": float(lambda_effect_df["cv_rmse_delta_lambda0_minus_full"].abs().mean()),
        "max_abs_cv_rmse_delta": float(lambda_effect_df["cv_rmse_delta_lambda0_minus_full"].abs().max()),
        "mean_abs_train_rmse_delta": float(lambda_effect_df["train_rmse_delta_lambda0_minus_full"].abs().mean()),
        "max_abs_train_rmse_delta": float(lambda_effect_df["train_rmse_delta_lambda0_minus_full"].abs().max()),
        "mean_abs_train_r2_delta": float(lambda_effect_df["train_r2_delta_lambda0_minus_full"].abs().mean()),
        "max_abs_train_r2_delta": float(lambda_effect_df["train_r2_delta_lambda0_minus_full"].abs().max()),
    }
    summary_path = os.path.join(validation_dir, f"{experiment}_monotone_bernstein_lambda_effect_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    degree_curve_df = pd.DataFrame(degree_curve_rows)
    degree_curve_path = os.path.join(validation_dir, f"{experiment}_monotone_bernstein_degree_oof_curve.csv")
    degree_curve_df.to_csv(degree_curve_path, index=False)
    save_degree_cv_error_plots(
        degree_curve_df=degree_curve_df,
        validation_dir=validation_dir,
        experiment=experiment,
    )

    degree_summary_rows = []
    if not degree_curve_df.empty:
        for (reagent, direction), g in degree_curve_df.groupby(["reagent", "direction"]):
            valid = g[g["cv_valid"]].copy()
            if valid.empty:
                continue
            valid = valid.sort_values("degree")
            idx_best = valid["cv_rmse_mean"].idxmin()
            best_row = valid.loc[idx_best]
            rmse_min = float(best_row["cv_rmse_mean"])
            rmse_min_std = float(best_row["cv_rmse_std"])
            one_se_limit = rmse_min + rmse_min_std
            one_se_candidates = valid[valid["cv_rmse_mean"] <= one_se_limit]
            one_se_degree = int(one_se_candidates["degree"].min()) if not one_se_candidates.empty else int(best_row["degree"])
            degree_summary_rows.append(
                {
                    "reagent": reagent,
                    "direction": direction,
                    "best_degree_by_cv_rmse": int(best_row["degree"]),
                    "best_cv_rmse_mean": rmse_min,
                    "best_cv_rmse_std": rmse_min_std,
                    "one_se_degree": one_se_degree,
                    "best_degree_is_boundary": bool(
                        int(best_row["degree"]) == int(valid["degree"].min())
                        or int(best_row["degree"]) == int(valid["degree"].max())
                    ),
                    "min_degree_tested": int(valid["degree"].min()),
                    "max_degree_tested": int(valid["degree"].max()),
                    "curvature_l1_at_best": float(best_row["curvature_l1"]),
                    "edge_slope_abs_at_best": float(best_row["edge_slope_abs"]),
                }
            )
    degree_summary_path = os.path.join(validation_dir, f"{experiment}_monotone_bernstein_degree_oof_summary.csv")
    pd.DataFrame(degree_summary_rows).to_csv(degree_summary_path, index=False)

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved lambda effect detail: {lambda_effect_path}")
    print(f"Saved lambda effect summary: {summary_path}")
    print(f"Saved degree OOF curve: {degree_curve_path}")
    print(f"Saved degree OOF summary: {degree_summary_path}")


if __name__ == "__main__":
    main()
