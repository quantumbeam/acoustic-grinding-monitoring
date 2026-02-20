import argparse
import glob
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from exp2_monotone_bernstein_model import (
    build_shared_dataset,
    choose_hyperparams,
    infer_monotone_direction,
    parse_float_list,
    parse_int_list,
)
from monotone_bernstein import BernsteinMonotoneConfig, BernsteinMonotoneRegressor, save_model

plt.style.use(["science", "ieee", "no-latex"])
EXPORT_DPI = 600


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
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
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
    out_pdf = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def collect_exp3_targets_by_reagent(base_path: str, reagents: list[str]) -> dict[str, list[int]]:
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
    parser = argparse.ArgumentParser(description="Train exp2 monotone Bernstein models with degree selection by BIC.")
    parser.add_argument("--reagent", type=str, default="all", choices=["NaCl", "Citricacid", "MSG", "all"])
    parser.add_argument("--trial", type=str, default="all", choices=["1st", "2nd", "3rd", "all"])
    parser.add_argument("--constraint", type=str, default="increasing", choices=["auto", "increasing", "decreasing"])
    parser.add_argument("--metric", type=str, default="all", choices=["all", "D50", "Dmean", "Dmode"])
    parser.add_argument("--degree-candidates", type=str, default="2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--lambda-candidates", type=str, default="0")
    parser.add_argument("--cv-mode", type=str, default="group", choices=["none", "group"])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--plot-output-dir", type=str, default="results")
    parser.add_argument(
        "--validation-dir",
        type=str,
        default="results/model_validation",
    )
    parser.add_argument(
        "--metrics-output-root",
        type=str,
        default="results/SI_figs",
    )
    args = parser.parse_args()
    if args.metric == "all":
        run_all_metrics(args)
        return

    degree_candidates = parse_int_list(args.degree_candidates)
    lambda_candidates = parse_float_list(args.lambda_candidates)
    metric = str(args.metric)
    out_suffix = metric_suffix(metric)
    size_label = metric_axis_label(metric)

    experiment = "exp2"
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
    dataset_path = os.path.join(args.output_dir, f"{experiment}_monotone_bernstein_dataset_raw_bic{out_suffix}.csv")
    pd.DataFrame(data_array, columns=["particle_size", "ae_power_mV2", "grind_min", "trial", "reagent"]).to_csv(
        dataset_path, index=False
    )

    curve_rows = []
    summary_rows = []
    metrics_rows = []
    p2ae_models: dict[str, BernsteinMonotoneRegressor] = {}

    for current_reagent in np.unique(data_array[:, 4]):
        mask = data_array[:, 4] == current_reagent
        reagent_data = data_array[mask]

        size_vals = np.array(reagent_data[:, 0], dtype=float)
        ae_vals = np.array(reagent_data[:, 1], dtype=float)
        trial_labels = np.array(reagent_data[:, 3], dtype=object)

        for direction in ["particle2ae", "ae2particle"]:
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
                continue

            best_aic = valid.loc[valid["aic"].idxmin()]
            best_bic = valid.loc[valid["bic"].idxmin()]

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
                f"monotone_bernstein_model_bic_{direction}_{current_reagent}_{experiment}{out_suffix}.joblib",
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
                    "method": "monotone_bernstein_bic",
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
                f"{experiment}_monotone_bernstein_plot_bic_{direction}_{current_reagent}{out_suffix}.png",
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
                f"{experiment}_monotone_bernstein_bic_curve_{current_reagent}_{direction}{out_suffix}.png",
            )
            save_bic_curve_plot(
                curve_df=local_df,
                out_path=plot_path,
                title=f"{experiment} {current_reagent} {direction}",
                cv_best_degree=int(cv_best_degree),
            )

            print(
                f"[{current_reagent}][{direction}] "
                f"CV={int(cv_best_degree)}, AIC={int(best_aic['degree'])}, BIC={int(best_bic['degree'])}"
            )

    curve_df = pd.DataFrame(curve_rows)
    curve_path = os.path.join(args.validation_dir, f"{experiment}_monotone_bernstein_bic_curve{out_suffix}.csv")
    curve_df.to_csv(curve_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        args.validation_dir, f"{experiment}_monotone_bernstein_bic_selection_summary{out_suffix}.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    if metric == "D50":
        metrics_dir = args.metrics_output_root
    else:
        metrics_dir = os.path.join(args.metrics_output_root, "mean_and_mode", metric)
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{experiment}_monotone_bernstein_metrics_both_directions_bic{out_suffix}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    threshold_path = None
    if metric == "D50":
        exp3_psd_base = os.path.join("data", "powder_size_distribution", "exp3")
        threshold_rows = []
        targets_by_reagent = collect_exp3_targets_by_reagent(
            base_path=exp3_psd_base,
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
                        "Model": "monotone_bernstein_bic_particle2ae_exp2",
                    }
                )

        threshold_path = os.path.join(args.output_dir, "exp3_ae_thresholds_from_exp2_monotone_bernstein_bic.csv")
        pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved BIC curve: {curve_path}")
    print(f"Saved BIC selection summary: {summary_path}")
    if threshold_path is not None:
        print(f"Saved exp3 AE thresholds: {threshold_path}")


if __name__ == "__main__":
    main()
