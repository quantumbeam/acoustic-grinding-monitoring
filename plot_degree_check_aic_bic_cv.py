import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monotone_bernstein import BernsteinMonotoneConfig, BernsteinMonotoneRegressor


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def group_cv_rmse(
    x_data: np.ndarray,
    y_data: np.ndarray,
    groups: np.ndarray,
    degree: int,
    monotone: str,
    lambda_smooth: float = 0.0,
) -> float:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return float("nan")

    fold_rmses = []
    for g in unique_groups:
        test_mask = groups == g
        train_mask = ~test_mask
        if np.sum(train_mask) < 2 or np.sum(test_mask) < 1:
            continue
        try:
            model = BernsteinMonotoneRegressor(
                BernsteinMonotoneConfig(
                    degree=int(degree),
                    monotone=str(monotone),
                    lambda_smooth=float(lambda_smooth),
                )
            ).fit(x_data[train_mask], y_data[train_mask])
            pred = model.predict(x_data[test_mask])
            fold_rmses.append(rmse(y_data[test_mask], pred))
        except RuntimeError:
            continue

    if not fold_rmses:
        return float("nan")
    return float(np.mean(fold_rmses))


def configure_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot vertical AIC/BIC/CV degree-check panels (default degrees: 3-7) for D50."
    )
    parser.add_argument(
        "--curve-csv",
        type=str,
        default="results/SI_plots/model_validation/D50/exp2_monotone_bernstein_bic_curve.csv",
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="results/paper_plots/exp2_monotone_bernstein_dataset_raw_bic.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/SI_plots/model_validation/D50",
    )
    parser.add_argument("--degrees", type=str, default="3,4,5,6,7")
    parser.add_argument("--degree-min", type=int, default=None)
    parser.add_argument("--degree-max", type=int, default=None)
    args = parser.parse_args()

    if args.degree_min is not None and args.degree_max is not None:
        if args.degree_min > args.degree_max:
            raise ValueError("--degree-min must be <= --degree-max")
        degrees = list(range(int(args.degree_min), int(args.degree_max) + 1))
    else:
        degrees = parse_int_list(args.degrees)
    os.makedirs(args.out_dir, exist_ok=True)

    curve_df = pd.read_csv(args.curve_csv)
    ds = pd.read_csv(args.dataset_csv)

    configure_plot_style()

    cv_rows = []
    for (reagent, direction), sub in curve_df.groupby(["reagent", "direction"], dropna=False):
        sub = sub[sub["valid"]].copy()
        sub = sub[sub["degree"].isin(degrees)].copy().sort_values("degree")
        if sub.empty:
            continue

        monotone = str(sub["constraint_direction"].iloc[0])
        reagent_ds = ds[ds["reagent"] == reagent].copy()
        if reagent_ds.empty:
            continue

        size_vals = reagent_ds["particle_size"].to_numpy(dtype=float)
        ae_vals = reagent_ds["ae_power_mV2"].to_numpy(dtype=float)
        trial_labels = reagent_ds["trial"].to_numpy(dtype=object)

        if direction == "particle2ae":
            x_data, y_data = size_vals, ae_vals
            y_label = "AIC/BIC value"
        else:
            x_data, y_data = ae_vals, size_vals
            y_label = "AIC/BIC value"

        cv_vals = []
        for d in sub["degree"].to_numpy(dtype=int):
            cv_val = group_cv_rmse(
                x_data=x_data,
                y_data=y_data,
                groups=trial_labels,
                degree=int(d),
                monotone=monotone,
                lambda_smooth=0.0,
            )
            cv_vals.append(cv_val)
            cv_rows.append(
                {
                    "reagent": str(reagent),
                    "direction": str(direction),
                    "degree": int(d),
                    "cv_rmse": float(cv_val),
                }
            )

        x_deg = sub["degree"].to_numpy(dtype=int)
        aic = sub["aic"].to_numpy(dtype=float)
        bic = sub["bic"].to_numpy(dtype=float)
        cv = np.asarray(cv_vals, dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

        axes[0].plot(x_deg, aic, "o-", color="black")
        if np.any(np.isfinite(aic)):
            i = int(np.nanargmin(aic))
            axes[0].axvline(int(x_deg[i]), color="black", linestyle="--", alpha=0.6)
        axes[0].set_ylabel("AIC")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_deg, bic, "s-", color="tab:blue")
        if np.any(np.isfinite(bic)):
            i = int(np.nanargmin(bic))
            axes[1].axvline(int(x_deg[i]), color="tab:blue", linestyle="--", alpha=0.6)
        axes[1].set_ylabel("BIC")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x_deg, cv, "^-", color="tab:green")
        if np.any(np.isfinite(cv)):
            i = int(np.nanargmin(cv))
            axes[2].axvline(int(x_deg[i]), color="tab:green", linestyle="--", alpha=0.6)
        axes[2].set_ylabel("CV RMSE")
        axes[2].set_xlabel("Bernstein degree")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()

        base = f"exp2_degree_check_aic_bic_cv_{reagent}_{direction}_D50_deg{min(degrees)}to{max(degrees)}"
        out_png = os.path.join(args.out_dir, f"{base}.png")
        out_pdf = os.path.join(args.out_dir, f"{base}.pdf")
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")

    cv_df = pd.DataFrame(cv_rows)
    cv_out = os.path.join(
        args.out_dir,
        f"exp2_monotone_bernstein_cv_curve_deg{min(degrees)}to{max(degrees)}_D50.csv",
    )
    cv_df.to_csv(cv_out, index=False)
    print(f"Saved: {cv_out}")


if __name__ == "__main__":
    main()
