import argparse
import csv
import glob
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt


def parse_timestamp_from_filename(file_path: str):
    match = re.search(r"(\d{8}_\d{6})", os.path.basename(file_path))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def get_d50(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Dx (50)"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        return float(parts[1])
    except (OSError, ValueError) as e:
        print(f"Could not read or parse D50 from {file_path}: {e}")
    return None


def parse_grind_min(file_path: str):
    match = re.search(r"grind(\d+)min", os.path.basename(file_path))
    if not match:
        return None
    return float(match.group(1))


def linear_fit_r(x_vals, y_vals):
    n = len(x_vals)
    if n < 2:
        return 0.0, 0.0, float("nan")
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    sxx = sum((x - x_mean) ** 2 for x in x_vals)
    syy = sum((y - y_mean) ** 2 for y in y_vals)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    slope = 0.0 if sxx == 0.0 else sxy / sxx
    intercept = y_mean - slope * x_mean
    r = float("nan") if sxx == 0.0 or syy == 0.0 else sxy / (sxx * syy) ** 0.5
    return slope, intercept, r


def main():
    parser = argparse.ArgumentParser(
        description="Plot sequential D50 ratios vs elapsed time between grind steps."
    )
    parser.add_argument("--experiment", type=str, default="exp2")
    parser.add_argument("--reagent", type=str, default="all")
    parser.add_argument("--trial", type=str, default="all")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("results", "discussion", "d50_ratio_timeseries"),
    )
    args = parser.parse_args()

    base_dir = os.path.join("powder_size_distribution_data", args.experiment)
    reagent_dirs = (
        sorted(glob.glob(os.path.join(base_dir, "*")))
        if args.reagent == "all"
        else [os.path.join(base_dir, args.reagent)]
    )

    os.makedirs(args.out_dir, exist_ok=True)

    for reagent_dir in reagent_dirs:
        if not os.path.isdir(reagent_dir):
            continue
        reagent = os.path.basename(reagent_dir)
        trial_dirs = (
            sorted(glob.glob(os.path.join(reagent_dir, "*")))
            if args.trial == "all"
            else [os.path.join(reagent_dir, args.trial)]
        )

        for trial_dir in trial_dirs:
            if not os.path.isdir(trial_dir):
                continue
            trial = os.path.basename(trial_dir)
            psd_files = sorted(glob.glob(os.path.join(trial_dir, "*.csv")))
            if not psd_files:
                continue

            points = []
            for psd_file in psd_files:
                grind_min = parse_grind_min(psd_file)
                if grind_min is None:
                    continue
                d50 = get_d50(psd_file)
                if d50 is None:
                    continue
                timestamp = parse_timestamp_from_filename(psd_file)
                points.append((grind_min, timestamp, d50, psd_file))

            if len(points) < 2:
                continue

            points.sort(key=lambda x: (x[0], x[1] or datetime.min))

            ratio_rows = []
            for prev, curr in zip(points[:-1], points[1:]):
                prev_min, _, prev_d50, prev_file = prev
                curr_min, _, curr_d50, curr_file = curr
                delta_min = curr_min - prev_min
                if delta_min <= 0:
                    continue
                ratio = curr_d50 / prev_d50 if prev_d50 != 0 else float("nan")
                ratio_rows.append(
                    {
                        "prev_grind_min": prev_min,
                        "next_grind_min": curr_min,
                        "delta_min": delta_min,
                        "d50_prev": prev_d50,
                        "d50_next": curr_d50,
                        "d50_ratio": ratio,
                        "prev_file": os.path.basename(prev_file),
                        "next_file": os.path.basename(curr_file),
                    }
                )

            if not ratio_rows:
                continue

            out_csv = os.path.join(
                args.out_dir,
                f"d50_ratio_{args.experiment}_{reagent}_{trial}.csv",
            )
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(ratio_rows[0].keys()))
                writer.writeheader()
                writer.writerows(ratio_rows)
            print(f"Saved ratio table: {out_csv}")

            plot_rows = ratio_rows
            if len(plot_rows) >= 4:
                plot_rows = plot_rows[1:4]

            delta_vals = [row["delta_min"] for row in plot_rows]
            ratio_vals = [row["d50_ratio"] for row in plot_rows]
            slope, intercept, r = linear_fit_r(delta_vals, ratio_vals)

            plt.figure(figsize=(10, 7))
            plt.scatter(delta_vals, ratio_vals, s=80, c="black")
            plt.plot(delta_vals, ratio_vals, "k--", alpha=0.6)
            if len(delta_vals) >= 2:
                x_min = min(delta_vals)
                x_max = max(delta_vals)
                x_fit = [x_min, x_max]
                y_fit = [slope * x + intercept for x in x_fit]
                plt.plot(x_fit, y_fit, "k-", label="linear fit")
            plt.xlabel("Elapsed time between grind steps (min)")
            plt.ylabel("D50 ratio (next / prev)")
            title = f"D50 ratio (next/prev) vs elapsed time ({reagent} {trial})"
            if len(delta_vals) >= 2:
                title += f" r={r:.3f}"
            plt.title(title)
            if len(delta_vals) >= 2:
                plt.legend()
            out_plot = os.path.join(
                args.out_dir,
                f"d50_ratio_plot_{args.experiment}_{reagent}_{trial}.png",
            )
            plt.savefig(out_plot, dpi=300)
            plt.close()
            print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()
