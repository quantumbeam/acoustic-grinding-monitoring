import argparse
import glob
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from ae_fft import calculate_fft_power

RUN_OUTPUT_DIR = os.path.join("analysis_results", "run_04_plot_ae_power_trends")


def norm_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def parse_timestamp(file_path: str):
    match = re.search(r"(\d{8}_\d{6})", os.path.basename(file_path))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def moving_average(data, window_size=4):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def load_cache(cache_file):
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def update_cache_policy_a(cache, cache_file, file_paths):
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


def configure_plot_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 18,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "lines.linewidth": 1.2,
            "lines.markersize": 6,
            "legend.frameon": False,
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot AE power trends with moving averages for the time-scheduled grinding trials."
    )
    parser.add_argument("--experiment", type=str, default="exp2")
    parser.add_argument("--reagent", type=str, default="all")
    parser.add_argument("--trial", type=str, default="all")
    parser.add_argument("--grind-min", type=int, default=25)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default=RUN_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    base_dir = os.path.join("data/ae", args.experiment)
    if args.reagent == "all":
        candidate_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
        preferred = ["MSG", "Citricacid", "NaCl"]
        preferred_dirs = [os.path.join(base_dir, name) for name in preferred]
        reagent_dirs = [d for d in preferred_dirs if d in candidate_dirs]
        if not reagent_dirs:
            reagent_dirs = candidate_dirs
    else:
        reagent_dirs = [os.path.join(base_dir, args.reagent)]

    os.makedirs(args.out_dir, exist_ok=True)

    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ae_power_cache.json")
    cache = load_cache(cache_file)

    reagent_to_files = {}
    for reagent_dir in reagent_dirs:
        if not os.path.isdir(reagent_dir):
            continue
        reagent = os.path.basename(reagent_dir)
        if args.trial == "all":
            trial_dirs = sorted(glob.glob(os.path.join(reagent_dir, "*")))
        else:
            trial_dirs = [os.path.join(reagent_dir, args.trial)]

        for trial_dir in trial_dirs:
            if not os.path.isdir(trial_dir):
                continue
            trial_name = os.path.basename(trial_dir)
            pattern = os.path.join(trial_dir, f"*grind{args.grind_min}min*.csv")
            files = sorted(glob.glob(pattern))
            if files:
                reagent_to_files[(reagent, trial_name)] = files

    if not reagent_to_files:
        print("No matching files found.")
        return

    all_files = [path for paths in reagent_to_files.values() for path in paths]
    cache = update_cache_policy_a(cache, cache_file, all_files)

    configure_plot_style()

    for (reagent, trial_name), files in reagent_to_files.items():
        files_sorted = sorted(
            files,
            key=lambda p: (
                parse_timestamp(p) or datetime.min,
                os.path.basename(p),
            ),
        )

        power_values = []
        for file_path in files_sorted:
            key = norm_path(file_path)
            power = cache.get(key)
            if power is None:
                power = calculate_fft_power(file_path)
            if power is None:
                print(f"Skip (no power): {file_path}")
                continue
            power_values.append(power * 1e6)

        if not power_values:
            print(f"No valid data for {reagent} {args.trial}.")
            continue

        x_vals = np.arange(1, len(power_values) + 1)
        smoothed = moving_average(np.array(power_values), window_size=args.window)
        smooth_x = np.arange(args.window, len(power_values) + 1)

        plt.figure(figsize=(12, 8))
        plt.plot(
            x_vals,
            power_values,
            "o-",
            color="black",
            label="Original Data",
        )
        if len(smoothed):
            plt.plot(
                smooth_x,
                smoothed,
                "x--",
                color="red",
                label=f"{args.window}-point Moving Average",
            )

        plt.xlabel("Number of motions")
        plt.ylabel(r"Total spectral power($\mathrm{mV}^2$)")
        plt.legend(loc="upper right")
        plt.tight_layout()

        base_name = f"ae_power_trend_{reagent}_{trial_name}_{args.grind_min}min"
        out_png = os.path.join(args.out_dir, f"{base_name}.png")
        plt.savefig(out_png, dpi=args.dpi)
        plt.close()
        print(f"Saved plot: {out_png}")



if __name__ == "__main__":
    main()
