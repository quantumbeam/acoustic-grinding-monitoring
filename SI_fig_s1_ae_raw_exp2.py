import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEADER_LINES = 12
ADC_OFFSET = 32768.0
ADC_SCALE = 32768.0
DEFAULT_SAMPLING_RATE_HZ = 2_000_000.0
TARGET_TRIAL = "1st"
TARGET_GRIND_MIN = 25
TARGET_REAGENTS = ["NaCl", "Citricacid", "MSG"]
OUTPUT_DIR = os.path.join("results", "SI_figs", "AE_raw")
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 9.6
AX_LABEL_FONTSIZE = 17
TICK_FONTSIZE = 13
PANEL_TITLE_FONTSIZE = 16


def try_set_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update(
        {
            "axes.labelsize": AX_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "font.size": TICK_FONTSIZE,
        }
    )


def find_first_ae_file(exp: str, reagent: str, trial: str, grind_min: int) -> str | None:
    ae_dir = os.path.join("data", "ae", exp, reagent, trial)
    pattern = os.path.join(ae_dir, f"*grind{grind_min}min*.csv")
    candidates = sorted(glob.glob(pattern))
    return candidates[0] if candidates else None


def load_signal_voltage(file_path: str, header_lines: int = HEADER_LINES) -> np.ndarray:
    raw = pd.read_csv(
        file_path,
        header=None,
        skiprows=header_lines,
        names=["amplitude"],
        dtype=np.float64,
    )["amplitude"].to_numpy()
    return (raw - ADC_OFFSET) / ADC_SCALE


def downsample_for_plot(signal: np.ndarray, max_points: int) -> np.ndarray:
    if len(signal) <= max_points:
        return signal
    idx = np.linspace(0, len(signal) - 1, max_points, dtype=int)
    return signal[idx]


def time_axis_ms(n: int, sampling_rate_hz: float) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sampling_rate_hz * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create Fig. S1: AE raw signals for exp2 first grinding operations "
            "(25 min, 1st trial) for NaCl, Citricacid, and MSG."
        )
    )
    parser.add_argument("--exp", default="exp2", help="Experiment directory under data/ae (default: exp2).")
    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        default=DEFAULT_SAMPLING_RATE_HZ,
        help="Sampling rate in Hz for time-axis conversion.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=120000,
        help="Number of initial samples to use from each AE file.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=8000,
        help="Max points per subplot after downsampling for drawing.",
    )
    args = parser.parse_args()

    try_set_plot_style()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    excerpt_dir = os.path.join(OUTPUT_DIR, "excerpts")
    os.makedirs(excerpt_dir, exist_ok=True)

    selected = []
    panel_labels = ["(a)", "(b)", "(c)"]
    plot_data = []

    for reagent in TARGET_REAGENTS:
        file_path = find_first_ae_file(args.exp, reagent, TARGET_TRIAL, TARGET_GRIND_MIN)
        if file_path is None:
            raise FileNotFoundError(
                f"No AE file found for {reagent} ({args.exp}/{TARGET_TRIAL}, grind{TARGET_GRIND_MIN}min)."
            )

        signal_v = load_signal_voltage(file_path)
        signal_slice = signal_v[: args.samples]
        signal_plot = downsample_for_plot(signal_slice, args.max_plot_points)

        t_slice_ms = time_axis_ms(len(signal_slice), args.sampling_rate_hz)
        t_plot_ms = time_axis_ms(len(signal_plot), args.sampling_rate_hz)

        excerpt_df = pd.DataFrame(
            {
                "time_ms": t_slice_ms,
                "amplitude_v": signal_slice,
            }
        )
        excerpt_path = os.path.join(excerpt_dir, f"ae_raw_{reagent.lower()}_excerpt.csv")
        excerpt_df.to_csv(excerpt_path, index=False)

        selected.append(
            {
                "reagent": reagent,
                "trial": TARGET_TRIAL,
                "grind_min": TARGET_GRIND_MIN,
                "selected_file": file_path,
                "total_samples_in_file": int(len(signal_v)),
                "samples_exported": int(len(signal_slice)),
                "sampling_rate_hz": args.sampling_rate_hz,
                "excerpt_csv": excerpt_path,
            }
        )
        plot_data.append((reagent, t_plot_ms, signal_plot))

    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True)
    for i, (reagent, t_ms, amp_v) in enumerate(plot_data):
        axes[i].plot(t_ms, amp_v, color="black", linewidth=0.9)
        axes[i].set_ylabel("Voltage (V)")
        axes[i].set_title(f"{panel_labels[i]} {reagent}", loc="left", fontsize=PANEL_TITLE_FONTSIZE)
        axes[i].grid(alpha=0.2, linewidth=0.4)
        axes[i].tick_params(axis="both", labelsize=TICK_FONTSIZE)

    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))

    png_path = os.path.join(OUTPUT_DIR, "ae_raw_exp2_25min_1st.png")
    pdf_path = os.path.join(OUTPUT_DIR, "ae_raw_exp2_25min_1st.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    selected_df = pd.DataFrame(selected)
    manifest_path = os.path.join(OUTPUT_DIR, "ae_raw_selected_files.csv")
    selected_df.to_csv(manifest_path, index=False)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved excerpts in: {excerpt_dir}")


if __name__ == "__main__":
    main()
