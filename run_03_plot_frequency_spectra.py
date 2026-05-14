import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = "exp2"
TRIAL = "1st"
TARGET_GRIND_MIN = 25
HEADER_LINES = 12
SAMPLING_RATE_HZ = 2_000_000.0
START_SAMPLE_INDEX = 200_013
END_SAMPLE_INDEX = 1_200_012
OUTPUT_DIR = os.path.join("analysis_results", "run_03_plot_frequency_spectra")
MATERIALS = ["NaCl", "Citricacid", "MSG"]
OUTPUT_NAME_MAP = {
    "NaCl": "NaCl",
    "Citricacid": "Citricacid",
    "MSG": "Ajinomoto",
}


def try_set_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
        }
    )


def find_grind_files(material: str, grind_min: int) -> list[str]:
    pattern = os.path.join(
        "data",
        "ae",
        EXPERIMENT,
        material,
        TRIAL,
        f"*grind{grind_min}min*.csv",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for {material} grind{grind_min}min ({TRIAL}).")
    return files


def load_voltage_signal(file_path: str) -> np.ndarray:
    raw = pd.read_csv(
        file_path,
        header=None,
        skiprows=HEADER_LINES,
        names=["amplitude"],
        dtype=np.float64,
    )["amplitude"].to_numpy()
    signal_v = (raw - 32768.0) / 32768.0
    end_idx = min(END_SAMPLE_INDEX, len(signal_v))
    if START_SAMPLE_INDEX >= end_idx:
        raise ValueError(f"Invalid sample window for file: {file_path}")
    return signal_v[START_SAMPLE_INDEX:end_idx]


def fft_amplitude(signal_v: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal_v)
    fft_vals = np.fft.rfft(signal_v)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    amp = (2.0 / n) * np.abs(fft_vals)
    return freqs, amp


def main() -> None:
    try_set_plot_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest_rows = []

    for material in MATERIALS:
        grind_files = find_grind_files(material, TARGET_GRIND_MIN)
        first_file = grind_files[0]
        last_file = grind_files[-1]

        sig_first = load_voltage_signal(first_file)
        sig_last = load_voltage_signal(last_file)

        f_first, a_first = fft_amplitude(sig_first, SAMPLING_RATE_HZ)
        f_last, a_last = fft_amplitude(sig_last, SAMPLING_RATE_HZ)

        # Plot 50-1000 kHz and scale y-axis within this visible range.
        mask_first = (f_first >= 50_000) & (f_first <= 1_000_000)
        mask_last = (f_last >= 50_000) & (f_last <= 1_000_000)

        a_first_mv = a_first[mask_first] * 1e3
        a_last_mv = a_last[mask_last] * 1e3
        freq_first_khz = f_first[mask_first] / 1000.0
        freq_last_khz = f_last[mask_last] / 1000.0

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(
            freq_first_khz,
            a_first_mv,
            color="#E74C3C",
            lw=1.0,
            alpha=0.95,
            label="First grind",
        )
        ax.plot(
            freq_last_khz,
            a_last_mv,
            color="#555555",
            lw=1.0,
            linestyle=(0, (2, 1)),
            alpha=0.9,
            label="Last grind",
        )
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Amplitude (mV)")
        ax.set_xlim(50, 1000)
        y_max = max(float(np.max(a_first_mv)), float(np.max(a_last_mv)))
        ax.set_ylim(0.0, y_max * 1.05 if y_max > 0 else 0.1)
        ax.legend(frameon=False)
        fig.tight_layout()

        out_stem = f"fft_plot_{OUTPUT_NAME_MAP[material]}"
        png_path = os.path.join(OUTPUT_DIR, f"{out_stem}.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        manifest_rows.append(
            {
                "material": material,
                "label_for_output": OUTPUT_NAME_MAP[material],
                "trial": TRIAL,
                "grind_min": TARGET_GRIND_MIN,
                "n_files_for_grind": len(grind_files),
                "first_file": first_file,
                "last_file": last_file,
                "png": png_path,
            }
        )
        print(f"Saved: {png_path}")

    manifest_path = os.path.join(OUTPUT_DIR, "frequency_spectra_first_last_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
