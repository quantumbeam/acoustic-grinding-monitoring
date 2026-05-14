import argparse
import subprocess
import sys


SCRIPTS = [
    "run_01_plot_frequency_spectra.py",
    "run_02_plot_ae_power_trends.py",
    "run_03_train_particle_size_ae_model.py",
    "run_04_validate_autonomous_stopping.py",
    "run_05_plot_force_speed_dependence.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all main-text analysis scripts in order.")
    parser.parse_args()

    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, "-u", script], check=True)


if __name__ == "__main__":
    main()
