import argparse
import subprocess
import sys


SCRIPTS = [
    "run_06_plot_initial_particle_size.py",
    "run_07_compare_exhaustive_and_manual_grinding.py",
    "run_08_check_run_to_run_reproducibility.py",
    "run_09_compare_lattice_and_plateau.py",
    "run_10_compute_material_screening_indices.py",
    "run_11_plot_material_screening_powerlaw.py",
    "run_12_evaluate_representative_diameters.py",
    "run_13_build_representative_diameter_table.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all revision analysis scripts in order. "
        "run_11 reads the training dataset exported by run_03, so run the "
        "main-text pipeline first."
    )
    parser.parse_args()

    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, "-u", script], check=True)


if __name__ == "__main__":
    main()
