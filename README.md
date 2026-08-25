# Acoustic Grinding Analysis

This repository contains the analysis scripts and minimal helpers used to regenerate the included outputs from the released data. 

## Data

Download the dataset archive from Zenodo:

https://doi.org/10.5281/zenodo.18064323

The link is the concept DOI and resolves to the most recent version of the
deposit, which includes the measurements added during revision.

Extract it into the repository root so that the input data are available as:

```text
data/
  ae/
    exp2/         # time-scheduled grinding trials used for model training
    exp3/         # autonomous size-control validation trials
    discussion/   # auxiliary force/speed trials used in the discussion
  powder_size_distribution/
    exp2/
    exp3/
    additional_experiments/
      initial_psd/           # sieved starting powders, before any grinding
      exhaustive_grinding/   # 60 min robotic grinding and the manual reference
```

The `exp2` and `exp3` directory names are dataset labels. The executable scripts below use matching analysis names.

The `data` directory is intentionally ignored by Git. Generated analysis outputs are also ignored and are written to `analysis_results`.

## Environment

This project uses `uv`.

```bash
uv sync
```

## Reproduce Main-Text Analysis Outputs

Run the complete analysis pipeline from the repository root:

```bash
uv run python run_all_main_analysis.py
```

To run individual steps:

```bash
uv run python run_01_plot_frequency_spectra.py
uv run python run_02_plot_ae_power_trends.py
uv run python run_03_train_particle_size_ae_model.py
uv run python run_04_validate_autonomous_stopping.py
uv run python run_05_plot_force_speed_dependence.py
```

Outputs are written under:

```text
analysis_results
  run_01_plot_frequency_spectra
  run_02_plot_ae_power_trends
  run_03_train_particle_size_ae_model
  run_04_validate_autonomous_stopping
  run_05_plot_force_speed_dependence
```

## Reproduce Revision Analyses

These analyses were added when the manuscript was revised. They produce the
figures and tables of the Electronic Supplementary Information and the
screening values quoted in the discussion.

```bash
uv run python run_all_revision_analysis.py
```

To run individual steps:

```bash
uv run python run_06_plot_initial_particle_size.py
uv run python run_07_compare_exhaustive_and_manual_grinding.py
uv run python run_08_check_run_to_run_reproducibility.py
uv run python run_09_compare_lattice_and_plateau.py
uv run python run_10_compute_material_screening_indices.py
uv run python run_11_plot_material_screening_powerlaw.py
uv run python run_12_evaluate_representative_diameters.py
uv run python run_13_build_representative_diameter_table.py
```

Two of them read the output of another script, so mind the order:
`run_11` reads the training dataset exported by `run_03`, and `run_13` reads the
per-metric evaluation produced by `run_12`.

Outputs are written under:

```text
analysis_results
  run_06_plot_initial_particle_size
  run_07_compare_exhaustive_and_manual_grinding
  run_08_check_run_to_run_reproducibility
  run_09_compare_lattice_and_plateau
  run_11_plot_material_screening_powerlaw
  run_12_evaluate_representative_diameters
  run_13_build_representative_diameter_table
```

`run_10_compute_material_screening_indices.py` writes no files; it prints the
two screening indices to the terminal.

## Executable Scripts

`run_01_plot_frequency_spectra.py` generates first/last AE frequency spectrum comparisons for the 25 min grinding trials.

`run_02_plot_ae_power_trends.py` generates AE total spectral power trends with the 4-point moving average.

`run_03_train_particle_size_ae_model.py` trains monotone Bernstein polynomial models from the time-scheduled grinding trials and exports fitted models, metrics, model-validation files, and AE threshold values.

`run_04_validate_autonomous_stopping.py` validates the autonomous size-control stopping trials using the trained particle-size/AE models in the `run_03_train_particle_size_ae_model` output directory.

`run_05_plot_force_speed_dependence.py` generates the force and speed dependence plots used in the discussion.

`run_06_plot_initial_particle_size.py` measures the initial condition of the sieved starting powders and compares it with the earliest ground state.

`run_07_compare_exhaustive_and_manual_grinding.py` compares 60 min of robotic grinding with a manually ground reference, and reports the fine and coarse tails that separate under-grinding from agglomeration.

`run_08_check_run_to_run_reproducibility.py` compares the AE trajectories of the three replicate runs of each material.

`run_09_compare_lattice_and_plateau.py` tabulates the reported lattice parameters and crystal habits against the plateau AE level of each material.

`run_10_compute_material_screening_indices.py` computes the initial-to-steady-state AE ratio and the power-law exponent of the AE feature against the median diameter.

`run_11_plot_material_screening_powerlaw.py` plots the log-log calibration behind that exponent, using the training dataset exported by `run_03_train_particle_size_ae_model.py`.

`run_12_evaluate_representative_diameters.py` trains and evaluates Gaussian process models for the median diameter, the mean and the mode as candidate representative diameters.

`run_13_build_representative_diameter_table.py` builds the comparison table of the three candidate metrics from the evaluation produced by `run_12_evaluate_representative_diameters.py`.

## Helper Modules

These files are imported by the executable scripts and are not normally run directly:

```text
ae_fft.py
bernstein.py
```

## License

This repository is released under the MIT License in [LICENSE](./LICENSE).
