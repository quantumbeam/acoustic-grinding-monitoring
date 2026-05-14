# Acoustic Grinding Analysis

This repository contains the analysis scripts and minimal helpers used to regenerate the included outputs from the released data. 

## Data

Download the dataset archive from Zenodo:

https://doi.org/10.5281/zenodo.18064324

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

## Executable Scripts

`run_01_plot_frequency_spectra.py` generates first/last AE frequency spectrum comparisons for the 25 min grinding trials.

`run_02_plot_ae_power_trends.py` generates AE total spectral power trends with the 4-point moving average.

`run_03_train_particle_size_ae_model.py` trains monotone Bernstein polynomial models from the time-scheduled grinding trials and exports fitted models, metrics, model-validation files, and AE threshold values.

`run_04_validate_autonomous_stopping.py` validates the autonomous size-control stopping trials using the trained particle-size/AE models in the `run_03_train_particle_size_ae_model` output directory.

`run_05_plot_force_speed_dependence.py` generates the force and speed dependence plots used in the discussion.

## Helper Modules

These files are imported by the executable scripts and are not normally run directly:

```text
ae_fft.py
bernstein.py
```

## License

This repository is released under the MIT License in [LICENSE](./LICENSE).
