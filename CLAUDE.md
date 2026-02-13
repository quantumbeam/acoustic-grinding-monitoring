# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for predicting particle size distribution (D50) from acoustic emission (AE) signals during powder grinding, using Gaussian Process Regression (GPR). The codebase produces figures, tables, and models for an academic paper.

## Running Scripts

All scripts are run with `uv run python <script>.py`. No build step or test suite exists.

```bash
# Core pipeline (must run exp2 first to generate models)
uv run python exp2_gpr_model.py                          # Train GPR models (outputs .joblib)
uv run python exp3_gpr_control_analysis.py               # Evaluate control reliability using exp2 models
uv run python discussion_AE_bias_and_variance.py         # Discussion: variance analysis
uv run python discussion_force_and_speed_dependene.py    # Discussion: force/speed dependence

# exp2_gpr_model.py supports filtering
uv run python exp2_gpr_model.py --reagent MSG --trial 1st
uv run python exp2_gpr_model.py --reagent all --trial all
```

Dependencies are managed via `pyproject.toml` with `uv`. Python 3.10+.

## Architecture

### Data Pipeline

```
Raw AE CSV (16-bit) → fft_processing.py → ae_power_cache.json → analysis scripts → results/
```

- `fft_processing.py` is the shared utility. It computes FFT power in the 100 kHz–1 MHz band from raw AE signals. All other scripts import `calculate_fft_power()` from it.
- `ae_power_cache.json` caches FFT results keyed by file path + mtime. Scripts read/update this cache to avoid redundant computation.

### Dual-Direction GPR Models (trained in exp2)

- **particle2ae** (forward): D50 target → predicted AE power. Used for threshold-based control (Method P2AE).
- **ae2particle** (inverse): AE power → estimated D50. Used for estimation-based evaluation (Method AE2P).

Models are saved as `results/gpr_model_{direction}_{material}_exp2.joblib`. exp3 and discussion scripts load these models.

### Materials and Trials

Three materials (NaCl, Citricacid, MSG), three trials each (1st, 2nd, 3rd). Data is organized as `data/{ae,powder_size_distribution}/{exp2,exp3}/{Material}/{trial}/*.csv`.

### Script Naming Convention

- `exp2_*` / `exp3_*`: Core experiment scripts aligned with paper sections
- `discussion_*`: Discussion section analyses, outputs go to `results/discussion/`
- `plot_*` / `fig1_*`: Supplementary visualization scripts
- `evaluate_*`: Model evaluation and sensitivity analysis

### Output Structure

- `results/`: Top-level for exp2/exp3 outputs (models, plots, CSVs)
- `results/discussion/`: Discussion section outputs
- `results/SI/`: Supplementary information plots

## Key Conventions

- AE power is computed in V² internally, displayed/plotted in mV² (×1e6 scaling factor `AE_SCALE_TO_MV2`)
- Moving average window size is 4 by default across all scripts
- Matplotlib uses `scienceplots` with large font sizes (font.size=24, axes.labelsize=32) for publication figures
- PSD files contain D50 on lines starting with `Dx (50)`, extracted by `get_d50()`
- AE filenames encode timestamps: `YYYYMMDD_HHMMSS{Material}_grind{N}min.csv`
- Comments and variable names mix Japanese and English
