# Acoustic Powder Monitoring

This repository contains data and analysis scripts for a study on predicting particle size distribution from acoustic emission (AE) intensity during powder grinding.

## Data Structure

* `ae_data/`: Contains the raw acoustic emission data.
* `powder_size_distribution_data/`: Contains the particle size distribution data measured after grinding.

Details about the experiments and data organization can be found in the `README.md` files within each directory.

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package and environment management. It is significantly faster than standard `pip`.

### 1. Install uv

If you haven't installed `uv` yet, run the following command:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
source $HOME/.local/bin/env

```

### 2. Environment Setup

Create a virtual environment and install the required libraries:

```bash
# Create a virtual environment (.venv)
uv venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install pandas numpy scikit-learn matplotlib tqdm

```

## Usage (Paper-Aligned)

This README is organized to reflect the paper structure. The core scripts for the paper are **exp2**, **exp3**, and **discussion**. Plotting and evaluation scripts are treated as **supplementary**.

### Core Scripts (exp2 / exp3 / discussion)

#### 1) `exp2_gpr_model.py` (EXP2 main analysis)
Builds the GPR models that map AE-derived features to D50 (and the inverse), using EXP2 data. It manages the AE power cache (`ae_power_cache.json`) and outputs model files, plots, and metrics under `results/`.

```bash
uv run python exp2_gpr_model.py
```

Common options:
```bash
uv run python exp2_gpr_model.py --reagent MSG --trial 1st
uv run python exp2_gpr_model.py --reagent NaCl --trial all
uv run python exp2_gpr_model.py --reagent all --trial all
```

#### 2) `exp3_control_reliability_analysis.py` (EXP3 reliability / control)
Analyzes EXP3 control reliability using AE power and PSD measurements, producing EXP3-focused outputs in `results/`.

```bash
uv run python exp3_control_reliability_analysis.py
```

#### 3) `discussion_AE_bias_and_variance.py` (DISCUSSION analysis)
Quantifies AE bias/variance behavior across trials and materials for the discussion section, and writes plots/CSVs under `results/discussion/`.

```bash
uv run python discussion_AE_bias_and_variance.py
```

#### 4) `discussion_force_and_speed_dependene.py` (DISCUSSION auxiliary)
Additional discussion analysis related to force/speed dependence.

```bash
uv run python discussion_force_and_speed_dependene.py
```

### Supplementary Scripts (plots / evaluation)

#### 5) `evaluate_moving_average_window_gpr.py` (evaluation)
Evaluates GPR performance across moving-average window sizes. Writes CSVs and plots under `results/` and `results/moving_average/`.

```bash
uv run python evaluate_moving_average_window_gpr.py
```

#### 6) `plot_moving_average_timeseries.py` (supplementary plot)
Plots time-series AE power with moving averages for each material/trial.

```bash
uv run python plot_moving_average_timeseries.py
```

#### 7) `plot_d50_ratio_timeseries.py` (supplementary plot)
Plots D50 ratio trends across grinding time.

```bash
uv run python plot_d50_ratio_timeseries.py
```

#### 8) `fig1_schematic_psd.py` (figure)
Generates the schematic PSD figure used in the paper.

```bash
uv run python fig1_schematic_psd.py
```

### Shared Utility (FFT)

#### `fft_processing.py`
Utility module for computing AE power (FFT in the 100 kHz–1 MHz band). This is used by other scripts; typically you do not run it directly unless you are inspecting a single file.

```bash
uv run python fft_processing.py
```
