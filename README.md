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

## Usage

### 1. `fft_processing.py`

This script calculates the total Acoustic Emission (AE) power within the 100 kHz - 1 MHz frequency band for individual AE data files. It applies a Hamming window and performs a Fast Fourier Transform (FFT) as described in the paper.

**Dependencies:** `pandas`, `numpy`

**How to run:**

```bash
python fft_processing.py

```

The script includes a `SAMPLE_FILE` variable that can be modified to process different files. It also demonstrates processing a few example files.

### 2. `gpr_model.py`

This script builds a Gaussian Process Regression (GPR) model to predict D50 particle size from the 4-point moving average of AE power. It creates a dataset by matching time-series AE data (processed by `fft_processing.py`) with corresponding D50 values from `powder_size_distribution_data` files.

**Key Features:**

* **Time-series Matching:** Accurately matches final D50 values with the smoothed AE power at the end of each grinding duration.
* **Combined Data for GPR:** When processing multiple reagents or trials (e.g., using `--reagent all` or `--trial all`), all selected data points are combined into a single dataset to train a more robust GPR model.
* **Trial-specific Visualization:** The resulting plot visually distinguishes data points from different trials using unique markers (e.g., circles, crosses, triangles) to provide clearer insights into experimental variations while still using all data for the global model.

**Model Implementation Details:**
The Gaussian Process Regression model is implemented using `scikit-learn` with a configuration designed to optimize hyperparameters automatically based on the data:

* **Kernel Configuration:** A combination of Constant, RBF (Radial Basis Function), and White Noise kernels is used to capture both the signal trend and noise.
* `kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)`


* **Optimization:** The hyperparameters (length scale, noise level) are initialized at 1.0 and optimized without strict bounds to find the maximum likelihood estimate.
* **Robustness:** `n_restarts_optimizer=10` is used to run the optimizer multiple times with different initializations, preventing the model from getting stuck in local optima.
* **Normalization:** Target values (D50) are normalized (`normalize_y=True`) during training to improve convergence.

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`

**How to run:**

The script automatically manages a cache for pre-calculated Acoustic Emission (AE) power values (`ae_power_cache.json`).

* **First Run / Cache Update:** On the first run, or if the cache is missing or incomplete for the data required by your arguments, the script will automatically calculate the necessary AE power values and update `ae_power_cache.json`. This initial step might take some time (e.g., ~10 minutes for a full dataset), but subsequent runs will be significantly faster as they will load from the cache.
* **Subsequent Runs:** For subsequent runs, the script will load the pre-computed values from `ae_power_cache.json`, significantly speeding up data loading and processing.

You can specify the reagent and trial using command-line arguments:

```bash
# Process only 'NaCl' for the '1st' trial. Cache will be updated if needed.
python gpr_model.py --reagent NaCl --trial 1st

# Process all trials for 'Citricacid'. Cache will be updated if needed.
python gpr_model.py --reagent Citricacid --trial all

# Process all reagents for the '2nd' trial. Cache will be updated if needed.
python gpr_model.py --reagent all --trial 2nd

# Process all reagents and all trials (default behavior). Cache will be updated if needed.
python gpr_model.py

```

**Output Files:**

* Plots are saved in the `results/` directory with dynamic filenames based on the reagent and trial (e.g., `results/gpr_plot_NaCl_1st.png`, `results/gpr_plot_all_all.png`).
* A CSV file containing R-squared and average variance metrics is also saved in the `results/` directory. If `--reagent all` is used, the CSV will contain metrics for each reagent processed (e.g., `results/gpr_metrics_by_reagent_all.csv`). Otherwise, it will be specific to the chosen reagent (e.g., `results/gpr_metrics_NaCl_all.csv`).
