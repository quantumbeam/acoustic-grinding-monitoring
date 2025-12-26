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
curl -LsSf https://astral.sh/uv/install.sh | sh
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
uv pip install pandas numpy scikit-learn matplotlib

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
*   **Time-series Matching:** Accurately matches final D50 values with the smoothed AE power at the end of each grinding duration.
*   **Combined Data for GPR:** When processing multiple reagents or trials (e.g., using `--reagent all` or `--trial all`), all selected data points are combined into a single dataset to train a more robust GPR model.
*   **Trial-specific Visualization:** The resulting plot visually distinguishes data points from different trials using unique markers (e.g., circles, crosses, triangles) to provide clearer insights into experimental variations while still using all data for the global model.

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`

**How to run:**

You can specify the reagent and trial using command-line arguments:

```bash
# Process only 'NaCl' for the '1st' trial
python gpr_model.py --reagent NaCl --trial 1st

# Process all trials for 'Citricacid'
python gpr_model.py --reagent Citricacid --trial all

# Process all reagents for the '2nd' trial
python gpr_model.py --reagent all --trial 2nd

# Process all reagents and all trials (default behavior)
python gpr_model.py
```

**Important Note:** The script requires access to files in the `ae_data` directory. If `ae_data/` is listed in your `.gitignore` file, the script will not be able to read the AE data. Please temporarily remove or comment out `ae_data/` from `.gitignore` before running this script.

The script will save a plot of the GPR results (e.g., `gpr_exp2_results.png`) in the current directory.
