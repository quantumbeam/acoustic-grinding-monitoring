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

(Here, you can add how to run your analysis scripts in the future.)
Example:

```bash
python src/analyze_ae.py

```
