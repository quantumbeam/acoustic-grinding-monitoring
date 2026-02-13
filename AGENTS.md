# Repository Guidelines

## Project Structure & Module Organization
This repository is script-centric (no `src/` package). Main analysis scripts live at the repo root:
- `exp2_*.py`: EXP2 model training/evaluation (GPR and related plots).
- `exp3_*.py`: EXP3 control analysis (for example `exp3_gpr_control_analysis.py`).
- `discussion_*.py`: discussion-specific analyses (for example `discussion_exp3_gpr_ae2p_evaluation.py`).
- `fft_processing.py`: shared FFT/AE power utility used by multiple scripts.

Data and generated artifacts:
- `data/ae/{exp2,exp3,discussion}/` and `data/powder_size_distribution/{exp2,exp3}/`
- `results/`, including `results/discussion/` and `results/SI/`

## Build, Test, and Development Commands
Use `uv` with Python 3.10+.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Run key scripts:

```bash
uv run python exp2_gpr_model.py
uv run python exp3_gpr_control_analysis.py
uv run python discussion_exp3_gpr_ae2p_evaluation.py
```

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables and descriptive uppercase constants (`RESULTS_DIR`).
- Keep script names explicit by scope and method: `exp2_*`, `exp3_*`, `discussion_*`, and include method tags like `gpr` when relevant.
- Prefer small helper functions for repeated parsing/calculation logic.

## Testing Guidelines
There is no formal unit test suite currently. Validate changes with script-level smoke runs and output checks:
- Confirm scripts execute with `uv run python <script>.py`.
- Verify expected outputs are created in `results/...` (CSV/PDF/PNG).
- For refactors, compare key CSV columns and row counts before/after.

## Commit & Pull Request Guidelines
Git history uses short, imperative messages (for example: `Add isotonic plot`, `Refactor code structure...`, `Update plot`).
- Commit format: `<Verb> <scope>` (example: `Refactor exp3 AE2P export`).
- Keep commits focused on one change area.
- PRs should include:
  1. Purpose and affected scripts.
  2. Repro commands used.
  3. Output paths changed (for example `results/discussion/ae2p_upper_error_points.csv`).
  4. Before/after figures when plot behavior changes.
