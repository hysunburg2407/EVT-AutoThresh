# EVT-AutoThresh

Automatic threshold selection for rainfall extremes using EVT.  
This repository provides the complete tool (**Tkinter GUI + core logic**) in a **single Python file** for easy review and use.

## Quickstart (GUI)

Run (from the repository root):

- pip install -r requirements.txt
- python EVT_AutoThresh.py

More details: see `examples/HOW_TO_RUN.txt`.

## Who is this for?
- **Anyone**: download the repo and run the tool with the commands above.
- **Reviewers/Researchers**: full source code is provided in `EVT_AutoThresh.py` for transparency and reproducibility.

## Inputs
- Time series files (CSV) with two columns: `date`, `value` (any date format pandas can parse).
- Multiple files can be processed in one run.

## Outputs
- Per-file results table (selected threshold and key diagnostics).
- Summary table across all files.
- Optional diagnostic plots for reporting (if enabled in the GUI).

## How to try (example)
See **`examples/`** for sample input files and instructions (`HOW_TO_RUN.txt`).

## Repository structure
- `EVT_AutoThresh.py` : complete tool (GUI + core logic)
- `examples/` : sample rainfall CSV files + run guide
- `requirements.txt`
- `LICENSE`
