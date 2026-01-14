# EVT-AutoThresh

Automatic threshold selection for rainfall extremes using EVT.  
This repo contains the **engine** (core calculations) and a **Tkinter GUI** for easy use.

## Quickstart (GUI)

Run:
pip install -r requirements.txt
python app_gui/main.py

More details: see `examples/HOW_TO_RUN.txt`.

## Who is this for?
- **Non-coders**: use the GUI (`app_gui`) to run the tool with buttons.
- **Researchers/Power users**: use the engine (`tool_engine`) to script batch runs.

## Inputs
- Time series files (CSV) with two columns: `date`, `value` (any date format pandas can parse).
- Multiple files can be processed in one run.

## Outputs
- Per-file results table (threshold(s), diagnostics).
- Summary table across all files.
- Optional plots for reporting.

## How to try (example)
See **`examples/`** for a tiny sample input and instructions (`HOW_TO_RUN.txt`, coming soon).

## Repository structure
- app_gui/      # Tkinter app (buttons/windows); calls the engine
- tool_engine/  # Core calculations: I/O, preprocessing, analysis, reporting
- examples/     # Small sample data and how-to-run notes
- requirements.txt
- LICENSE
