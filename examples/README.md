# Examples

This folder contains small sample rainfall time-series CSV files that can be used to test EVT-AutoThresh quickly.

## Files included
- `GP 3816.csv`
- `GP 3820.csv`
- `GP 4356.csv`

Each file is a daily rainfall time series with two columns:

- `date` : date/time (any format pandas can parse)
- `value`: rainfall amount (numeric)

## How to run (GUI)
1. Install requirements (from repo root):
   - `pip install -r requirements.txt`

2. Start the GUI:
   - `python EVT_AutoThresh.py`

3. In the GUI, select any of the example CSV files above as input and run the analysis.

See also: `HOW_TO_RUN.txt`
