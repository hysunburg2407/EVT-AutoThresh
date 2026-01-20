# EVT-AutoThresh: A GUI tool to automate the Threshold Selection process for extreme events extraction under Extreme Value Theory assumptions

- Author: Shubham Dixit & Kamlesh Kumar Pandey
- Department of Civil Engineering
- Indian Institute of Technology (BHU), Varanasi, India 

## Relevant Work
This repository contains code for a paper currently under review.

## Abstract
Extreme Value Theory (EVT)-based peaks-over-threshold (POT) analysis is applied to quantify hydrometeorological extremes, yet threshold selection remains a major bottleneck for reproducible modelling. In practice, thresholds are often set using fixed-percentile rules, which can yield exceedance sets that represent different degrees of extremeness across locations. As an alternative, thresholds may be selected from diagnostic plots, but the process is interpretive and therefore difficult to reproduce and scale. EVT-AutoThresh is presented as a GUI-based tool for automated right-tail threshold selection for positive, unbounded variables. Generalized Pareto models are fitted across candidate thresholds, and the onset of tail behaviour is identified using EVT-consistent evidence from parameter-stability and return-level agreement diagnostics, including checks against the exponential limiting case. Batch processing, data-quality screening, and standardized reporting are implemented for large-sample applications. An India-wide rainfall case study is presented, revealing spatial variability in EVT-consistent thresholds and underscoring the need for automated, diagnostic-driven threshold selection tools.

## Quickstart (GUI)

Run (from the repository root):

- pip install -r requirements.txt
- python EVT_AutoThresh.py

More details: see `examples/HOW_TO_RUN.txt`.

## Inputs
- Time series files (CSV) with two columns: `date`, `value` (any date format pandas can parse).
- Multiple files can be processed in one run.

## Outputs
- Per-file results table (selected threshold and key diagnostics).
- Summary table across all files.
- Optional diagnostic plots for reporting (if enabled in the GUI).

## How to try (example)
See **`examples/`** for sample input files and instructions (`HOW_TO_RUN.txt`).

