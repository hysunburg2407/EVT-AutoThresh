# EVT-AutoThresh: A GUI tool to automate threshold selection for extreme-event extraction under Extreme Value Theory assumptions

- Authors: Shubham Dixit and Kamlesh Kumar Pandey
- Department of Civil Engineering
- Indian Institute of Technology (BHU), Varanasi, India

## Relevant Work
This repository supports the manuscript **“A Reproducible Diagnostic Framework for EVT-Consistent Automatic Threshold Selection in Peaks-over-Threshold Analysis.”**

## Overview
Extreme Value Theory (EVT)-based peaks-over-threshold (POT) analysis is widely used to quantify hydrometeorological extremes, yet threshold selection remains a major bottleneck for reproducible modelling. EVT-AutoThresh formalizes parameter-stability and return-level-agreement diagnostics within an explicit rule-based workflow for automatic threshold selection. The tool evaluates candidate thresholds, fits Generalized Pareto models, examines parameter stability and GPD–exponential return-level agreement, and reports the resulting threshold together with the supporting diagnostics. A graphical interface and batch-processing workflow support both individual and multi-site analyses.

## Quickstart (GUI)

Run from the repository root:

- `pip install -r requirements.txt`
- `python EVT_AutoThresh.py`

More details are provided in `examples/HOW_TO_RUN.txt`.

## Inputs
- Time-series files (CSV) with two columns: `date`, `value` (using any date format that pandas can parse).
- Multiple files can be processed in one run.

## Outputs
- Per-file threshold-wise results and diagnostics.
- Final selected threshold for each input series.
- Summary output across multiple files.
- Diagnostic plots for reporting and verification.

## Example data
See the `examples/` directory for sample input files and instructions.

## Supplementary Material S1
Detailed interface views relocated from the main manuscript are documented in [`supplementary_interface/`](supplementary_interface/). This repository therefore provides the source code, sample data, implementation instructions, and supplementary interface material referenced as **Supplementary Material S1** in the revised manuscript.

