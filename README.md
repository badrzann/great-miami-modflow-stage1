# GMRW DSI-550 and Forward-Run Consolidated Archive

This repository is a consolidated working archive for the Great Miami River Watershed (GMRW) SWAT-MODFLOW3 calibration workflow. It brings together the key DSI-550, prior/calibration ensemble, PEST++ artifact, observation, and final result files in one place for easier access, review, and transfer.

## Purpose

The files here document and support the calibration workflow used for:

- DSI-550 / fast-ensemble calibration runs
- prior and calibration ensemble analysis
- PEST++ parameter/observation updates
- groundwater observation processing
- final lithology / output summaries

This is a copy-based archive of the working project data. It is intentionally organized for readability and reproducibility, while the original model directories remain in place.

## Repository structure

```text
DSI550_FORWARD/
├── 01_DSI550_core/
│   ├── broad_gsa/
│   ├── calibration_results_all_stages/
│   ├── dsi_results_so_far.txt
│   ├── stage1_run/
│   ├── stage2_run/
│   ├── stage3_run/
│   ├── stage3_run_old/
│   ├── README_fast550.md
│   ├── swat_modflow_ies_dsi550_fast_stage1.pst
│   └── swat_modflow_ies_dsi550_fast_stage2.pst
├── 02_dsi550_ensemble/
│   └── swat_modflow_ies_new.*
├── 03_ensembles/
│   ├── CALIBRATION_ENSEMBLE/
│   └── PRIOR_ENSEMBLE/
├── 04_dsi_results/
│   ├── DSI_CALVAL268/
│   ├── DSI_POSTERIOR/
│   ├── DSI_PRIOR100/
│   └── DATA_WORTH/
├── 05_pest_families/
│   ├── swat_modflow_full_v2.*
│   ├── swat_modflow_no3.*
│   ├── swat_modflow_no3_500.*
│   ├── swat_modflow_ies_new.*
│   └── swat_modflow_ies_prior100.*
├── 06_observations/
│   ├── obs_data/
│   └── 104_wells/
├── 07_jacobian/
│   └── swat_modflow_ies_new.jco
├── 08_dsi_figures/
│   ├── annual_gw_head_uncertainty_dsi_corrected.*
│   ├── dsi_head_distribution.*
│   └── dsi_streamflow_distribution.*
├── 09_forward_run_scripts/
│   ├── forward_run.py
│   ├── forward_run_fullperiod.py
│   ├── forward_run_no3.py
│   └── forward_run_v5.py
├── 10_RESULTS_lithology/
│   ├── lithology_master.csv
│   ├── lithology_qa.csv
│   └── lithology_results.zip
├── .git/
└── README.md
```

## Key folders

### 01_DSI550_core
This holds the fast 550-realization PESTPP-IES calibration package. It contains the stage 1 and stage 2 calibration files, stage outputs, and notes related to the fast DSI workflow.

### 02_dsi550_ensemble
This folder stores the DSI-550 ensemble outputs, including the main `swat_modflow_ies_new.*` files.

### 03_ensembles
This contains the calibration and prior ensemble runs used for the ensemble calibration workflow:

- `CALIBRATION_ENSEMBLE`
- `PRIOR_ENSEMBLE`

### 04_dsi_results
This holds the result summaries and supporting plots for DSI and data-worth analyses, including:

- `DSI_CALVAL268`
- `DSI_POSTERIOR`
- `DSI_PRIOR100`
- `DATA_WORTH`

### 05_pest_families
This contains the core PEST++ families used in the calibration process, including:

- `swat_modflow_full_v2.*`
- `swat_modflow_no3.*`
- `swat_modflow_no3_500.*`
- `swat_modflow_ies_new.*`
- `swat_modflow_ies_prior100.*`

### 06_observations
This includes the processed observations used for calibration, including:

- streamflow and head target files
- combined observation vectors
- well metadata / 104-well observation context

### 07_jacobian
This folder contains the Jacobian output for the DSI-550 run to support sensitivity and update diagnostics.

### 08_dsi_figures
This folder stores the main figure products used to summarize the DSI results and uncertainty diagnostics.

### 09_forward_run_scripts
This contains the forward-model driver scripts for the main calibration experiments.

### 10_RESULTS_lithology
This contains the final lithology outputs produced from OCR-derived well logs:

- `lithology_master.csv`
- `lithology_qa.csv`
- `lithology_results.zip`

## Observation notes

The observation workflow is represented in the `06_observations/obs_data` section. The processing file `README_observations.txt` describes the streamflow and groundwater head targets and the combined observation vector structure used in calibration.

## Rationale for this archive

This package is designed to provide a clean, portable, and minimally redundant snapshot of the most relevant calibration artifacts. It is meant to keep the key DSI and forward-run files together without mixing in the very large model runtime directories, full backups, and worker copies.

## Recommended use

1. Start with `01_DSI550_core/README_fast550.md` for the fast DSI calibration workflow.
2. Review `06_observations/` for the calibration target definitions.
3. Inspect `05_pest_families/` for PEST++ parameter and observation files.
4. Use `08_dsi_figures/` and `04_dsi_results/` to assess DSI performance.
5. Use `10_RESULTS_lithology/` for the final lithology product outputs.

## Notes

- This is a consolidated copy, not a replacement for the full working model directories.
- Original model, ensemble, and worker runs remain elsewhere in the project tree.
- The included data are intended for discussion, reuse, and archival review.
