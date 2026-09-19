# Fast 550-Realization PESTPP-IES Calibration (Two-Stage)

This package runs a fast but defensible two-stage calibration using new files only.
It does not overwrite the previous successful runs.

## Files

- `swat_modflow_ies_dsi550_fast_stage1.pst`
- `swat_modflow_ies_dsi550_fast_stage2.pst`
- `launch_fast550_stage1.ps1`
- `launch_fast550_stage2.ps1`

## Stage 1 (stability-first, fast)

Purpose: one guarded update from prior to avoid unstable first-step jumps.

Settings:

- `ies_num_reals(550)`
- `noptmax = 1`
- `ies_lambda_mults(0.7,1.4)`
- `ies_initial_lambda(1.0)`
- `ies_accept_phi_fac(1.1)`
- `ies_use_approx(true)`
- `ies_subset_size(16)`
- `max_run_fail(50)`

Run:

```powershell
cd D:\nasrin\swatmf_run\fast550_package
.\launch_fast550_stage1.ps1
```

Outputs are in `stage1_run`.

## Stage 2 (speed-first refinement)

Purpose: restart from Stage 1 ensemble and refine quickly.

Settings:

- `noptmax = 3`
- `ies_lambda_mults(1.0)`
- `ies_initial_lambda(1.0)`
- `ies_accept_phi_fac(1.05)`
- `ies_par_en(swat_modflow_ies_dsi550_fast_stage1.1.par.csv)`
- `ies_obs_en(swat_modflow_ies_dsi550_fast_stage1.1.obs.csv)`

Run:

```powershell
cd D:\nasrin\swatmf_run\fast550_package
.\launch_fast550_stage2.ps1
```

Outputs are in `stage2_run`.

## Why this is faster than the old 3-lambda schedule

- Stage 1 uses 2 lambdas (not 3) and only 1 iteration.
- Stage 2 uses 1 lambda per iteration.
- This sharply reduces forward runs per iteration for an expensive model.

## Exactly 8 workers + master

Both launch scripts:

- enforce `NumWorkers = 8`
- start workers `worker1` to `worker8`
- run one master process in a clean run directory

## What to check before Stage 2

In `stage1_run`, confirm these exist:

- `swat_modflow_ies_dsi550_fast_stage1.1.par.csv`
- `swat_modflow_ies_dsi550_fast_stage1.1.obs.csv`
- `.rec` and `.phi.*.csv` files without critical failures

## Early-stop guidance for Stage 2

Stop Stage 2 early if one of these occurs:

- median phi improvement becomes small for two consecutive iterations
- forecast-relevant metrics stop improving
- ensemble spread collapses too strongly

If Stage 2 stalls or destabilizes, rerun Stage 2 with a two-lambda fallback (`0.8,1.2`) for one rescue iteration.
