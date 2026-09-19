DSI EVALUATION FOLDER — 550 Realizations
==========================================
Generated: 2026-04-30
Source: D:\nasrin\swatmf_run\fast550_package\stage3_run\

Calibration setup: PEST++ IES, SWAT-MODFLOW3
Ensemble size: 550 (549 numbered + 1 BASE)

PRIOR DATA
----------
swat_modflow_ies_dsi550_fast_stage1.0.par.csv
  - Original prior parameter ensemble (550 real.) — start of Stage 1
  - Use as: prior par ensemble for DSI

swat_modflow_ies_dsi550_fast_stage3.weights.csv
  - Observation weight ensemble (550 real.)
  - Use as: DSI observation weights

swat_modflow_ies_dsi550_fast_stage3.obs+noise.csv
  - Observed values + realizations of measurement noise (550 real.)
  - Use as: DSI data ensemble (d_obs)

STAGE 2 FINAL (used as Stage 3 restart / intermediate prior)
-------------------------------------------------------------
swat_modflow_ies_dsi550_fast_stage2.3.par.csv   — Stage 2, iteration 3, accepted parameters
swat_modflow_ies_dsi550_fast_stage2.3.obs.csv   — Stage 2, iteration 3, accepted simulated obs

STAGE 3 STARTING POINT (iteration 0)
--------------------------------------
swat_modflow_ies_dsi550_fast_stage3.0.par.csv   — Stage 3 initial par ensemble (same as stage1.0)
swat_modflow_ies_dsi550_fast_stage3.0.obs.csv   — Stage 3 iteration 0 simulated obs (= stage2.3.obs reordered)

STAGE 3 ITERATION 1 — FINAL ACCEPTED (main DSI result)
--------------------------------------------------------
swat_modflow_ies_dsi550_fast_stage3.1.par.csv
  - Posterior parameter ensemble after iteration 1 (550 real.)
  - Use as: final calibrated par ensemble for DSI

swat_modflow_ies_dsi550_fast_stage3.1.obs.csv
  - Posterior simulated obs ensemble after iteration 1 (550 real.)
  - Use as: final simulated obs ensemble for DSI

STAGE 3 ITERATION 1 — REJECTED CANDIDATE
------------------------------------------
swat_modflow_ies_dsi550_fast_stage3.rejected.1.par.csv
swat_modflow_ies_dsi550_fast_stage3.rejected.1.obs.csv
  - Lambda-testing candidates that were rejected (phi was worse)
  - For reference/diagnostics only

CONTROL FILE
------------
swat_modflow_ies_dsi550_fast_stage3.pst
  - PEST++ control file with all parameter/observation definitions
  - Use to read parameter names, observation names, groups (flow_cal, flow_val, head_cal)

PHI SUMMARIES
-------------
swat_modflow_ies_dsi550_fast_stage3.phi.actual.csv   — per-realization actual phi by iteration
swat_modflow_ies_dsi550_fast_stage3.phi.group.csv    — phi by obs group by iteration
swat_modflow_ies_dsi550_fast_stage3.phi.composite.csv — composite phi by iteration

PHI PROGRESS (mean actual phi)
  iteration 0: 7152
  iteration 1:  116  (98.4% reduction)

ITERATION 1 METRICS (median ensemble vs observed)
  flow_cal (180 obs): NSE=0.695, KGE=0.673, PBIAS=-16.94%, PICP=80.0%, MPIW=188 m3/s
  flow_val ( 72 obs): NSE=0.756, KGE=0.792, PBIAS= 1.01%, PICP=97.2%, MPIW=242 m3/s

NOTE: Iteration 2 is currently running. Once complete, copy stage3.2.par.csv and
stage3.2.obs.csv into this folder to update the posterior with the latest results.
