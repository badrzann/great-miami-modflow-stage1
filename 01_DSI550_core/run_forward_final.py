#!/usr/bin/env python3
r"""
run_forward_final.py
====================
Perform a single forward run using the Stage-3 Iteration-3 final calibrated
parameters from:
    stage3_run/stage3.3.base.par   (MAP / deterministic best-estimate)

Workflow
--------
1. Read final calibrated parameters from stage3.3.base.par
2. Back up current pest_pars.dat in worker1
3. Write final params to worker1/pest_pars.dat
4. Run forward_run_fullperiod.py from worker1 (applies SWAT + MODFLOW params,
   runs SWAT-MODFLOW3.exe, writes sim_flow_fp.dat)
5. Read simulated flow (252 monthly values) from worker1/sim_flow_fp.dat
6. Load observed flow from the Stage-3 PST observation section
7. Compute NSE, R², RMSE, KGE, PBIAS for:
      flow_cal : 2003-01 → 2017-12  (180 months, calibration)
      flow_val : 2018-01 → 2023-12  ( 72 months, validation)
8. Print comparison table vs Stage-3 Iter-3 ensemble-median metrics
9. Save: forward_run_final_results/forward_run_metrics.csv
         forward_run_final_results/simulated_vs_observed.csv

Usage
-----
    cd D:\nasrin\swatmf_run\fast550_package
    D:\GMRW\.venv\Scripts\python.exe run_forward_final.py
"""

import os, sys, re, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\nasrin\swatmf_run\fast550_package")
STAGE3_DIR  = BASE_DIR / "stage3_run"
WORKER_DIR  = Path(r"D:\nasrin\swatmf_run\worker1")
PYTHON_EXE  = Path(r"D:\GMRW\.venv\Scripts\python.exe")

PAR_FILE    = STAGE3_DIR / "swat_modflow_ies_dsi550_fast_stage3.3.base.par"  # final MAP params
PEST_DAT    = WORKER_DIR / "pest_pars.dat"               # model reads this
PST_FILE    = WORKER_DIR / "swat_modflow_ies_dsi550_fast_stage3.pst"
SIM_FILE    = WORKER_DIR / "sim_flow_fp.dat"
FWD_SCRIPT  = WORKER_DIR / "forward_run_fullperiod.py"

OUT_DIR     = BASE_DIR / "forward_run_final_results"

# ── Periods ────────────────────────────────────────────────────────────────
CAL_START, CAL_END = "2003-01", "2017-12"   # 180 months
VAL_START, VAL_END = "2018-01", "2023-12"   #  72 months
N_CAL, N_VAL = 180, 72

# ── Stage-3 Iter-3 ensemble-median metrics for reference ──────────────────
ITER3_METRICS = {
    "flow_cal": {"NSE": 0.763, "R2": 0.763, "RMSE": 55.60, "KGE": 0.792, "PBIAS": -5.08},
    "flow_val": {"NSE": 0.773, "R2": 0.773, "RMSE": 53.25, "KGE": 0.853, "PBIAS":  2.64},
}

# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

def read_base_par(path: Path) -> dict:
    """Read PEST .par file (single point format)  →  {name: value}."""
    params = {}
    header_seen = False
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("single") and not header_seen:
            header_seen = True
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                params[parts[0].lower()] = float(parts[1])
            except ValueError:
                pass
    return params


def write_pest_dat(path: Path, params: dict) -> None:
    """Write params to pest_pars.dat in two-column format (name  value)."""
    lines = []
    for name, val in params.items():
        # Use exponential notation for very small/large floates
        if abs(val) > 0 and (abs(val) < 1e-4 or abs(val) >= 1e6):
            lines.append(f"{name:<14}{val:.12e}")
        else:
            lines.append(f"{name:<14}{val:.12f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="latin-1")


def extract_obs_from_pst(pst_path: Path) -> pd.DataFrame:
    """Parse PST observation section → DataFrame with obs_name, value, weight, group."""
    rows = []
    in_obs = False
    for line in Path(pst_path).read_text(encoding="latin-1").splitlines():
        ls = line.strip()
        if re.match(r'^\* observation data', ls, re.IGNORECASE):
            in_obs = True
            continue
        if re.match(r'^\*', ls) and in_obs:
            in_obs = False
        if in_obs and ls.startswith("flow_"):
            parts = ls.split()
            if len(parts) >= 4:
                rows.append({
                    "obs_name": parts[0],
                    "value":    float(parts[1]),
                    "weight":   float(parts[2]),
                    "group":    parts[3],
                })
    return pd.DataFrame(rows)


# ── Metrics ────────────────────────────────────────────────────────────────

def nse(obs, sim):
    num = np.sum((obs - sim) ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - num / den if den > 0 else np.nan

def kge(obs, sim):
    r   = np.corrcoef(obs, sim)[0, 1]
    b   = np.mean(sim) / np.mean(obs)
    g   = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    return 1.0 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)

def pbias(obs, sim):
    return 100.0 * np.sum(obs - sim) / np.sum(obs)

def rmse(obs, sim):
    return np.sqrt(np.mean((obs - sim) ** 2))

def r2(obs, sim):
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def compute_metrics(obs_arr, sim_arr) -> dict:
    obs_arr = np.asarray(obs_arr, dtype=float)
    sim_arr = np.asarray(sim_arr, dtype=float)
    mask = (obs_arr > 0) & np.isfinite(obs_arr) & np.isfinite(sim_arr)
    o, s = obs_arr[mask], sim_arr[mask]
    return {
        "NSE":   round(nse(o, s), 4),
        "R2":    round(r2(o, s), 4),
        "RMSE":  round(rmse(o, s), 4),
        "KGE":   round(kge(o, s), 4),
        "PBIAS": round(pbias(o, s), 4),
        "N":     int(mask.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ── 1. Read final parameters ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  STAGE-3 FORWARD RUN WITH FINAL CALIBRATED PARAMETERS")
    print(f"{'='*65}")
    print(f"\n  Parameter source : {PAR_FILE}")

    if not PAR_FILE.exists():
        sys.exit(f"ERROR: {PAR_FILE} not found.")
    params = read_base_par(PAR_FILE)
    print(f"\n  Final calibrated parameters ({len(params)} total):")
    for k, v in params.items():
        print(f"    {k:<14} = {v}")

    # ── 2. Back up current pest_pars.dat ─────────────────────────────────
    bak = WORKER_DIR / "pest_pars.dat.pre_forward_bak"
    if PEST_DAT.exists():
        shutil.copy2(PEST_DAT, bak)
        print(f"\n  Backed up pest_pars.dat → {bak.name}")

    # ── 3. Write final params to pest_pars.dat ────────────────────────────
    write_pest_dat(PEST_DAT, params)
    print(f"  Written final params to {PEST_DAT}")
    print("  pest_pars.dat content:")
    for line in PEST_DAT.read_text(encoding="latin-1").splitlines():
        print(f"    {line}")

    # ── 4. Run forward model ──────────────────────────────────────────────
    print(f"\n  Running  {FWD_SCRIPT.name}  ...  [{time.strftime('%H:%M:%S')}]")
    print(f"  Working dir : {WORKER_DIR}")
    print(f"  Python      : {PYTHON_EXE}")
    print("-" * 65)
    t0 = time.time()

    proc = subprocess.Popen(
        [str(PYTHON_EXE), str(FWD_SCRIPT)],
        cwd=str(WORKER_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace"
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    elapsed = time.time() - t0
    print("-" * 65)
    print(f"  Finished in {elapsed:.1f}s  (exit={proc.returncode})  [{time.strftime('%H:%M:%S')}]")

    if proc.returncode != 0:
        sys.exit(f"\nERROR: forward_run_fullperiod.py exited with code {proc.returncode}")

    # ── 5. Read simulated flow ────────────────────────────────────────────
    if not SIM_FILE.exists():
        sys.exit(f"ERROR: {SIM_FILE} not found after run.")
    sim_values = [float(v) for v in SIM_FILE.read_text().splitlines() if v.strip()]
    if len(sim_values) != N_CAL + N_VAL:
        print(f"\n  WARNING: Expected {N_CAL + N_VAL} sim values, got {len(sim_values)}")
    sim_cal = np.array(sim_values[:N_CAL])
    sim_val = np.array(sim_values[N_CAL:N_CAL + N_VAL])
    print(f"\n  Simulated flow read: {len(sim_values)} values")
    print(f"    Cal period ({CAL_START}–{CAL_END}): {len(sim_cal)} values, "
          f"mean={sim_cal.mean():.2f}, min={sim_cal.min():.2f}, max={sim_cal.max():.2f}")
    print(f"    Val period ({VAL_START}–{VAL_END}): {len(sim_val)} values, "
          f"mean={sim_val.mean():.2f}, min={sim_val.min():.2f}, max={sim_val.max():.2f}")

    # ── 6. Load observed flow from PST ────────────────────────────────────
    obs_df = extract_obs_from_pst(PST_FILE)
    obs_cal = obs_df[obs_df["group"] == "flow_cal"]["value"].values
    obs_val = obs_df[obs_df["group"] == "flow_val"]["value"].values
    print(f"\n  Observed flow loaded: {len(obs_cal)} cal + {len(obs_val)} val")

    if len(obs_cal) != N_CAL or len(obs_val) != N_VAL:
        print(f"  WARNING: obs counts ({len(obs_cal)}, {len(obs_val)}) don't match "
              f"expected ({N_CAL}, {N_VAL})")

    n = min(len(obs_cal), len(sim_cal))
    metrics_cal = compute_metrics(obs_cal[:n], sim_cal[:n])

    n = min(len(obs_val), len(sim_val))
    metrics_val = compute_metrics(obs_val[:n], sim_val[:n])

    # ── 7. Print comparison table ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FORWARD RUN METRICS  vs  Stage-3 Iter-3 Ensemble Median")
    print(f"{'='*65}")
    print(f"\n  {'Period':<12} {'Metric':<8} {'Forward Run':>12} {'Iter-3 Median':>14}")
    print(f"  {'-'*50}")

    for period, m_fwd, ref_key in [
        ("flow_cal", metrics_cal, "flow_cal"),
        ("flow_val", metrics_val, "flow_val"),
    ]:
        ref = ITER3_METRICS[ref_key]
        for metric in ["NSE", "R2", "RMSE", "KGE", "PBIAS"]:
            fwd_val = m_fwd[metric]
            ref_val = ref[metric]
            diff    = fwd_val - ref_val
            diff_str = f"({diff:+.3f})"
            print(f"  {period:<12} {metric:<8} {fwd_val:>12.4f} {ref_val:>12.3f}  {diff_str}")
        print()

    # ── 8. Save results ───────────────────────────────────────────────────
    # Metrics CSV
    rows = []
    for period, m, ref_key in [
        ("flow_cal", metrics_cal, "flow_cal"),
        ("flow_val", metrics_val, "flow_val"),
    ]:
        ref = ITER3_METRICS[ref_key]
        for metric in ["NSE", "R2", "RMSE", "KGE", "PBIAS"]:
            rows.append({
                "period":        period,
                "metric":        metric,
                "forward_run":   m[metric],
                "iter3_median":  ref[metric],
                "difference":    round(m[metric] - ref[metric], 4),
            })
    metrics_df = pd.DataFrame(rows)
    metrics_out = OUT_DIR / "forward_run_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"  Saved metrics → {metrics_out}")

    # Simulated vs Observed CSV
    dates_cal = pd.date_range(CAL_START, periods=N_CAL, freq="MS")
    dates_val = pd.date_range(VAL_START, periods=N_VAL, freq="MS")
    df_cal = pd.DataFrame({
        "date":    dates_cal,
        "period":  "flow_cal",
        "obs_cms": obs_cal[:len(dates_cal)] if len(obs_cal) >= len(dates_cal) else np.full(len(dates_cal), np.nan),
        "sim_cms": sim_cal[:len(dates_cal)],
    })
    df_val = pd.DataFrame({
        "date":    dates_val,
        "period":  "flow_val",
        "obs_cms": obs_val[:len(dates_val)] if len(obs_val) >= len(dates_val) else np.full(len(dates_val), np.nan),
        "sim_cms": sim_val[:len(dates_val)],
    })
    ts_df = pd.concat([df_cal, df_val], ignore_index=True)
    ts_out = OUT_DIR / "simulated_vs_observed.csv"
    ts_df.to_csv(ts_out, index=False)
    print(f"  Saved time series → {ts_out}")

    print(f"\n{'='*65}")
    print("  FORWARD RUN COMPLETE")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
