"""
Stage 3 IES ensemble metrics: NSE, KGE, PBIAS, PICP, MPIW
for flow_cal and flow_val periods.
"""

import numpy as np
import pandas as pd
import re
import sys

# ── paths ────────────────────────────────────────────────────────────────────
STAGE3_DIR = r"D:\nasrin\swatmf_run\fast550_package\stage3_run"
PST_FILE   = f"{STAGE3_DIR}\\swat_modflow_ies_dsi550_fast_stage3.pst"
OBS_CSV    = f"{STAGE3_DIR}\\swat_modflow_ies_dsi550_fast_stage3.1.obs.csv"
PI_LOWER   = 2.5   # percentile
PI_UPPER   = 97.5  # percentile

# ── 1. parse PST observations ─────────────────────────────────────────────────
print("Reading PST observations...")
obs_records = []
in_obs = False
with open(PST_FILE, "r") as fh:
    for line in fh:
        stripped = line.strip()
        if re.match(r'^\* observation data', stripped, re.IGNORECASE):
            in_obs = True
            continue
        if in_obs and stripped.startswith("*"):
            break
        if in_obs and stripped:
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    obs_records.append({
                        "name":   parts[0].lower(),
                        "obsval": float(parts[1]),
                        "weight": float(parts[2]),
                        "group":  parts[3].lower(),
                    })
                except ValueError:
                    pass

pst_obs = pd.DataFrame(obs_records).set_index("name")
print(f"  PST obs loaded: {len(pst_obs)} total")
print(f"  Groups: {pst_obs['group'].value_counts().to_dict()}")

# ── 2. load ensemble obs CSV ──────────────────────────────────────────────────
print("\nReading ensemble obs CSV...")
ens = pd.read_csv(OBS_CSV, index_col=0)
ens.columns = [c.lower() for c in ens.columns]
# drop BASE row if present for ensemble stats (keep for reference)
ens_nobase = ens[ens.index.astype(str).str.lower() != "base"]
print(f"  Realizations (excl. BASE): {len(ens_nobase)}")
print(f"  Obs columns: {len(ens.columns)}")

# ── 3. metric functions ───────────────────────────────────────────────────────
def nse(obs, sim):
    num = np.sum((obs - sim) ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - num / den if den > 0 else np.nan

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta  = np.mean(sim) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def pbias(obs, sim):
    return 100.0 * np.sum(obs - sim) / np.sum(obs) if np.sum(obs) != 0 else np.nan

def picp_mpiw(obs, ens_vals):
    """PI coverage probability and mean interval width at PI_LOWER/PI_UPPER."""
    lo = np.percentile(ens_vals, PI_LOWER, axis=0)
    hi = np.percentile(ens_vals, PI_UPPER, axis=0)
    covered = np.sum((obs >= lo) & (obs <= hi))
    picp = 100.0 * covered / len(obs)
    mpiw = np.mean(hi - lo)
    return picp, mpiw

# ── 4. compute per group ──────────────────────────────────────────────────────
def calc_group(group_name):
    group_obs = pst_obs[pst_obs["group"] == group_name]
    cols = [n for n in group_obs.index if n in ens.columns]
    if not cols:
        print(f"  [WARN] No matching columns for {group_name}")
        return
    obs_vals = group_obs.loc[cols, "obsval"].values
    sim_all  = ens_nobase[cols].values          # shape: (n_real, n_obs)
    sim_med  = np.median(sim_all, axis=0)       # median across realizations

    # filter zero-weight obs (validation has weight=0 but we still want metrics)
    mask = np.isfinite(obs_vals) & (obs_vals > 0)
    o  = obs_vals[mask]
    sm = sim_med[mask]
    sa = sim_all[:, mask]

    n_nse   = nse(o, sm)
    n_kge   = kge(o, sm)
    n_pbias = pbias(o, sm)
    n_picp, n_mpiw = picp_mpiw(o, sa)

    print(f"\n  ══ {group_name.upper()} ({len(o)} obs) ══")
    print(f"  NSE   = {n_nse:.3f}")
    print(f"  KGE   = {n_kge:.3f}")
    print(f"  PBIAS = {n_pbias:.2f}%")
    print(f"  PICP  = {n_picp:.1f}%  ({PI_LOWER}–{PI_UPPER}% PI)")
    print(f"  MPIW  = {n_mpiw:.1f} m³/s")

print("\n" + "="*55)
print(" STAGE 3 — ITERATION 1 — ENSEMBLE METRICS (median sim)")
print("="*55)
calc_group("flow_cal")
calc_group("flow_val")
print("\nDone.")
