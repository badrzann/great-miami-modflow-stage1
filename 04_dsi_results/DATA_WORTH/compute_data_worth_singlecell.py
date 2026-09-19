"""
compute_data_worth_singlecell.py
================================
Replicate the paper's Table 5 FOSM/GENLINPRED data-worth analysis exactly:

Paper methodology (Section 2.3.2, Table 5):
  - Prediction target: groundwater head at ONE specific cell
  - For each calibration observation (streamflow + each well), compute
    the % increase in posterior predictive variance when that observation
    is individually omitted from the full calibration dataset.
  - Uses FOSM (First-Order Second-Moment) / Schur complement approach.

Paper used GENLINPRED (PEST) with a perturbation Jacobian.
We use the ensemble-based Jacobian from the 100-member prior (ES approach),
which gives the same FOSM result when N is large enough.

FOSM variance formula:
  sigma2_post(O)       = sigma2_prior - C_sp C_oo^{-1} C_ps
  sigma2_post(O\{i})   = sigma2_prior - C_sp' C_oo'^{-1} C_ps'
  data_worth_i (%)     = [sigma2_post(O\{i}) - sigma2_post(O)] / sigma2_post(O) * 100

where O = full calibration dataset, O\{i} = dataset with obs i removed.

We do this for EACH of the 85 well locations as the prediction target
(not just one cell), and also report a summary for a single representative cell.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

BASE = "D:/GMRW/finalresult/swatmf_run"
OUT_DIR = Path("D:/GMRW/finalresult/GMRW_Tables")

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading prior ensemble and observation data...")
prior100 = pd.read_csv(f"{BASE}/swat_modflow_ies_prior100.0.obs.csv", index_col=0)
obs_data = pd.read_csv(f"{BASE}/swat_modflow_ies_prior100.adjusted.obs_data.csv")
obs_data.set_index("name", inplace=True)

all_cols  = prior100.columns.tolist()
cal_all   = [c for c in all_cols if obs_data.loc[c, "weight"] > 0]
cal_flow  = [c for c in cal_all if obs_data.loc[c, "group"] == "flow_cal"]
cal_head  = [c for c in cal_all if obs_data.loc[c, "group"] == "head_cal"]

N = prior100.shape[0]  # 100 realizations
print(f"  Ensemble: {N} realizations, {len(all_cols)} obs")
print(f"  Calibration: {len(cal_all)} (flow: {len(cal_flow)}, head: {len(cal_head)})")

# ── Well locations ─────────────────────────────────────────────────────────────
well_rc = []
with open(f"{BASE}/modflow.obs") as f:
    lines = f.readlines()
for line in lines[2:]:
    parts = line.strip().split()
    if len(parts) >= 2:
        try:
            well_rc.append((int(parts[0]) - 1, int(parts[1]) - 1))
        except ValueError:
            pass

# ── Ensemble matrices ─────────────────────────────────────────────────────────
O = prior100[cal_all].values  # (100, 265)
h = obs_data.loc[cal_all, "value"].values  # (265,)

# Measurement noise
groups  = obs_data.loc[cal_all, "group"].values
sig_d   = np.where(groups == "head_cal", 0.1, 0.1 * np.abs(h))
sig_d   = np.maximum(sig_d, 0.01)
C_d_diag = sig_d ** 2  # (265,)

# Anomalies
o_bar = O.mean(axis=0)  # (265,)
A_o   = O - o_bar       # (100, 265)

# C_oo = A_o^T A_o / (N-1) + diag(C_d)
C_oo = A_o.T @ A_o / (N - 1) + np.diag(C_d_diag)  # (265, 265)
C_oo_inv = np.linalg.inv(C_oo)  # (265, 265)


def fosm_variance_single_pred(pred_col, cal_cols_list, O_ens, C_oo_matrix, C_oo_inv_matrix):
    """
    Compute FOSM posterior variance for a single prediction target.
    
    pred_col: column name of the prediction target
    Uses the ensemble covariance approach.
    """
    # Prediction ensemble
    s = prior100[pred_col].values  # (100,)
    s_bar = s.mean()
    A_s = s - s_bar  # (100,)
    
    # Prior variance
    sig2_prior = np.var(s, ddof=1)
    
    # Cross-covariance C_sp = A_s^T A_o / (N-1)  shape: (265,) i.e. (1, 265) flattened
    C_sp = A_s @ A_o / (N - 1)  # (265,) — cross-cov between pred and obs
    
    # Posterior variance: sig2_post = sig2_prior - C_sp @ C_oo^{-1} @ C_sp
    reduction = C_sp @ C_oo_inv_matrix @ C_sp
    sig2_post = sig2_prior - reduction
    
    return sig2_prior, max(sig2_post, 0)


def fosm_variance_omit_one(pred_col, omit_idx):
    """
    Compute FOSM posterior variance when observation at index omit_idx 
    is removed from the calibration dataset.
    """
    # Build reduced matrices (remove column omit_idx)
    mask = np.ones(len(cal_all), dtype=bool)
    mask[omit_idx] = False
    
    A_o_red = A_o[:, mask]  # (100, 264)
    C_d_red = C_d_diag[mask]
    
    C_oo_red = A_o_red.T @ A_o_red / (N - 1) + np.diag(C_d_red)  # (264, 264)
    C_oo_inv_red = np.linalg.inv(C_oo_red)
    
    s = prior100[pred_col].values
    s_bar = s.mean()
    A_s = s - s_bar
    
    sig2_prior = np.var(s, ddof=1)
    C_sp_red = A_s @ A_o_red / (N - 1)  # (264,)
    
    reduction = C_sp_red @ C_oo_inv_red @ C_sp_red
    sig2_post = sig2_prior - reduction
    
    return max(sig2_post, 0)


def fosm_variance_omit_group(pred_col, omit_indices):
    """
    Compute FOSM posterior variance when a GROUP of observations
    (identified by their indices) are removed from the calibration dataset.
    """
    mask = np.ones(len(cal_all), dtype=bool)
    mask[omit_indices] = False
    
    A_o_red = A_o[:, mask]
    C_d_red = C_d_diag[mask]
    
    C_oo_red = A_o_red.T @ A_o_red / (N - 1) + np.diag(C_d_red)
    C_oo_inv_red = np.linalg.inv(C_oo_red)
    
    s = prior100[pred_col].values
    s_bar = s.mean()
    A_s = s - s_bar
    
    sig2_prior = np.var(s, ddof=1)
    C_sp_red = A_s @ A_o_red / (N - 1)
    
    reduction = C_sp_red @ C_oo_inv_red @ C_sp_red
    sig2_post = sig2_prior - reduction
    
    return max(sig2_post, 0)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Per-well prediction targets (like paper Table 5 for each well)
# For each of the 85 wells, treat that well as the prediction target,
# then compute data worth of every other observation.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FOSM DATA-WORTH ANALYSIS — SINGLE-CELL PREDICTION TARGETS")
print("=" * 70)

# Flow obs indices
flow_indices = [i for i, c in enumerate(cal_all) if obs_data.loc[c, "group"] == "flow_cal"]
# Head obs indices  
head_indices = [i for i, c in enumerate(cal_all) if obs_data.loc[c, "group"] == "head_cal"]

# For each well as prediction target, compute:
#   (a) Full posterior variance (all 265 obs)
#   (b) Posterior variance when ALL streamflow obs are omitted (85 head-only)
#   (c) Posterior variance when each individual well is omitted one at a time
#   (d) Variance increase % for streamflow group
#   (e) Variance increase % for each individual head obs
results_all = []

for w_idx, pred_well in enumerate(cal_head):
    r, c = well_rc[w_idx]
    obs_val = obs_data.loc[pred_well, "value"]
    wname = pred_well.replace("head_", "").replace("_sim", "")
    
    # (a) Full posterior variance
    sig2_prior, sig2_full = fosm_variance_single_pred(
        pred_well, cal_all, O, C_oo, C_oo_inv
    )
    
    # (b) Omit ALL streamflow observations
    sig2_noflow = fosm_variance_omit_group(pred_well, flow_indices)
    
    # Variance increase when streamflow is omitted
    if sig2_full > 0:
        flow_worth = (sig2_noflow - sig2_full) / sig2_full * 100
    else:
        flow_worth = 0.0
    
    # (c) Omit each individual head observation one at a time
    well_worths = []
    for h_idx_pos, h_idx in enumerate(head_indices):
        sig2_omit = fosm_variance_omit_one(pred_well, h_idx)
        if sig2_full > 0:
            worth = (sig2_omit - sig2_full) / sig2_full * 100
        else:
            worth = 0.0
        well_worths.append({
            "pred_well": wname,
            "pred_row": r + 1,
            "pred_col": c + 1,
            "pred_obs": obs_val,
            "omitted": cal_head[h_idx_pos].replace("head_", "").replace("_sim", ""),
            "omit_row": well_rc[h_idx_pos][0] + 1,
            "omit_col": well_rc[h_idx_pos][1] + 1,
            "omit_obs": obs_data.loc[cal_head[h_idx_pos], "value"],
            "var_inc_pct": max(worth, 0),
        })
    
    results_all.append({
        "pred_well": wname,
        "pred_row": r + 1,
        "pred_col": c + 1,
        "pred_obs": obs_val,
        "sig2_prior": sig2_prior,
        "sig2_full": sig2_full,
        "sig2_noflow": sig2_noflow,
        "flow_worth_pct": max(flow_worth, 0),
        "well_worths": well_worths,
    })
    
    if (w_idx + 1) % 10 == 0 or w_idx == 0 or w_idx == len(cal_head) - 1:
        print(f"  [{w_idx+1:3d}/{len(cal_head)}] {wname}: "
              f"prior={sig2_prior:.4f}, post_full={sig2_full:.6f}, "
              f"post_noflow={sig2_noflow:.6f}, flow_worth={flow_worth:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY: Mean data worth across all prediction targets
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  MEAN DATA WORTH ACROSS ALL 85 PREDICTION TARGETS")
print("=" * 70)

mean_flow_worth = np.mean([r["flow_worth_pct"] for r in results_all])
print(f"  Streamflow worth (mean across 85 pred targets): {mean_flow_worth:.2f}%")

# Aggregate per-omitted-well worth across all prediction targets
well_names = [cal_head[i].replace("head_", "").replace("_sim", "") for i in range(len(cal_head))]
mean_well_worth = np.zeros(len(cal_head))
for r in results_all:
    for ww in r["well_worths"]:
        idx = well_names.index(ww["omitted"])
        mean_well_worth[idx] += ww["var_inc_pct"]
mean_well_worth /= len(results_all)

# Sort by decreasing worth
order = np.argsort(-mean_well_worth)
print(f"\n  Top-10 wells by mean data worth (across all 85 predictions):")
for rank, idx in enumerate(order[:10]):
    r, c = well_rc[idx]
    obs_val = obs_data.loc[cal_head[idx], "value"]
    print(f"    {rank+1:2d}. {well_names[idx]:<6s} (row {r+1:3d}, col {c+1:3d}, obs={obs_val:.1f} m): "
          f"{mean_well_worth[idx]:.2f}%")

n_zero = (mean_well_worth < 0.005).sum()
print(f"\n  Wells with ~zero mean worth: {n_zero} of {len(cal_head)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER-STYLE TABLE 5: Pick a representative prediction cell
# Use the well closest to center of the model domain (like paper's cell 14517)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  TABLE 5 — SINGLE PREDICTION TARGET (representative well)")
print("=" * 70)

# Pick w040 (row 127, col 41) — a central well, or the well with max flow worth
# Let's try a few representative wells
for rep_name in ["w040", "w027", "w001"]:
    rep_idx = well_names.index(rep_name)
    rep = results_all[rep_idx]
    print(f"\n  Prediction target: {rep_name} "
          f"(row {rep['pred_row']}, col {rep['pred_col']}, obs={rep['pred_obs']:.1f} m)")
    print(f"    Prior variance:         {rep['sig2_prior']:.6f}")
    print(f"    Full posterior variance: {rep['sig2_full']:.6f}")
    print(f"    Streamflow worth:       {rep['flow_worth_pct']:.2f}%")
    
    # Sort well worths
    sorted_ww = sorted(rep["well_worths"], key=lambda x: -x["var_inc_pct"])
    print(f"    Top-5 wells:")
    for ww in sorted_ww[:5]:
        print(f"      {ww['omitted']:<6s} (row {ww['omit_row']:3d}, col {ww['omit_col']:3d}, "
              f"obs={ww['omit_obs']:.1f} m): {ww['var_inc_pct']:.2f}%")
    n_z = sum(1 for ww in sorted_ww if ww["var_inc_pct"] < 0.005)
    print(f"    Wells with ~zero worth: {n_z} of {len(sorted_ww)}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE TABLE 5 CSV — using MEAN across all 85 prediction targets
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SAVING TABLE 5 CSV (mean data worth across all prediction targets)")
print("=" * 70)

rows_out = []

# Streamflow row
rows_out.append({
    "Omitted observation": f"Streamflow observations ({len(cal_flow)} monthly)",
    "Variance increase (%)": round(mean_flow_worth, 2),
})

# Well rows sorted by decreasing worth
for idx in order:
    r, c = well_rc[idx]
    obs_val = obs_data.loc[cal_head[idx], "value"]
    rows_out.append({
        "Omitted observation": f"Well {well_names[idx]}  (row {r+1}, col {c+1}, obs={obs_val:.1f} m)",
        "Variance increase (%)": round(mean_well_worth[idx], 3),
    })

df_out = pd.DataFrame(rows_out)
out_csv = OUT_DIR / "table05_data_worth.csv"
df_out.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")
print(f"  Rows: {len(df_out)} (1 streamflow + {len(cal_head)} wells)")

# Also save per-prediction-target details
detail_rows = []
for r in results_all:
    # Streamflow
    detail_rows.append({
        "prediction_target": r["pred_well"],
        "omitted_obs": "streamflow",
        "var_inc_pct": round(r["flow_worth_pct"], 3),
    })
    for ww in r["well_worths"]:
        detail_rows.append({
            "prediction_target": ww["pred_well"],
            "omitted_obs": ww["omitted"],
            "var_inc_pct": round(ww["var_inc_pct"], 3),
        })

df_detail = pd.DataFrame(detail_rows)
out_detail = OUT_DIR / "table05_data_worth_detail.csv"
df_detail.to_csv(out_detail, index=False)
print(f"  Saved: {out_detail}")
print(f"  Rows: {len(df_detail)}")

# Print final summary table
print("\n" + "=" * 70)
print("  FINAL TABLE 5")
print("=" * 70)
print(df_out.to_string(index=False))
print("\nDone.")
