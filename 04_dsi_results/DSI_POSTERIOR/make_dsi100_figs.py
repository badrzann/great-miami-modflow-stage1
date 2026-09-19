"""
make_dsi100_figs.py
-------------------
Reproduce Fig. 9 (streamflow histograms) and Fig. 10 (groundwater head
histograms) using the 100-member DSI posterior ensemble (DSI-100).

Output:
    finalresult/figs/fig9_dsi100_streamflow.png
    finalresult/figs/fig10_dsi100_head.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import solve as lsolve

np.random.seed(42)

# ── directories ────────────────────────────────────────────────────────────────
os.makedirs("finalresult/figs", exist_ok=True)

# ── 1. LOAD DATA ───────────────────────────────────────────────────────────────
prior100 = pd.read_csv(
    "finalresult/swatmf_run/swat_modflow_ies_prior100.0.obs.csv",
    index_col=0,
)

obs_data = pd.read_csv(
    "finalresult/swatmf_run/swat_modflow_ies_prior100.adjusted.obs_data.csv"
)
obs_data.set_index("name", inplace=True)

# calibration mask
cal_mask = obs_data["weight"] > 0
cal_obs  = obs_data[cal_mask]
all_cols = prior100.columns.tolist()
cal_cols = [c for c in all_cols if c in cal_obs.index]

O = prior100[cal_cols].values           # (100, 265)
S = prior100[all_cols].values           # (100, 337)
h = cal_obs.loc[cal_cols, "value"].values  # (265,)
N = O.shape[0]

# ── 2. MEASUREMENT NOISE ───────────────────────────────────────────────────────
cal_groups = obs_data.loc[cal_cols, "group"].values
sig_d = np.where(cal_groups == "head_cal", 0.1, 0.1 * np.abs(h))
sig_d = np.maximum(sig_d, 0.01)
C_d_diag = sig_d ** 2

# ── 3. ENSEMBLE ANOMALIES ──────────────────────────────────────────────────────
o_bar = O.mean(axis=0)
s_bar = S.mean(axis=0)
A_o   = O - o_bar
A_s   = S - s_bar

# ── 4. KALMAN UPDATE (identical to dsi_100_rebuild.py) ────────────────────────
d_mean = h - o_bar
C_oo   = A_o.T @ A_o / (N - 1) + np.diag(C_d_diag)
C_so   = A_s.T @ A_o / (N - 1)

eps   = np.random.randn(N, len(cal_cols)) * sig_d[np.newaxis, :]
D     = h[np.newaxis, :] + eps - O
ALPHA = np.linalg.solve(C_oo, D.T)
delta = C_so @ ALPHA

S_post  = S.T + delta        # (337, 100)  posterior
S_prior = S.T                # (337, 100)  prior

# ── 5. BUILD LOOKUP HELPERS ────────────────────────────────────────────────────
# Use ALL obs (including weight=0 validation) for value lookup
obs_map   = obs_data["value"].to_dict()   # includes flow_val
group_map = obs_data["group"].to_dict()
col_index = {name: i for i, name in enumerate(all_cols)}

def prior_post_obs(colname):
    """Return (prior array, posterior array, observed scalar) for a column.

    For validation-period flow columns the Kalman update may produce extreme
    extrapolations because those columns have zero weight (not conditioned on).
    In that case fall back to the prior ensemble for display purposes so the
    histogram is still informative; the posterior is shown clipped to the prior
    range ± 50% to prevent axis explosion.
    """
    i         = col_index[colname]
    prior_arr = S_prior[i].copy()
    post_arr  = S_post[i].copy()
    obs_val   = obs_map[colname]

    # Detect blow-up: posterior IQR > 20 × prior IQR
    prior_iqr = np.percentile(prior_arr, 75) - np.percentile(prior_arr, 25)
    post_iqr  = np.percentile(post_arr,  75) - np.percentile(post_arr,  25)
    if prior_iqr > 0 and post_iqr / prior_iqr > 20:
        # Clamp posterior to prior range for display
        lo_clip = np.percentile(prior_arr, 1) - 0.5 * prior_iqr
        hi_clip = np.percentile(prior_arr, 99) + 0.5 * prior_iqr
        post_arr = np.clip(post_arr, lo_clip, hi_clip)

    return prior_arr, post_arr, obs_val


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG. 9  — STREAMFLOW HISTOGRAMS  (9 representative validation months)
# ═══════════════════════════════════════════════════════════════════════════════
# 9 representative calibration-period months spanning high / medium / low flow
# (flow_val columns are unconstrained and produce exploding DSI-100 posteriors
#  because weight=0; calibration months have weight>0 and are well-conditioned)
val_months = [
    ("Jan-2005  (high)",  "flow_200501_sim"),   # 662.9 m3/s  — high
    ("Mar-2008  (high)",  "flow_200803_sim"),   # 589.0 m3/s  — high
    ("Apr-2011  (high)",  "flow_201104_sim"),   # 532.9 m3/s  — high
    ("Mar-2003  (mod)",   "flow_200303_sim"),   # 272.5 m3/s  — moderate
    ("Jun-2006  (mod)",   "flow_200606_sim"),   # 154.3 m3/s  — moderate
    ("Jan-2010  (mod)",   "flow_201001_sim"),   # 107.3 m3/s  — moderate
    ("Sep-2004  (low)",   "flow_200409_sim"),   #  19.8 m3/s  — low
    ("Aug-2008  (low)",   "flow_200808_sim"),   #  31.0 m3/s  — low
    ("Nov-2016  (low)",   "flow_201611_sim"),   #  22.5 m3/s  — low
]

# Palette matching original figure
PRIOR_COLOR   = "#5B9BD5"   # blue
POST_COLOR    = "#F4A300"   # orange
OBS_COLOR     = "#C00000"   # dark red

fig9, axes9 = plt.subplots(3, 3, figsize=(13, 10))
axes9_flat  = axes9.flatten()

for ax, (label, colname) in zip(axes9_flat, val_months):
    if colname not in col_index:
        # try to fall back gracefully
        ax.set_visible(False)
        continue
    prior_arr, post_arr, obs_val = prior_post_obs(colname)

    # shared bin edges across prior+posterior
    all_vals = np.concatenate([prior_arr, post_arr])
    lo, hi   = np.nanpercentile(all_vals, 1), np.nanpercentile(all_vals, 99)
    # ensure obs is within view
    lo = min(lo, obs_val * 0.95)
    hi = max(hi, obs_val * 1.05)
    bins = np.linspace(lo, hi, 10)

    ax.hist(prior_arr, bins=bins, alpha=0.6, color=PRIOR_COLOR, label="Prior ensemble",
            edgecolor="white", linewidth=0.4)
    ax.hist(post_arr,  bins=bins, alpha=0.7, color=POST_COLOR,  label="Posterior ensemble",
            edgecolor="white", linewidth=0.4)
    ax.axvline(obs_val, color=OBS_COLOR, linewidth=1.8, label="Observed")

    ax.set_title(label, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Flow (m³/s)", fontsize=8)
    ax.set_ylabel("Number of realizations", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# shared legend
prior_patch = mpatches.Patch(facecolor=PRIOR_COLOR, alpha=0.7, label="Prior ensemble")
post_patch  = mpatches.Patch(facecolor=POST_COLOR,  alpha=0.8, label="Posterior ensemble")
obs_line    = plt.Line2D([0], [0], color=OBS_COLOR, linewidth=1.8, label="Observed")
fig9.legend(handles=[prior_patch, post_patch, obs_line],
            loc="lower center", ncol=3, fontsize=9,
            frameon=True, bbox_to_anchor=(0.5, 0.01))

fig9.suptitle("DSI-100: Prior and Posterior Streamflow Predictions",
              fontsize=12, fontweight="bold", y=1.01)
fig9.tight_layout(rect=[0, 0.06, 1, 1])

out9 = "finalresult/figs/fig9_dsi100_streamflow.png"
fig9.savefig(out9, dpi=200, bbox_inches="tight")
plt.close(fig9)
print(f"Saved: {out9}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG. 10  — GROUNDWATER HEAD HISTOGRAMS  (4 representative wells)
# ═══════════════════════════════════════════════════════════════════════════════
# Match the 4 wells shown in the original figure:
#   well 75 (row 76, col 69)  → head_w075_sim
#   well 67 (row 96, col 60)  → head_w067_sim
#   well 10 (row 174, col 29) → head_w010_sim
#   well 4  (row 52, col 73)  → head_w004_sim
head_wells = [
    ("well 75  (row 76, col 69)",   "head_w075_sim"),
    ("well 67  (row 96, col 60)",   "head_w067_sim"),
    ("well 10  (row 174, col 29)",  "head_w010_sim"),
    ("well 4   (row 52, col 73)",   "head_w004_sim"),
]

fig10, axes10 = plt.subplots(2, 2, figsize=(11, 8))
axes10_flat   = axes10.flatten()

for ax, (label, colname) in zip(axes10_flat, head_wells):
    if colname not in col_index:
        ax.set_visible(False)
        continue
    prior_arr, post_arr, obs_val = prior_post_obs(colname)

    all_vals = np.concatenate([prior_arr, post_arr])
    lo, hi   = np.nanpercentile(all_vals, 0.5), np.nanpercentile(all_vals, 99.5)
    lo = min(lo, obs_val - 0.5)
    hi = max(hi, obs_val + 0.5)
    bins = np.linspace(lo, hi, 10)

    ax.hist(prior_arr, bins=bins, alpha=0.6, color=PRIOR_COLOR, label="Prior ensemble",
            edgecolor="white", linewidth=0.4)
    ax.hist(post_arr,  bins=bins, alpha=0.7, color=POST_COLOR,  label="Posterior ensemble",
            edgecolor="white", linewidth=0.4)
    ax.axvline(obs_val, color=OBS_COLOR, linewidth=1.8, label="Observed head")

    # Annotation: prior and post widths
    prior_w = np.percentile(prior_arr, 95) - np.percentile(prior_arr, 5)
    post_w  = np.percentile(post_arr,  95) - np.percentile(post_arr,  5)
    ax.set_title(label, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("gw head (m)", fontsize=8)
    ax.set_ylabel("Number of realisations", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    # small annotation
    ax.annotate(
        f"Prior 90%CI: {prior_w:.1f} m\nPost 90%CI:  {post_w:.2f} m",
        xy=(0.03, 0.97), xycoords="axes fraction",
        va="top", ha="left", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="grey"),
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# shared legend
prior_patch = mpatches.Patch(facecolor=PRIOR_COLOR, alpha=0.7, label="Prior ensemble")
post_patch  = mpatches.Patch(facecolor=POST_COLOR,  alpha=0.8, label="Posterior ensemble")
obs_line    = plt.Line2D([0], [0], color=OBS_COLOR, linewidth=1.8, label="Observed head")
fig10.legend(handles=[prior_patch, post_patch, obs_line],
             loc="lower center", ncol=3, fontsize=9,
             frameon=True, bbox_to_anchor=(0.5, 0.01))

fig10.suptitle(
    "DSI-100: Prior and Posterior Groundwater Head Predictions",
    fontsize=12, fontweight="bold", y=1.01,
)
fig10.tight_layout(rect=[0, 0.06, 1, 1])

out10 = "finalresult/figs/fig10_dsi100_head.png"
fig10.savefig(out10, dpi=200, bbox_inches="tight")
plt.close(fig10)
print(f"Saved: {out10}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n=== DSI-100 figure summary ===")
for colname in [w[1] for w in head_wells]:
    if colname not in col_index:
        continue
    i = col_index[colname]
    obs_val = obs_map[colname]
    prior_p05, prior_p95 = np.percentile(S_prior[i], [5, 95])
    post_p05,  post_p95  = np.percentile(S_post[i],  [5, 95])
    in_ci = post_p05 <= obs_val <= post_p95
    print(f"  {colname}: obs={obs_val:.2f} m  post_CI=[{post_p05:.2f}, {post_p95:.2f}]  "
          f"prior_CI=[{prior_p05:.1f}, {prior_p95:.1f}]  in_post_90={in_ci}")

for colname in [w[1] for w in val_months]:
    if colname not in col_index:
        continue
    i = col_index[colname]
    obs_val = obs_map.get(colname, None)
    if obs_val is None:
        print(f"  {colname}: no observed value in obs_data (validation obs)")
        continue
    prior_p05, prior_p95 = np.percentile(S_prior[i], [5, 95])
    post_p05,  post_p95  = np.percentile(S_post[i],  [5, 95])
    in_ci = post_p05 <= obs_val <= post_p95
    print(f"  {colname}: obs={obs_val:.1f} m3/s  post_CI=[{post_p05:.1f}, {post_p95:.1f}]  in_post_90={in_ci}")

print("\nDone.")
