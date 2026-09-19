"""
make_data_worth_figs.py
=======================
Replicates the data-worth analysis from Qasemipour et al. (2026) Figs. 12 & 13
for the GMRW watershed using an analytical Ensemble Smoother (ES) update.

Methodology (same as the paper):
  - Run ES update TWICE on the 100-member prior ensemble:
      Case A (full):      condition on flow_cal + head_cal  (265 obs)
      Case B (flow-only): condition on flow_cal only          (180 obs)
  - Uncertainty increase = Q95−Q5 of Case B  −  Q95−Q5 of Case A
    at the 85 GW head observation locations.

Fig. 12 equivalent:
  Spatial map of the uncertainty increase per well location, IDW-interpolated
  to the full MODFLOW grid (197 × 135), masked to the active watershed.

Fig. 13 equivalent:
  Head-distribution histograms for 4 key wells (w044, w057, w065, w079)
  showing:  Prior  |  Posterior (full)  |  Posterior (flow-only)
  — so the viewer can directly see how much head observations contribute.

Outputs
-------
  finalresult/figs/fig12_dataworth_spatial.png
  finalresult/figs/fig13_dataworth_hists.png
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

np.random.seed(42)

os.makedirs("finalresult/figs", exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
NROW, NCOL = 197, 135
BASE       = "finalresult/swatmf_run"

# ── colours (match fig06-10 palette) ──────────────────────────────────────────
C_PRIOR   = "#72B7E8"   # steel blue
C_FULL    = "#F4A300"   # amber  (full posterior)
C_FONLY   = "#E04040"   # red-orange (flow-only posterior)
C_OBS     = "#C00000"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading prior ensemble and observation data …")
prior100 = pd.read_csv(
    f"{BASE}/swat_modflow_ies_prior100.0.obs.csv", index_col=0
)
obs_data = pd.read_csv(
    f"{BASE}/swat_modflow_ies_prior100.adjusted.obs_data.csv"
)
obs_data.set_index("name", inplace=True)

all_cols  = prior100.columns.tolist()
S_prior   = prior100[all_cols].values          # (100, 337) – full prior ensemble

# calibration obs
cal_all   = [c for c in all_cols if obs_data.loc[c, "weight"] > 0]
cal_flow  = [c for c in cal_all  if obs_data.loc[c, "group"] == "flow_cal"]
cal_head  = [c for c in cal_all  if obs_data.loc[c, "group"] == "head_cal"]

print(f"  flow_cal: {len(cal_flow)},  head_cal: {len(cal_head)},  total: {len(cal_all)}")

# head column indices in all_cols (for later use)
head_idx = [all_cols.index(c) for c in cal_head]  # 85 indices

# observed values
h_full  = obs_data.loc[cal_all,  "value"].values   # (265,)
h_flow  = obs_data.loc[cal_flow, "value"].values   # (180,)

N = prior100.shape[0]   # 100


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ANALYTICAL ES UPDATE  (Woodbury-form, same as dsi_100_rebuild.py)
# ══════════════════════════════════════════════════════════════════════════════
def es_update(S, cal_cols_list, h_obs, obs_data_df, n_real=100):
    """
    Ensemble Smoother update.

    Parameters
    ----------
    S            : (N, n_pred) prior ensemble   (all columns)
    cal_cols_list: list of calibration column names
    h_obs        : (n_cal,) observed values
    obs_data_df  : DataFrame with 'group' per obs name
    n_real       : number of posterior realizations to return

    Returns
    -------
    S_post : (n_pred, n_real) posterior ensemble
    """
    O = prior100[cal_cols_list].values    # (N, n_cal) simulated cal-obs

    # Measurement noise
    groups   = obs_data_df.loc[cal_cols_list, "group"].values
    sig_d    = np.where(groups == "head_cal", 0.1, 0.1 * np.abs(h_obs))
    sig_d    = np.maximum(sig_d, 0.01)
    C_d_diag = sig_d ** 2

    # Anomalies
    o_bar = O.mean(axis=0)          # (n_cal,)
    s_bar = S.mean(axis=0)          # (n_pred,)
    A_o   = O - o_bar               # (N, n_cal)
    A_s   = S - s_bar               # (N, n_pred)

    # Covariance matrices
    C_oo = A_o.T @ A_o / (N - 1) + np.diag(C_d_diag)   # (n_cal, n_cal)
    C_so = A_s.T @ A_o / (N - 1)                         # (n_pred, n_cal)

    # Perturbed-observations posterior ensemble
    eps  = np.random.randn(n_real, len(cal_cols_list)) * sig_d[np.newaxis, :]
    D    = h_obs[np.newaxis, :] + eps - O[:n_real]       # (n_real, n_cal)
    ALPHA = np.linalg.solve(C_oo, D.T)                    # (n_cal, n_real)
    delta = C_so @ ALPHA                                   # (n_pred, n_real)

    S_post = S[:n_real].T + delta    # (n_pred, n_real)
    return S_post


print("\nRunning ES update — Case A (flow + head) …")
S_full  = es_update(S_prior, cal_all,  h_full,  obs_data)   # (337, 100)

print("Running ES update — Case B (flow only) …")
S_fonly = es_update(S_prior, cal_flow, h_flow,  obs_data)   # (337, 100)

print("ES updates complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  UNCERTAINTY INCREASE AT 85 HEAD LOCATIONS
# ══════════════════════════════════════════════════════════════════════════════
# Extract the 85 head columns from each ensemble
H_prior  = S_prior[:, head_idx]   # (100, 85) — transpose → (85, 100)
H_full   = S_full [head_idx, :]   # (85, 100)
H_fonly  = S_fonly[head_idx, :]   # (85, 100)

def q95_q05(arr2d):
    """arr2d: (n_pts, n_real) → spread per point."""
    return np.percentile(arr2d, 95, axis=1) - np.percentile(arr2d, 5, axis=1)

spread_prior = q95_q05(H_prior.T)   # (85,) prior spread
spread_full  = q95_q05(H_full)      # (85,) full posterior spread
spread_fonly = q95_q05(H_fonly)     # (85,) flow-only posterior spread

delta_spread = spread_fonly - spread_full   # > 0 means head obs helped

print(f"\nPrior Q95-Q5 spread  : {spread_prior.mean():.2f} m  (mean over 85 wells)")
print(f"Full-posterior spread: {spread_full.mean():.2f} m")
print(f"Flow-only spread     : {spread_fonly.mean():.2f} m")
print(f"Mean increase (Δ)    : {delta_spread.mean():.2f} m")
print(f"Max increase          : {delta_spread.max():.2f} m at well idx {np.argmax(delta_spread)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LOAD SPATIAL INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
def load_ibound():
    vals = []
    with open(f"{BASE}/modflow_GMRW.bas") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.upper() == "FREE":
                continue
            for tok in s.split():
                try:
                    vals.append(int(float(tok)))
                except ValueError:
                    pass
            if len(vals) >= NROW * NCOL:
                break
    return np.array(vals[:NROW * NCOL], dtype=int).reshape(NROW, NCOL)


def load_river_mask():
    mask = np.zeros((NROW, NCOL), dtype=bool)
    riv_path = f"{BASE}/swatmf_river2grid.txt"
    if not os.path.exists(riv_path):
        return mask
    with open(riv_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 2:
            try:
                cell_num = int(parts[1])
                r0 = (cell_num - 1) // NCOL
                c0 = (cell_num - 1) % NCOL
                if 0 <= r0 < NROW and 0 <= c0 < NCOL:
                    mask[r0, c0] = True
            except (ValueError, IndexError):
                pass
        i += 3
    return mask


def load_wells():
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
    return well_rc   # list of (row0, col0)


ibound      = load_ibound()
river_mask  = load_river_mask()
watershed   = (ibound > 0)
well_rc     = load_wells()    # 85 wells, same order as head_w001..head_w085

wr = np.array([r for r, c in well_rc])   # (85,)
wc = np.array([c for r, c in well_rc])   # (85,)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIG. 12  — Spatial map of uncertainty increase
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig. 12: Spatial uncertainty increase map ──")

# Build grid of column/row indices for interpolation target
rows_flat = np.arange(NROW)
cols_flat = np.arange(NCOL)
cc, rr = np.meshgrid(cols_flat, rows_flat)    # (NROW, NCOL)
pts_all = np.column_stack([cc.ravel(), rr.ravel()])  # (NROW*NCOL, 2)

# Known points: well column, well row, value
pts_known = np.column_stack([wc, wr])          # (85, 2)
vals_known = delta_spread                       # (85,)

# IDW interpolation to full grid
# scipy griddata uses linear/cubic — use linear on the 85 scatter points
# then fill holes with nearest-neighbour
grid_linear  = griddata(pts_known, vals_known, pts_all, method="linear")
grid_nearest = griddata(pts_known, vals_known, pts_all, method="nearest")
grid_idw     = np.where(np.isnan(grid_linear), grid_nearest, grid_linear)
grid_idw     = grid_idw.reshape(NROW, NCOL)
grid_idw     = np.where(watershed, grid_idw, np.nan)   # mask to watershed

vmax12 = np.nanpercentile(grid_idw, 98)   # robust colorbar ceiling

fig12, ax12 = plt.subplots(figsize=(7, 9), dpi=200)
fig12.patch.set_facecolor("white")
ax12.set_facecolor("#e8e8e8")

im12 = ax12.imshow(grid_idw, cmap="YlOrRd", vmin=0, vmax=vmax12,
                   aspect="auto", interpolation="bilinear", origin="upper")

# ── Stream network ───────────────────────────────────────────────────────────
if river_mask.any():
    riv_rgba = np.zeros((NROW, NCOL, 4))
    riv_rgba[river_mask] = [26/255, 95/255, 180/255, 0.85]
    ax12.imshow(riv_rgba, aspect="auto", interpolation="none", origin="upper")

# ── Watershed boundary ───────────────────────────────────────────────────────
ax12.contour(watershed.astype(float), levels=[0.5],
             colors="black", linewidths=0.8)

# ── Wells — white fill with black edge, sized by Δ uncertainty ───────────────
well_norm = (delta_spread - delta_spread.min()) / (delta_spread.max() - delta_spread.min() + 1e-9)
sc_sizes  = 25 + 130 * well_norm
ax12.scatter(wc, wr, s=sc_sizes, c="#2DC653",
             edgecolors="#114B26", linewidths=0.8, zorder=7)

cb12 = fig12.colorbar(im12, ax=ax12, shrink=0.75, pad=0.02)
cb12.set_label("Uncertainty increase  Q95−Q5 (m)\n[flow-only posterior − full posterior]",
               fontsize=9, rotation=270, labelpad=22)
cb12.ax.tick_params(labelsize=8)

ax12.set_title("Spatial Increase in GW Head Uncertainty\n"
               "Caused by Excluding Head Observations (GMRW)",
               fontsize=11, fontweight="bold", pad=6)
ax12.set_xticks([]); ax12.set_yticks([])
for sp in ax12.spines.values():
    sp.set_edgecolor("#555"); sp.set_linewidth(0.8)

legend12 = [
    Line2D([0], [0], color="#1A5FB4", lw=2.0, label="Stream network"),
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="#2DC653", markeredgecolor="#114B26",
           markeredgewidth=0.8, markersize=6, lw=0,
           label="Obs. wells  (size ∝ Δ uncertainty)"),
]
ax12.legend(handles=legend12, loc="upper left", fontsize=8,
            frameon=True, framealpha=0.90)

out12 = "finalresult/figs/fig12_dataworth_spatial.png"
fig12.tight_layout()
fig12.savefig(out12, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig12)
print(f"  Saved: {out12}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIG. 13  — Head histograms (4 key wells): Prior / Full / Flow-only
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig. 13: Head distribution histograms (data worth) ──")

# 4 key wells: head_w044, head_w057, head_w065, head_w079
# head_wNNN → 0-indexed position NNN-1 in head_idx list
KEY_WELLS = [
    ("w044", 43, "Well w044"),
    ("w057", 56, "Well w057"),
    ("w065", 64, "Well w065"),
    ("w079", 78, "Well w079"),
]

fig13, axes13 = plt.subplots(1, 4, figsize=(16, 4.5), dpi=200)
fig13.patch.set_facecolor("white")

BINS = 25
ALPHA_P = 0.55
ALPHA_Q = 0.70

for ax, (wname, widx, title) in zip(axes13.flat, KEY_WELLS):
    # Extract realization vectors for this well (widx = 0-based position in head_idx)
    prior_vals = H_prior.T[widx, :]    # shape (100,)
    full_vals  = H_full[widx, :]       # shape (100,)
    fonly_vals = H_fonly[widx, :]      # shape (100,)

    obs_val    = obs_data.loc[f"head_{wname}_sim", "value"]

    all_vals = np.concatenate([prior_vals, full_vals, fonly_vals])
    xlim = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))

    bin_edges = np.linspace(xlim[0], xlim[1], BINS + 1)

    ax.hist(prior_vals, bins=bin_edges, color=C_PRIOR, alpha=ALPHA_P,
            label="Prior", edgecolor="none", density=False)
    ax.hist(full_vals,  bins=bin_edges, color=C_FULL,  alpha=ALPHA_Q,
            label="Posterior (flow+head)", edgecolor="none", density=False)
    ax.hist(fonly_vals, bins=bin_edges, color=C_FONLY, alpha=0.60,
            label="Posterior (flow only)", edgecolor="none", density=False)

    ax.axvline(obs_val, color=C_OBS, linewidth=1.8, linestyle="-",
               label="Observed", zorder=8)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("GW Head (m)", fontsize=9)
    ax.set_ylabel("Count" if ax is axes13[0] else "", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#888"); sp.set_linewidth(0.7)

# Common legend on right
legend_patches = [
    mpatches.Patch(facecolor=C_PRIOR, alpha=ALPHA_P, label="Prior (100 real.)"),
    mpatches.Patch(facecolor=C_FULL,  alpha=ALPHA_Q, label="Posterior — flow + head"),
    mpatches.Patch(facecolor=C_FONLY, alpha=0.60,    label="Posterior — flow only"),
    Line2D([0], [0], color=C_OBS, lw=1.8, label="Observed"),
]
fig13.legend(handles=legend_patches, loc="upper right", fontsize=8.5,
             frameon=True, framealpha=0.9, ncol=4,
             bbox_to_anchor=(0.99, 1.01))

fig13.suptitle(
    "GW Head Distributions: Worth of Head Observations  (GMRW — DSI Analytical ES)",
    fontsize=11, fontweight="bold", y=1.06
)
fig13.tight_layout()

out13 = "finalresult/figs/fig13_dataworth_hists.png"
fig13.savefig(out13, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig13)
print(f"  Saved: {out13}")

print("\nDone — Fig. 12 and 13 (data worth) generated successfully.")
