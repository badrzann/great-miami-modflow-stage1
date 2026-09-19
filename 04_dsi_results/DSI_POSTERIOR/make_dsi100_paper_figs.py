"""
make_dsi100_paper_figs.py
-------------------------
Reproduces Fig. 9 and Fig. 10 from the paper using the DSI-100 posterior
ensemble. Visual style is kept IDENTICAL to the published figures:
   • Blue  = Prior ensemble
   • Orange = Posterior ensemble (DSI-100)
   • Red line = Observed

Numerical fix: uses the Woodbury identity (100×100 inversion) instead of
directly inverting the 265×265 C_oo matrix, which was numerically unstable
for the validation-period flow outputs.

Outputs:
   finalresult/figs/fig09_dsi100_flow.png
   finalresult/figs/fig10_dsi100_head.png
"""

import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)
os.makedirs("finalresult/figs", exist_ok=True)

# ── palette matching the original paper ───────────────────────────────────────
C_PRIOR  = "#72B7E8"   # steel blue  (prior)
C_POST   = "#F4A300"   # amber       (posterior)
C_OBS    = "#C00000"   # dark red    (observed line)
ALPHA_P  = 0.55        # prior alpha
ALPHA_Q  = 0.70        # posterior alpha

# ═══════════════════════════════════════════════════════════════════════════════
#  1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
prior100 = pd.read_csv(
    "finalresult/swatmf_run/swat_modflow_ies_prior100.0.obs.csv", index_col=0
)
obs_data = pd.read_csv(
    "finalresult/swatmf_run/swat_modflow_ies_prior100.adjusted.obs_data.csv"
)
obs_data.set_index("name", inplace=True)

all_cols = prior100.columns.tolist()
col_idx  = {c: i for i, c in enumerate(all_cols)}

# calibration obs (weight > 0)  ← used to condition the update
cal_cols = [c for c in all_cols if obs_data.loc[c, "weight"] > 0]
h_obs    = obs_data.loc[cal_cols, "value"].values   # (265,)
N        = len(prior100)                             # 100

O = prior100[cal_cols].values   # (100, 265)  simulated cal-obs
S = prior100[all_cols].values   # (100, 337)  all predictions (cal + val)

# ═══════════════════════════════════════════════════════════════════════════════
#  2.  MEASUREMENT NOISE
# ═══════════════════════════════════════════════════════════════════════════════
cal_groups = obs_data.loc[cal_cols, "group"].values
sig_d      = np.where(cal_groups == "head_cal", 0.1, 0.1 * np.abs(h_obs))
sig_d      = np.maximum(sig_d, 0.01)   # minimum noise floor

# ═══════════════════════════════════════════════════════════════════════════════
#  3.  ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════
o_bar = O.mean(axis=0)          # (265,)
s_bar = S.mean(axis=0)          # (337,)
A_o   = O - o_bar               # (100, 265)
A_s   = S - s_bar               # (100, 337)

# scale observation anomaly by 1/sig_d (for SVD computation)
A_os  = A_o / sig_d[np.newaxis, :]   # (100, 265)

# ═══════════════════════════════════════════════════════════════════════════════
#  4.  TRUNCATED SVD UPDATE  (matches PESTPP-DSI's n_singular_vals truncation)
#
#  Eigenvalue spectrum of M = I + A_os A_os.T/(N-1) has only ~5 modes with
#  significant signal (eigenvalues > 1e4).  The remaining 95 modes are near-
#  unity noise that produces spurious posterior updates when inverted.
#  Keeping k modes is equivalent to PESTPP-DSI's truncated SVD regularization.
#
#  Reduced-rank Woodbury for C_oo_k = Vt_k.T diag(sv_k^2/(N-1)) Vt_k + diag(C_d):
#    C_oo_k^{-1} d = d/C_d - Vt_k.T solve(CORE, Vt_k (d/C_d))
#    CORE (k×k) = diag((N-1)/sv_k^2) + Vt_k diag(1/C_d) Vt_k.T
# ═══════════════════════════════════════════════════════════════════════════════
# SVD of scaled observation anomaly matrix
U_svd, sv_all, Vt_all = np.linalg.svd(A_os, full_matrices=False)  # A_os (100,265)

# choose k: keep modes where eigenvalue of M > 1% of max
# eigenvalue of M = 1 + sv^2/(N-1);  threshold on sv^2/(N-1) > 0.01*(max sv^2/(N-1))
ev_all  = sv_all**2 / (N - 1)
ev_max  = ev_all[0]
k       = int((ev_all > 0.01 * ev_max).sum())
k       = max(k, 5)   # keep at least 5 modes
k       = min(k, N - 1)
print(f"Retaining k={k} SVD modes  (eigenvalue threshold: {0.01*ev_max:.1f})")

sv_k  = sv_all[:k]    # (k,)
Vt_k  = Vt_all[:k]   # (k, 265)

# Build k×k core matrix  (positive definite, safe to invert)
Vt_k_Cd_inv = Vt_k / (sig_d**2)[np.newaxis, :]         # (k,265) = Vt_k @ diag(1/C_d)
CORE  = np.diag((N - 1) / sv_k**2) + Vt_k_Cd_inv @ Vt_k.T   # (k,k)
CORE_inv = np.linalg.inv(CORE)
print(f"CORE condition number: {np.linalg.cond(CORE):.2e}")

def coo_inv_times(D):
    """Truncated-SVD reduced-rank solve for C_oo_k^{-1} D  (stable, k×k).
    D : (265,) or (265, n_real)"""
    Dd = D / (sig_d**2)[:, None] if D.ndim == 2 else D / sig_d**2  # d/C_d
    Wd = Vt_k @ Dd                                   # (k,) or (k, n_real)
    return Dd - Vt_k.T @ (CORE_inv @ Wd)             # (265,) or (265, n_real)

# ═══════════════════════════════════════════════════════════════════════════════
#  5.  POSTERIOR ENSEMBLE  (perturbed observations update)
#  s'_i = s_i + C_so_k C_oo_k^{-1} (h + eps_i - o_i)
# ═══════════════════════════════════════════════════════════════════════════════
rng  = np.random.default_rng(42)
eps  = rng.standard_normal((N, len(cal_cols))) * sig_d[np.newaxis, :]   # (100,265)
D_ens = h_obs[np.newaxis, :] + eps - O                                   # (100,265)

# C_oo_k^{-1} applied to each column of D_ens.T  →  (265, 100)
ALPHA  = coo_inv_times(D_ens.T)

# truncated cross-covariance  C_so_k = A_s.T @ (A_o projected) / (N-1)
# A_o_k = U_k @ diag(sv_k) @ Vt_k  →  A_s.T @ A_o_k = A_s.T @ U_k[:,k] @ diag(sv_k) @ Vt_k
A_o_k  = (U_svd[:, :k] * sv_k[np.newaxis, :]) @ Vt_k    # (100, 265)
C_so_k = A_s.T @ A_o_k / (N - 1)                         # (337, 265)

delta  = C_so_k @ ALPHA       # (337, 100)
S_post = S.T + delta          # (337, 100)  posterior ensemble

# physical floor for flow outputs (clip negative predictions)
for ci, cname in enumerate(all_cols):
    if "flow" in cname:
        S_post[ci] = np.clip(S_post[ci], 0, None)

# diagnostic check
for test in ["flow_201803_sim", "flow_201904_sim", "flow_202101_sim", "flow_202312_sim"]:
    ci    = col_idx[test]
    p5, p95 = np.percentile(S_post[ci], [5, 95])
    obs_v = obs_data.loc[test, "value"]
    print(f"  {test}: obs={obs_v:.1f}  post_CI=[{p5:.1f}, {p95:.1f}]")

# ── helper to get prior / posterior arrays and observed value ──────────────────
obs_map = obs_data["value"].to_dict()

def get_arrays(colname):
    ci = col_idx[colname]
    return S[: , ci], S_post[ci], obs_map[colname]   # prior=row of S (100,)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG. 9 — STREAMFLOW HISTOGRAMS  (same 9 validation months as paper)
# ═══════════════════════════════════════════════════════════════════════════════
flow_panels = [
    ("Mar-2018", "flow_201803_sim"),
    ("Sep-2018", "flow_201809_sim"),
    ("Apr-2019", "flow_201904_sim"),
    ("Dec-2019", "flow_201912_sim"),
    ("May-2020", "flow_202005_sim"),
    ("Jan-2021", "flow_202101_sim"),
    ("Sep-2021", "flow_202109_sim"),
    ("Apr-2022", "flow_202204_sim"),
    ("Dec-2023", "flow_202312_sim"),
]

fig9, axes9 = plt.subplots(3, 3, figsize=(13, 10),
                            gridspec_kw={"hspace": 0.45, "wspace": 0.35})
for ax, (title, col) in zip(axes9.flatten(), flow_panels):
    prior_arr, post_arr, obs_val = get_arrays(col)

    # bin edges: cover both prior and posterior visible range
    lo = min(np.percentile(prior_arr, 1), np.percentile(post_arr, 1), obs_val * 0.85)
    hi = max(np.percentile(prior_arr, 99), np.percentile(post_arr, 99), obs_val * 1.15)
    lo = max(lo, 0)
    bins = np.linspace(lo, hi, 10)

    ax.hist(prior_arr, bins=bins, alpha=ALPHA_P, color=C_PRIOR,
            edgecolor="white", linewidth=0.5, label="Prior ensemble")
    ax.hist(post_arr,  bins=bins, alpha=ALPHA_Q, color=C_POST,
            edgecolor="white", linewidth=0.5, label="Posterior ensemble")
    ax.axvline(obs_val, color=C_OBS, linewidth=1.8, label="Observed")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Flow (m³/s)", fontsize=8)
    ax.set_ylabel("Number of realizations", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", linewidth=0.4, alpha=0.45, color="grey")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

prior_patch = mpatches.Patch(facecolor=C_PRIOR, alpha=0.65, label="Prior ensemble")
post_patch  = mpatches.Patch(facecolor=C_POST,  alpha=0.75, label="Posterior ensemble")
obs_line    = plt.Line2D([0], [0], color=C_OBS, linewidth=1.8, label="Observed")
fig9.legend(handles=[prior_patch, post_patch, obs_line],
            loc="lower center", ncol=3, fontsize=9.5,
            frameon=True, bbox_to_anchor=(0.5, 0.0),
            framealpha=0.9, edgecolor="lightgrey")

fig9.suptitle(
    "Fig. 9.  Prior and posterior distributions of the DSI-100 statistical surrogate\n"
    "model predictions of streamflow (m³/s) for specific times during the prediction\n"
    "period (2018–2023).",
    fontsize=9.5, x=0.5, y=1.01, ha="center"
)
fig9.tight_layout(rect=[0, 0.06, 1, 1])

out9 = "finalresult/figs/fig09_dsi100_flow.png"
fig9.savefig(out9, dpi=200, bbox_inches="tight")
plt.close(fig9)
print(f"\nSaved: {out9}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG. 10 — GROUNDWATER HEAD HISTOGRAMS  (same 4 wells as paper)
# ═══════════════════════════════════════════════════════════════════════════════
head_panels = [
    ("well 75  (row 76, col 69)",  "head_w075_sim"),
    ("well 67  (row 96, col 60)",  "head_w067_sim"),
    ("well 10  (row 174, col 29)", "head_w010_sim"),
    ("well 4   (row 52, col 73)",  "head_w004_sim"),
]

fig10, axes10 = plt.subplots(2, 2, figsize=(11, 8),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})
for ax, (title, col) in zip(axes10.flatten(), head_panels):
    prior_arr, post_arr, obs_val = get_arrays(col)

    lo = min(np.percentile(prior_arr, 0.5), np.percentile(post_arr, 0.5), obs_val - 1)
    hi = max(np.percentile(prior_arr, 99.5), np.percentile(post_arr, 99.5), obs_val + 1)
    bins = np.linspace(lo, hi, 12)

    ax.hist(prior_arr, bins=bins, alpha=ALPHA_P, color=C_PRIOR,
            edgecolor="white", linewidth=0.5, label="Prior ensemble")
    ax.hist(post_arr,  bins=bins, alpha=ALPHA_Q, color=C_POST,
            edgecolor="white", linewidth=0.5, label="Posterior ensemble")
    ax.axvline(obs_val, color=C_OBS, linewidth=1.8, label="Observed head")

    # annotation: 90% CI widths
    prior_w = np.percentile(prior_arr, 95) - np.percentile(prior_arr, 5)
    post_w  = np.percentile(post_arr,  95) - np.percentile(post_arr,  5)
    ax.annotate(
        f"Prior 90%CI: {prior_w:.1f} m\nPost 90%CI:  {post_w:.3f} m",
        xy=(0.97, 0.97), xycoords="axes fraction",
        va="top", ha="right", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="lightgrey"),
    )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("gw head (m)", fontsize=8)
    ax.set_ylabel("Number of realisations", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", linewidth=0.4, alpha=0.45, color="grey")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

prior_patch = mpatches.Patch(facecolor=C_PRIOR, alpha=0.65, label="Prior ensemble")
post_patch  = mpatches.Patch(facecolor=C_POST,  alpha=0.75, label="Posterior ensemble")
obs_line    = plt.Line2D([0], [0], color=C_OBS, linewidth=1.8, label="Observed head")
fig10.legend(handles=[prior_patch, post_patch, obs_line],
             loc="lower center", ncol=3, fontsize=9.5,
             frameon=True, bbox_to_anchor=(0.5, 0.0),
             framealpha=0.9, edgecolor="lightgrey")

fig10.suptitle(
    "Fig. 10.  Prior and posterior distributions of the DSI-100 statistical surrogate\n"
    "model predictions of groundwater head (m) for specific times during the\n"
    "prediction period (2018–2023).",
    fontsize=9.5, x=0.5, y=1.01, ha="center"
)
fig10.tight_layout(rect=[0, 0.06, 1, 1])

out10 = "finalresult/figs/fig10_dsi100_head.png"
fig10.savefig(out10, dpi=200, bbox_inches="tight")
plt.close(fig10)
print(f"Saved: {out10}")

print("\nDone.  Both figures saved to finalresult/figs/")
