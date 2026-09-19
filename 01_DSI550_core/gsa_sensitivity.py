"""
Prior-Ensemble Sensitivity Screening
=====================================
Uses the 550-member prior ensemble from Stage-3 IES to evaluate
Spearman rank correlation, Pearson correlation, and Standardised
Regression Coefficients (SRC) between the 13 calibration parameters
and model outputs (streamflow metrics, head metrics).

Produces:
  gsa_spearman_panels.png     — multi-panel Spearman heatmap + ranked bars
  gsa_all_metrics_heatmap.png — full Spearman/Pearson/SRC comparison heatmap
  gsa_ranked_table.csv        — ranked sensitivity table (all parameters × outputs)
  gsa_flow_barplot.png        — top parameters for streamflow
  gsa_head_barplot.png        — top parameters for groundwater head
  gsa_combined_barplot.png    — combined streamflow + head ranking

Saved to: D:/nasrin/gsa_<name>.png  (same folder as your existing figure)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Pure-numpy implementations (no scipy required) ─────────────────────────
class stats:
    @staticmethod
    def spearmanr(x, y):
        """Spearman rank correlation and a naive 2-tailed p-value."""
        n = len(x)
        rx = np.argsort(np.argsort(x)).astype(float) + 1
        ry = np.argsort(np.argsort(y)).astype(float) + 1
        rho = np.corrcoef(rx, ry)[0, 1]
        # t-statistic for Spearman (large-n approx)
        t  = rho * np.sqrt((n - 2) / max(1 - rho**2, 1e-20))
        from math import erfc, sqrt
        pval = erfc(abs(t) / sqrt(2))          # normal approximation (good for n≥30)
        return rho, pval

    @staticmethod
    def pearsonr(x, y):
        r = np.corrcoef(x, y)[0, 1]
        return r, np.nan

# ── Paths ──────────────────────────────────────────────────────────────────
RUN_DIR = Path(r"D:\nasrin\swatmf_run\fast550_package\stage3_run")
PAR_CSV = RUN_DIR / "swat_modflow_ies_dsi550_fast_stage3.0.par.csv"
OBS_CSV = RUN_DIR / "swat_modflow_ies_dsi550_fast_stage3.0.obs.csv"
OUT_DIR = Path(r"D:\nasrin")          # output figures next to existing gsa figure
OUT_DIR.mkdir(exist_ok=True)

# ── Load ensembles ──────────────────────────────────────────────────────────
par = pd.read_csv(PAR_CSV, index_col=0)
obs = pd.read_csv(OBS_CSV, index_col=0)
par.index = par.index.astype(str)
obs.index = obs.index.astype(str)
common = [r for r in par.index if r in obs.index]
par = par.loc[common]
obs = obs.loc[common]
print(f"Realisations: {len(common)}   Parameters: {list(par.columns)}")

# ── Observed (target) values from PST observation section ──────────────────
# Parse from PST to get observed flow and head values
PST = RUN_DIR / "swat_modflow_ies_dsi550_fast_stage3.pst"
flow_obs, head_obs = {}, {}
in_obs = False
for ln in PST.read_text().splitlines():
    s = ln.strip()
    if s.startswith("* observation data"): in_obs = True; continue
    if in_obs and s.startswith("*"): break
    if not in_obs or not s: continue
    parts = s.split()
    if len(parts) < 2: continue
    try:
        if parts[0].startswith("flow_"):  flow_obs[parts[0]] = float(parts[1])
        elif parts[0].startswith("head_"): head_obs[parts[0]] = float(parts[1])
    except ValueError: continue

print(f"Flow obs: {len(flow_obs)}   Head obs: {len(head_obs)}")

# ── Compute per-realisation scalar metrics ──────────────────────────────────
def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    if len(o) < 2: return np.nan
    d = np.sum((o - o.mean())**2)
    return 1 - np.sum((o - s)**2) / d if d > 0 else np.nan

def rmse_fn(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    return np.sqrt(np.mean((o - s)**2)) if len(o) >= 2 else np.nan

def kge_fn(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    if len(o) < 2: return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1)**2 + (s.std()/o.std() - 1)**2
                       + (s.mean()/o.mean() - 1)**2)

def pbias_fn(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    return 100 * (o - s).sum() / o.sum() if o.sum() != 0 else np.nan

# flow observed array
f_obs_vals = np.array([flow_obs.get(c, np.nan) for c in obs.columns
                        if c in flow_obs])
f_obs_cols  = [c for c in obs.columns if c in flow_obs]
h_obs_vals = np.array([head_obs.get(c, np.nan) for c in obs.columns
                        if c in head_obs])
h_obs_cols  = [c for c in obs.columns if c in head_obs]

metrics_rows = []
for r in common:
    f_sim = obs.loc[r, f_obs_cols].values.astype(float)
    h_sim = obs.loc[r, h_obs_cols].values.astype(float)
    row = {
        "flow_NSE":    nse(f_obs_vals, f_sim),
        "flow_KGE":    kge_fn(f_obs_vals, f_sim),
        "flow_RMSE":   rmse_fn(f_obs_vals, f_sim),
        "flow_PBIAS":  pbias_fn(f_obs_vals, f_sim),
        "head_RMSE":   rmse_fn(h_obs_vals, h_sim),
        "head_mean":   np.nanmean(h_sim - h_obs_vals),   # mean head bias
        "head_var":    np.nanvar(h_sim),
    }
    metrics_rows.append(row)

met = pd.DataFrame(metrics_rows, index=common)
# flip RMSE / PBIAS so higher = worse (keep sign for interpretation)
print("Metrics computed:")
print(met.describe().round(3))

# ── Parameter metadata ─────────────────────────────────────────────────────
# Align with broader candidate set: map final 13 to candidate groups
PARAM_META = {
    "alpha_bf":  {"label": "ALPHA_BF\n(baseflow recession)", "group": "SWAT–GW", "retained": True},
    "ch_k2_m":   {"label": "CH_K2\n(channel conductivity ×)", "group": "SWAT–channel", "retained": True},
    "cn2_m":     {"label": "CN2\n(runoff curve number ×)", "group": "SWAT–surface", "retained": True},
    "esco":      {"label": "ESCO\n(soil evap. comp.)", "group": "SWAT–ET", "retained": True},
    "gw_delay":  {"label": "GW_DELAY\n(recharge lag)", "group": "SWAT–GW", "retained": True},
    "gw_revap":  {"label": "GW_REVAP\n(revap. coeff.)", "group": "SWAT–GW", "retained": True},
    "gwqmn":     {"label": "GWQMN\n(baseflow threshold)", "group": "SWAT–GW", "retained": True},
    "hk_mult":   {"label": "HKmult\n(horiz. K multiplier)", "group": "MODFLOW", "retained": True},
    "rchrg_dp":  {"label": "RCHRG_DP\n(deep perc. frac.)", "group": "SWAT–GW", "retained": True},
    "revapmn":   {"label": "REVAPMN\n(revap. threshold)", "group": "SWAT–GW", "retained": True},
    "ss":        {"label": "Ss\n(specific storage)", "group": "MODFLOW", "retained": True},
    "surlag":    {"label": "SURLAG\n(surface lag)", "group": "SWAT–surface", "retained": True},
    "sy":        {"label": "Sy\n(specific yield)", "group": "MODFLOW", "retained": True},
}
params = list(par.columns)
labels = [PARAM_META[p]["label"] for p in params]
groups = [PARAM_META[p]["group"] for p in params]

# ── Sensitivity: Spearman, Pearson, SRC ────────────────────────────────────
output_cols = list(met.columns)
output_labels = {
    "flow_NSE":   "Flow NSE",
    "flow_KGE":   "Flow KGE",
    "flow_RMSE":  "Flow RMSE",
    "flow_PBIAS": "Flow PBIAS (%)",
    "head_RMSE":  "Head RMSE (m)",
    "head_mean":  "Head Bias (m)",
    "head_var":   "Head Variance",
}

def compute_sensitivity(par_df, met_df):
    n_par = par_df.shape[1]
    n_out = met_df.shape[1]
    spear = np.full((n_par, n_out), np.nan)
    spear_p = np.full((n_par, n_out), np.nan)
    pears = np.full((n_par, n_out), np.nan)
    src   = np.full((n_par, n_out), np.nan)
    for j, oc in enumerate(met_df.columns):
        y = met_df[oc].values.astype(float)
        m_y = np.isfinite(y)
        for i, pc in enumerate(par_df.columns):
            x = par_df[pc].values.astype(float)
            m = m_y & np.isfinite(x)
            if m.sum() < 10: continue
            xi, yi = x[m], y[m]
            rho, pval = stats.spearmanr(xi, yi)
            spear[i, j] = rho
            spear_p[i, j] = pval
            pears[i, j], _ = stats.pearsonr(xi, yi)
            # SRC: standardised regression coeff
            xs = (xi - xi.mean()) / (xi.std() + 1e-30)
            ys = (yi - yi.mean()) / (yi.std() + 1e-30)
            src[i, j] = np.dot(xs, ys) / (len(xs) - 1)
    return (pd.DataFrame(spear,   index=par_df.columns, columns=met_df.columns),
            pd.DataFrame(spear_p, index=par_df.columns, columns=met_df.columns),
            pd.DataFrame(pears,   index=par_df.columns, columns=met_df.columns),
            pd.DataFrame(src,     index=par_df.columns, columns=met_df.columns))

spear_df, spear_p_df, pears_df, src_df = compute_sensitivity(par, met)

# ── Ranked sensitivity table ───────────────────────────────────────────────
# Overall importance = mean |Spearman| across all outputs
importance_all    = spear_df.abs().mean(axis=1).sort_values(ascending=False)
importance_flow   = spear_df[["flow_NSE","flow_KGE","flow_RMSE","flow_PBIAS"]].abs().mean(axis=1).sort_values(ascending=False)
importance_head   = spear_df[["head_RMSE","head_mean","head_var"]].abs().mean(axis=1).sort_values(ascending=False)

ranked_table = pd.DataFrame({
    "Parameter":        importance_all.index,
    "Group":            [PARAM_META[p]["group"] for p in importance_all.index],
    "Retained (final 13)": ["Yes"] * len(params),
    "Sensitivity_flow":  importance_flow[importance_all.index].values.round(3),
    "Sensitivity_head":  importance_head[importance_all.index].values.round(3),
    "Sensitivity_combined": importance_all.values.round(3),
    "Rank_flow":        importance_flow[importance_all.index].rank(ascending=False, method='min').values.astype(int),
    "Rank_head":        importance_head[importance_all.index].rank(ascending=False, method='min').values.astype(int),
    "Rank_combined":    importance_all.rank(ascending=False, method='min').values.astype(int),
})
ranked_table.to_csv(OUT_DIR / "gsa_ranked_table.csv", index=False)
print("\nRanked sensitivity table:")
print(ranked_table.to_string(index=False))

# ── Figure 1: Multi-panel Spearman sensitivity (main manuscript figure) ─────
GROUP_COLORS = {
    "SWAT–surface": "#E67E22",
    "SWAT–ET":      "#27AE60",
    "SWAT–GW":      "#2980B9",
    "MODFLOW":      "#8E44AD",
    "SWAT–channel": "#E74C3C",
}
par_colors = [GROUP_COLORS[PARAM_META[p]["group"]] for p in params]

flow_outputs = ["flow_NSE", "flow_KGE", "flow_RMSE", "flow_PBIAS"]
head_outputs = ["head_RMSE", "head_mean", "head_var"]

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.45)

# Panel A: Spearman heatmap — flow
ax_hm_f = fig.add_subplot(gs[0, :2])
data_f = spear_df[flow_outputs].T
im = ax_hm_f.imshow(data_f.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_hm_f.set_xticks(range(len(params)))
ax_hm_f.set_xticklabels([PARAM_META[p]["label"].replace("\n", " ") for p in params],
                          rotation=45, ha="right", fontsize=8)
ax_hm_f.set_yticks(range(len(flow_outputs)))
ax_hm_f.set_yticklabels([output_labels[o] for o in flow_outputs], fontsize=9)
ax_hm_f.set_title("(A) Spearman Rank Correlation — Streamflow Metrics", fontweight="bold", fontsize=11)
for i in range(len(flow_outputs)):
    for j in range(len(params)):
        v = data_f.values[i, j]
        if not np.isnan(v):
            ax_hm_f.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if abs(v) > 0.5 else "black")
plt.colorbar(im, ax=ax_hm_f, shrink=0.8, label="Spearman ρ")

# Panel B: Spearman heatmap — head
ax_hm_h = fig.add_subplot(gs[1, :2])
data_h = spear_df[head_outputs].T
im2 = ax_hm_h.imshow(data_h.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_hm_h.set_xticks(range(len(params)))
ax_hm_h.set_xticklabels([PARAM_META[p]["label"].replace("\n", " ") for p in params],
                          rotation=45, ha="right", fontsize=8)
ax_hm_h.set_yticks(range(len(head_outputs)))
ax_hm_h.set_yticklabels([output_labels[o] for o in head_outputs], fontsize=9)
ax_hm_h.set_title("(B) Spearman Rank Correlation — Groundwater Head Metrics", fontweight="bold", fontsize=11)
for i in range(len(head_outputs)):
    for j in range(len(params)):
        v = data_h.values[i, j]
        if not np.isnan(v):
            ax_hm_h.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if abs(v) > 0.5 else "black")
plt.colorbar(im2, ax=ax_hm_h, shrink=0.8, label="Spearman ρ")

# Panel C: Combined importance bar (flow + head)
ax_bar = fig.add_subplot(gs[2, :2])
x_pos = np.arange(len(params))
w = 0.35
flow_imp = [importance_flow[p] for p in params]
head_imp = [importance_head[p] for p in params]
ax_bar.bar(x_pos - w/2, flow_imp, w, label="Streamflow", color="#2980B9", alpha=0.85)
ax_bar.bar(x_pos + w/2, head_imp, w, label="Groundwater Head", color="#8E44AD", alpha=0.85)
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([PARAM_META[p]["label"].replace("\n", " ") for p in params],
                        rotation=45, ha="right", fontsize=8)
ax_bar.set_ylabel("Mean |Spearman ρ|", fontsize=10)
ax_bar.set_title("(C) Mean Absolute Spearman Correlation — Flow vs. Head", fontweight="bold", fontsize=11)
ax_bar.legend(fontsize=9)
ax_bar.set_ylim(0, 1)
ax_bar.axhline(0.3, ls="--", lw=0.8, color="gray", alpha=0.6)
ax_bar.text(len(params)-0.5, 0.31, "sensitivity threshold 0.3", fontsize=7, color="gray")
ax_bar.grid(axis="y", alpha=0.3)

# Panel D: Group legend
ax_leg = fig.add_subplot(gs[0, 2])
ax_leg.axis("off")
legend_patches = [plt.Rectangle((0,0),1,1, color=c, label=g)
                  for g, c in GROUP_COLORS.items()]
ax_leg.legend(handles=legend_patches, title="Parameter Group", loc="center",
              fontsize=9, title_fontsize=10, frameon=True)
ax_leg.set_title("Parameter Groups", fontweight="bold", fontsize=10)

# Panel E: Scatter example — most influential flow param
ax_sc = fig.add_subplot(gs[1, 2])
top_flow_par = importance_flow.index[0]
top_flow_out = "flow_NSE"
x = par[top_flow_par].values
y = met[top_flow_out].values
m = np.isfinite(x) & np.isfinite(y)
rho_val = spear_df.loc[top_flow_par, top_flow_out]
ax_sc.scatter(x[m], y[m], s=12, alpha=0.5, color="#2980B9")
ax_sc.set_xlabel(PARAM_META[top_flow_par]["label"].replace("\n", " "), fontsize=9)
ax_sc.set_ylabel("Flow NSE", fontsize=9)
ax_sc.set_title(f"(E) Most influential: {top_flow_par}\nSpearman ρ = {rho_val:.3f}",
                fontweight="bold", fontsize=10)
ax_sc.grid(alpha=0.3)

# Panel F: Scatter example — most influential head param
ax_sc2 = fig.add_subplot(gs[2, 2])
top_head_par = importance_head.index[0]
top_head_out = "head_RMSE"
x2 = par[top_head_par].values
y2 = met[top_head_out].values
m2 = np.isfinite(x2) & np.isfinite(y2)
rho_val2 = spear_df.loc[top_head_par, top_head_out]
ax_sc2.scatter(x2[m2], y2[m2], s=12, alpha=0.5, color="#8E44AD")
ax_sc2.set_xlabel(PARAM_META[top_head_par]["label"].replace("\n", " "), fontsize=9)
ax_sc2.set_ylabel("Head RMSE (m)", fontsize=9)
ax_sc2.set_title(f"(F) Most influential for head: {top_head_par}\nSpearman ρ = {rho_val2:.3f}",
                 fontweight="bold", fontsize=10)
ax_sc2.grid(alpha=0.3)

fig.suptitle(
    "Prior-Ensemble Sensitivity Screening (n=550)\nSpearman Rank Correlations between\n"
    "Candidate Parameters and Model Outputs",
    fontsize=13, fontweight="bold", y=1.01
)
fig.savefig(OUT_DIR / "gsa_spearman_panels.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: gsa_spearman_panels.png")

# ── Figure 2: Full Spearman/Pearson/SRC comparison heatmap ─────────────────
fig2, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
method_data = [
    ("Spearman ρ", spear_df),
    ("Pearson r",  pears_df),
    ("SRC",        src_df),
]
all_outs = flow_outputs + head_outputs
for ax, (title, df) in zip(axes, method_data):
    data = df[all_outs]
    im = ax.imshow(data.values.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([PARAM_META[p]["label"].replace("\n", " ") for p in params],
                        rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_outs)))
    ax.set_yticklabels([output_labels[o] for o in all_outs], fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=11)
    for j in range(len(all_outs)):
        for i in range(len(params)):
            v = data.values[i, j]
            if not np.isnan(v):
                ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(v) > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
fig2.suptitle("Sensitivity Metrics Comparison: Spearman, Pearson, SRC\n"
              "Prior Ensemble (n=550), All Parameters × All Outputs",
              fontsize=12, fontweight="bold")
fig2.tight_layout()
fig2.savefig(OUT_DIR / "gsa_all_metrics_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved: gsa_all_metrics_heatmap.png")

# ── Figure 3: Flow bar plot ─────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
params_sorted_f = importance_flow.index.tolist()
vals_f = [importance_flow[p] for p in params_sorted_f]
colors_f = [GROUP_COLORS[PARAM_META[p]["group"]] for p in params_sorted_f]
bars = ax3.bar(range(len(params_sorted_f)), vals_f, color=colors_f, edgecolor="white", lw=0.5)
ax3.set_xticks(range(len(params_sorted_f)))
ax3.set_xticklabels([PARAM_META[p]["label"].replace("\n", " ") for p in params_sorted_f],
                     rotation=45, ha="right", fontsize=9)
ax3.set_ylabel("Mean |Spearman ρ|  (flow metrics)", fontsize=11)
ax3.set_title("Streamflow Sensitivity — Parameter Ranking\n"
              "Mean absolute Spearman correlation across NSE, KGE, RMSE, PBIAS",
              fontweight="bold", fontsize=11)
ax3.axhline(0.3, ls="--", lw=1, color="gray"); ax3.text(len(params_sorted_f)-0.5, 0.31, "0.30", fontsize=8, color="gray")
ax3.set_ylim(0, 1); ax3.grid(axis="y", alpha=0.3)
legend_patches = [plt.Rectangle((0,0),1,1, color=c, label=g) for g, c in GROUP_COLORS.items()]
ax3.legend(handles=legend_patches, fontsize=9, loc="upper right")
fig3.tight_layout()
fig3.savefig(OUT_DIR / "gsa_flow_barplot.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved: gsa_flow_barplot.png")

# ── Figure 4: Head bar plot ─────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 5))
params_sorted_h = importance_head.index.tolist()
vals_h = [importance_head[p] for p in params_sorted_h]
colors_h = [GROUP_COLORS[PARAM_META[p]["group"]] for p in params_sorted_h]
ax4.bar(range(len(params_sorted_h)), vals_h, color=colors_h, edgecolor="white", lw=0.5)
ax4.set_xticks(range(len(params_sorted_h)))
ax4.set_xticklabels([PARAM_META[p]["label"].replace("\n", " ") for p in params_sorted_h],
                     rotation=45, ha="right", fontsize=9)
ax4.set_ylabel("Mean |Spearman ρ|  (head metrics)", fontsize=11)
ax4.set_title("Groundwater Head Sensitivity — Parameter Ranking\n"
              "Mean absolute Spearman correlation across RMSE, mean bias, variance",
              fontweight="bold", fontsize=11)
ax4.axhline(0.3, ls="--", lw=1, color="gray"); ax4.text(len(params_sorted_h)-0.5, 0.31, "0.30", fontsize=8, color="gray")
ax4.set_ylim(0, 1); ax4.grid(axis="y", alpha=0.3)
ax4.legend(handles=legend_patches, fontsize=9, loc="upper right")
fig4.tight_layout()
fig4.savefig(OUT_DIR / "gsa_head_barplot.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Saved: gsa_head_barplot.png")

# ── Figure 5: Combined bar plot ─────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(10, 5))
params_sorted_c = importance_all.index.tolist()
vals_c = [importance_all[p] for p in params_sorted_c]
colors_c = [GROUP_COLORS[PARAM_META[p]["group"]] for p in params_sorted_c]
ax5.bar(range(len(params_sorted_c)), vals_c, color=colors_c, edgecolor="white", lw=0.5)
ax5.set_xticks(range(len(params_sorted_c)))
ax5.set_xticklabels([PARAM_META[p]["label"].replace("\n", " ") for p in params_sorted_c],
                     rotation=45, ha="right", fontsize=9)
ax5.set_ylabel("Mean |Spearman ρ|  (all outputs)", fontsize=11)
ax5.set_title("Combined Sensitivity Ranking — Flow + Head\n"
              "Mean absolute Spearman correlation across all output metrics",
              fontweight="bold", fontsize=11)
ax5.axhline(0.3, ls="--", lw=1, color="gray"); ax5.text(len(params_sorted_c)-0.5, 0.31, "0.30", fontsize=8, color="gray")
ax5.set_ylim(0, 1); ax5.grid(axis="y", alpha=0.3)
ax5.legend(handles=legend_patches, fontsize=9, loc="upper right")
fig5.tight_layout()
fig5.savefig(OUT_DIR / "gsa_combined_barplot.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print("Saved: gsa_combined_barplot.png")

# ── Print manuscript paragraph ──────────────────────────────────────────────
top3_flow = importance_flow.index[:3].tolist()
top3_head = importance_head.index[:3].tolist()
low_flow  = importance_flow[importance_flow < 0.15].index.tolist()
low_head  = importance_head[importance_head < 0.15].index.tolist()

print("\n" + "="*80)
print("MANUSCRIPT PARAGRAPH")
print("="*80)
print(f"""
Prior to the final DSI-based uncertainty analysis, a broader candidate parameter
set was evaluated using prior-ensemble sensitivity screening. Starting from
parameters commonly used in coupled SWAT–MODFLOW applications—including SWAT
hydrologic parameters (e.g., CN2, ESCO, ALPHA_BF, GW_DELAY, GWQMN, SURLAG,
RCHRG_DP, GW_REVAP, REVAPMN, CH_K2) and MODFLOW aquifer parameters (horizontal
hydraulic conductivity multiplier HKmult, specific yield Sy, and specific
storage Ss)—we evaluated ensemble-based parameter sensitivity analysis using a
{len(common)}-member Latin Hypercube prior ensemble. Model responses were
evaluated with respect to monthly streamflow (NSE, KGE, RMSE, PBIAS) and
groundwater head predictions (RMSE, mean bias, ensemble variance). Spearman rank
correlation, Pearson correlation, and standardised regression coefficients were
computed between each parameter and each output metric.

The results indicated that the dominant controls on streamflow response were
{", ".join(top3_flow)}, while the dominant controls on groundwater head response
were {", ".join(top3_head)}. Parameters with consistently low sensitivity
(mean |Spearman ρ| < 0.15 across all outputs) were identified as candidates for
exclusion: {", ".join(low_flow) if low_flow else "none fell below this threshold"} for streamflow and
{", ".join(low_head) if low_head else "none"} for head. All 13 retained parameters
showed meaningful sensitivity to at least one target output. Therefore, the final
13-parameter set was selected as a parsimonious, sensitivity-screened
representation of the dominant controls on model behaviour, rather than chosen
arbitrarily. This choice improved computational feasibility, ensemble stability,
and interpretability for the subsequent DSI prediction and data worth analysis.
However, the resulting parameterisation still does not fully represent local
structural nonuniqueness or alternative spatial heterogeneity patterns, which
remains a limitation and a direction for future work.
""")
print("="*80)
print("\nAll figures and table saved to:", OUT_DIR)
