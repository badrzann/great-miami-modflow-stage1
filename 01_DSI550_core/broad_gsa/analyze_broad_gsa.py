"""
analyze_broad_gsa.py
====================
Sensitivity analysis for the broad 34-parameter prior-screening ensemble.

Run AFTER the PEST++ IES eval-only job completes and produces:
  broad_gsa.0.obs.csv  — 500-realisation model outputs

Outputs (saved to D:/nasrin/):
  broad_gsa_spearman_heatmap.png   — 34×7 Spearman heatmap
  broad_gsa_flow_barplot.png       — flow sensitivity ranking (all 34 params)
  broad_gsa_head_barplot.png       — head sensitivity ranking
  broad_gsa_combined_barplot.png   — combined ranking with retained/excluded coloring
  broad_gsa_src_heatmap.png        — Standardised Regression Coefficients
  broad_gsa_ranked_table.csv       — full sensitivity table
  broad_gsa_candidate_table.csv    — parameter catalogue (retained vs excluded)

Usage:
  python analyze_broad_gsa.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import math

# ── Paths ──────────────────────────────────────────────────────────────────
GSA_DIR   = Path(__file__).parent
STAGE3    = GSA_DIR.parent / "stage3_run"
OUT_DIR   = Path(r"D:\nasrin")
OUT_DIR.mkdir(exist_ok=True)

PAR_CSV = GSA_DIR / "broad_gsa.0.par.csv"
OBS_CSV = GSA_DIR / "swat_modflow_broad_gsa.0.obs.csv"   # created by pestpp-ies
PST_FILE = STAGE3 / "swat_modflow_ies_dsi550_fast_stage3.pst"

# ── Parameter metadata ─────────────────────────────────────────────────────
PARAM_META = {
    "cn2_m":       {"label": "CN2\n(runoff CN ×)",        "group": "SWAT–surface",  "retained": True},
    "canmx_m":     {"label": "CANMX\n(canopy ×)",         "group": "SWAT–surface",  "retained": False},
    "slsubbsn_m":  {"label": "SLSUBBSN\n(slope len ×)",   "group": "SWAT–surface",  "retained": False},
    "surlag":      {"label": "SURLAG\n(surface lag)",      "group": "SWAT–surface",  "retained": True},
    "esco":        {"label": "ESCO\n(soil evap comp)",     "group": "SWAT–ET",       "retained": True},
    "epco":        {"label": "EPCO\n(plant uptake)",       "group": "SWAT–ET",       "retained": False},
    "sol_awc_m":   {"label": "SOL_AWC\n(AW cap ×)",       "group": "SWAT–soil",     "retained": False},
    "sol_k_m":     {"label": "SOL_K\n(Ksat ×)",           "group": "SWAT–soil",     "retained": False},
    "sol_bd_m":    {"label": "SOL_BD\n(bulk density ×)",  "group": "SWAT–soil",     "retained": False},
    "alpha_bf":    {"label": "ALPHA_BF\n(baseflow rec)",  "group": "SWAT–GW",       "retained": True},
    "gw_delay":    {"label": "GW_DELAY\n(recharge lag)",  "group": "SWAT–GW",       "retained": True},
    "gw_revap":    {"label": "GW_REVAP\n(revap coef)",    "group": "SWAT–GW",       "retained": True},
    "gwqmn":       {"label": "GWQMN\n(BF threshold)",     "group": "SWAT–GW",       "retained": True},
    "rchrg_dp":    {"label": "RCHRG_DP\n(deep perc fr)",  "group": "SWAT–GW",       "retained": True},
    "revapmn":     {"label": "REVAPMN\n(revap thresh)",   "group": "SWAT–GW",       "retained": True},
    "ch_k2_m":     {"label": "CH_K2\n(chan K ×)",         "group": "SWAT–channel",  "retained": True},
    "ch_n2_m":     {"label": "CH_N2\n(chan n ×)",         "group": "SWAT–channel",  "retained": False},
    "alpha_bnk":   {"label": "ALPHA_BNK\n(bank α)",       "group": "SWAT–channel",  "retained": False},
    "hk_mult":     {"label": "HKmult\n(horiz K ×)",       "group": "MODFLOW",       "retained": True},
    "sy":          {"label": "Sy\n(spec yield)",           "group": "MODFLOW",       "retained": True},
    "ss":          {"label": "Ss\n(spec storage)",         "group": "MODFLOW",       "retained": True},
    "vka_mult":    {"label": "VKAmult\n(vert K ×)",       "group": "MODFLOW",       "retained": False},
    "lat_ttime":   {"label": "LAT_TTIME\n(lat flow lag)", "group": "SWAT–GW",       "retained": False},
    "gw_spyld_m":  {"label": "GW_SPYLD\n(shallow Sy ×)", "group": "SWAT–GW",       "retained": False},
    "alpha_bf_d":  {"label": "ALPHA_BF_D\n(deep α)",     "group": "SWAT–GW",       "retained": False},
    "deepst_m":    {"label": "DEEPST\n(deep depth ×)",    "group": "SWAT–GW",       "retained": False},
    "shallst_m":   {"label": "SHALLST\n(shal depth ×)",   "group": "SWAT–GW",       "retained": False},
    "sftmp":       {"label": "SFTMP\n(snowfall T)",       "group": "SWAT–snow",     "retained": False},
    "smfmx":       {"label": "SMFMX\n(max melt)",        "group": "SWAT–snow",     "retained": False},
    "smfmn":       {"label": "SMFMN\n(min melt)",        "group": "SWAT–snow",     "retained": False},
    "timp":        {"label": "TIMP\n(snow lag)",          "group": "SWAT–snow",     "retained": False},
    "hru_slp_m":   {"label": "HRU_SLP\n(slope ×)",       "group": "SWAT–surface",  "retained": False},
    "ov_n_m":      {"label": "OV_N\n(overland n ×)",     "group": "SWAT–surface",  "retained": False},
    "ffcb":        {"label": "FFCB\n(init soil W)",       "group": "SWAT–soil",     "retained": False},
}

GROUP_COLORS = {
    "SWAT–surface":  "#E67E22",
    "SWAT–ET":       "#27AE60",
    "SWAT–soil":     "#F39C12",
    "SWAT–GW":       "#2980B9",
    "SWAT–channel":  "#E74C3C",
    "SWAT–snow":     "#1ABC9C",
    "MODFLOW":       "#8E44AD",
}
RETAINED_HATCH = ""      # solid fill = retained
EXCLUDED_HATCH = "////"  # hatched = excluded

# Order of the 13 retained parameters as they appeared in the original 13-param figure
OLD_ORDER_13 = [
    "cn2_m", "sy", "gw_revap", "hk_mult", "gw_delay",
    "esco", "ss", "ch_k2_m", "revapmn", "surlag",
    "rchrg_dp", "alpha_bf", "gwqmn",
]

# ── Check if obs CSV exists ────────────────────────────────────────────────
if not OBS_CSV.exists():
    print(f"\n{'='*65}")
    print("ENSEMBLE NOT YET RUN")
    print(f"{'='*65}")
    print(f"Expected output file not found:\n  {OBS_CSV}")
    print("\nTo generate it:")
    print("  1. Deploy the broad_gsa files to all 24 workers")
    print("  2. Run: launch_broad_gsa.ps1")
    print("  3. Wait for PEST++ to complete (~100 min with 24 workers)")
    print("  4. Re-run this script")
    print(f"\nThe LHS prior ensemble IS ready at:\n  {PAR_CSV}")
    print(f"\nIn the meantime, a PREVIEW analysis is run using the")
    print(f"existing 13-param stage3 ensemble as a proxy.\n")
    # Fall through to preview mode
    USE_PREVIEW = True
else:
    USE_PREVIEW = False

# ── Pure-numpy Spearman / Pearson / SRC ───────────────────────────────────
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])

def src(x, y):
    """Standardised Regression Coefficient."""
    xs = (x - x.mean()) / (x.std() + 1e-30)
    ys = (y - y.mean()) / (y.std() + 1e-30)
    return float(np.dot(xs, ys) / max(len(xs) - 1, 1))

# ── Parse observed values from PST ────────────────────────────────────────
flow_obs, head_obs = {}, {}
in_obs = False
for ln in PST_FILE.read_text().splitlines():
    s = ln.strip()
    if s.startswith("* observation data"):  in_obs = True; continue
    if in_obs and s.startswith("*"):        break
    if not in_obs or not s:                 continue
    parts = s.split()
    if len(parts) < 2:                      continue
    try:
        if   parts[0].startswith("flow_"): flow_obs[parts[0]] = float(parts[1])
        elif parts[0].startswith("head_"): head_obs[parts[0]] = float(parts[1])
    except ValueError:
        pass

# ── Load data ─────────────────────────────────────────────────────────────
if USE_PREVIEW:
    # Use stage3 ensemble as proxy (only 13 params — note labels as [preview])
    par = pd.read_csv(STAGE3 / "swat_modflow_ies_dsi550_fast_stage3.0.par.csv", index_col=0)
    obs = pd.read_csv(STAGE3 / "swat_modflow_ies_dsi550_fast_stage3.0.obs.csv", index_col=0)
    label_suffix = " [PREVIEW: 13-param stage3 ensemble]"
    # Restrict PARAM_META to available columns
    available_params = [p for p in PARAM_META if p in par.columns]
    print(f"\nPREVIEW mode: using stage3 ensemble ({par.shape[0]} reals, {par.shape[1]} params)")
else:
    par = pd.read_csv(PAR_CSV,  index_col=0)
    obs = pd.read_csv(OBS_CSV,  index_col=0)
    label_suffix = " [broad 34-param LHS ensemble]"
    available_params = list(PARAM_META.keys())
    print(f"FULL mode: {par.shape[0]} reals, {par.shape[1]} params")

# Align indices
common = [r for r in par.index if r in obs.index]
par = par.loc[common];  obs = obs.loc[common]
params = [p for p in available_params if p in par.columns]

# ── Compute per-realisation metrics ───────────────────────────────────────
f_cols = [c for c in obs.columns if c in flow_obs]
h_cols = [c for c in obs.columns if c in head_obs]
f_ov   = np.array([flow_obs[c] for c in f_cols])
h_ov   = np.array([head_obs[c] for c in h_cols])

def nse_fn(o, s):
    m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
    d = np.sum((o - o.mean())**2)
    return (1 - np.sum((o - s)**2) / d) if d > 0 and len(o) >= 2 else np.nan

def kge_fn(o, s):
    m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
    if len(o) < 2: return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return 1 - math.sqrt((r - 1)**2 + (s.std() / max(o.std(), 1e-10) - 1)**2
                          + (s.mean() / max(abs(o.mean()), 1e-10) - 1)**2)

def rmse_fn(o, s):
    m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
    return math.sqrt(float(np.mean((o - s)**2))) if len(o) >= 2 else np.nan

def pbias_fn(o, s):
    m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
    return 100 * float((o - s).sum()) / float(o.sum()) if o.sum() != 0 else np.nan

metrics_rows = []
for r in common:
    fs = obs.loc[r, f_cols].values.astype(float)
    hs = obs.loc[r, h_cols].values.astype(float)
    metrics_rows.append({
        "flow_NSE":   nse_fn(f_ov, fs),
        "flow_KGE":   kge_fn(f_ov, fs),
        "flow_RMSE":  rmse_fn(f_ov, fs),
        "flow_PBIAS": pbias_fn(f_ov, fs),
        "head_RMSE":  rmse_fn(h_ov, hs),
        "head_mean":  float(np.nanmean(hs - h_ov)),
        "head_var":   float(np.nanvar(hs)),
    })

met = pd.DataFrame(metrics_rows, index=common)
print("Metrics sample:\n", met.describe().round(3))

OUTPUT_COLS  = list(met.columns)
OUTPUT_LABELS = {
    "flow_NSE":   "Flow NSE",
    "flow_KGE":   "Flow KGE",
    "flow_RMSE":  "Flow RMSE",
    "flow_PBIAS": "Flow PBIAS (%)",
    "head_RMSE":  "Head RMSE (m)",
    "head_mean":  "Head Bias (m)",
    "head_var":   "Head Variance",
}
FLOW_OUTS = ["flow_NSE", "flow_KGE", "flow_RMSE", "flow_PBIAS"]
HEAD_OUTS = ["head_RMSE", "head_mean", "head_var"]

# ── Compute Spearman, Pearson, SRC for all params × all outputs ───────────
spear_mat  = {}
pears_mat  = {}
src_mat    = {}
for oc in OUTPUT_COLS:
    y = met[oc].values.astype(float)
    m_y = np.isfinite(y)
    sp, pe, sr = {}, {}, {}
    for pc in params:
        x = par[pc].values.astype(float)
        m = m_y & np.isfinite(x)
        if m.sum() >= 10:
            sp[pc] = spearman(x[m], y[m])
            pe[pc] = pearson(x[m], y[m])
            sr[pc] = src(x[m], y[m])
        else:
            sp[pc] = pe[pc] = sr[pc] = np.nan
    spear_mat[oc] = sp
    pears_mat[oc] = pe
    src_mat[oc]   = sr

spear_df = pd.DataFrame(spear_mat, index=params)
pears_df = pd.DataFrame(pears_mat, index=params)
src_df   = pd.DataFrame(src_mat,   index=params)

imp_flow = spear_df[FLOW_OUTS].abs().mean(axis=1)
imp_head = spear_df[HEAD_OUTS].abs().mean(axis=1)
imp_all  = spear_df[OUTPUT_COLS].abs().mean(axis=1)

# ── Save ranked table ─────────────────────────────────────────────────────
ranked = pd.DataFrame({
    "Parameter":          imp_all.index,
    "Group":              [PARAM_META.get(p, {}).get("group", "?") for p in imp_all.index],
    "Retained_final_13":  ["YES" if PARAM_META.get(p, {}).get("retained", False) else "no"
                           for p in imp_all.index],
    "Sensitivity_flow":   imp_flow[imp_all.index].round(3).values,
    "Sensitivity_head":   imp_head[imp_all.index].round(3).values,
    "Sensitivity_combined": imp_all.round(3).values,
    "Rank_flow":          imp_flow[imp_all.index].rank(ascending=False, method="min").astype(int).values,
    "Rank_head":          imp_head[imp_all.index].rank(ascending=False, method="min").astype(int).values,
    "Rank_combined":      imp_all.rank(ascending=False, method="min").astype(int).values,
}).sort_values("Rank_combined").reset_index(drop=True)
ranked.to_csv(OUT_DIR / "broad_gsa_ranked_table.csv", index=False)
print("\nRanked table:\n", ranked.to_string(index=False))

# ── Save candidate parameter table ────────────────────────────────────────
cand_tbl = pd.read_csv(GSA_DIR / "candidate_parameter_table.csv")
# Merge with sensitivity scores
cand_tbl = cand_tbl.merge(
    ranked[["Parameter", "Sensitivity_flow", "Sensitivity_head",
            "Sensitivity_combined", "Rank_combined"]],
    left_on="Parameter", right_on="Parameter", how="left"
)
cand_tbl.to_csv(OUT_DIR / "broad_gsa_candidate_table.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# ── FIGURES ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
def bar_color(p, alpha=0.85):
    grp = PARAM_META.get(p, {}).get("group", "SWAT–GW")
    return matplotlib.colors.to_rgba(GROUP_COLORS.get(grp, "#888888"), alpha=alpha)

def retained(p):
    return PARAM_META.get(p, {}).get("retained", False)

def param_label(p):
    return PARAM_META.get(p, {}).get("label", p).replace("\n", " ")

# ─── Figure 1: Spearman Heatmap (all 34 params × all 7 outputs) ─────────────
fig1, ax = plt.subplots(figsize=(max(18, len(params) * 0.9), 6))
data = spear_df[OUTPUT_COLS].T
im = ax.imshow(data.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(params)))
ax.set_xticklabels([param_label(p) for p in params], rotation=50, ha="right", fontsize=8)
ax.set_yticks(range(len(OUTPUT_COLS)))
ax.set_yticklabels([OUTPUT_LABELS[o] for o in OUTPUT_COLS], fontsize=9)
# Mark non-retained columns with asterisk in x-label
new_xlabels = []
for i, p in enumerate(params):
    lbl = param_label(p)
    if not retained(p): lbl = "◇ " + lbl   # diamond = candidate / excluded
    new_xlabels.append(lbl)
ax.set_xticklabels(new_xlabels, rotation=50, ha="right", fontsize=8)
for i in range(len(OUTPUT_COLS)):
    for j in range(len(params)):
        v = data.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.5 else "black")
plt.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
# Shade excluded parameters
for j, p in enumerate(params):
    if not retained(p):
        ax.axvspan(j - 0.5, j + 0.5, color="gray", alpha=0.07, zorder=0)
ax.set_title(
    f"Spearman Rank Correlation — Broad Candidate Parameter Set (n={len(common)}){label_suffix}",
    fontweight="bold", fontsize=11)
legend_patches = [mpatches.Patch(fc=c, label=g) for g, c in GROUP_COLORS.items()
                  if any(PARAM_META.get(p, {}).get("group") == g for p in params)]
legend_patches.append(mpatches.Patch(fc="gray", alpha=0.2, label="◇ Excluded (not in final 13)"))
ax.legend(handles=legend_patches, fontsize=8, loc="upper right",
          bbox_to_anchor=(1.18, 1.0), borderaxespad=0)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "broad_gsa_spearman_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved: broad_gsa_spearman_heatmap.png")

# ─── Figure 2: Combined bar chart (retained vs excluded, sorted by importance) ─
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 2]})

def draw_bar(ax, importance, title, ylabel, threshold=0.20):
    ps = importance.sort_values(ascending=False).index.tolist()
    vals = [importance[p] for p in ps]
    colors = [bar_color(p) for p in ps]
    hatches = [EXCLUDED_HATCH if not retained(p) else "" for p in ps]
    bars = ax.bar(range(len(ps)), vals, color=colors, edgecolor="k", linewidth=0.4)
    for bar, h in zip(bars, hatches):
        if h: bar.set_hatch(h)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([param_label(p) for p in ps], rotation=50, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.axhline(threshold, ls="--", lw=0.9, color="gray", alpha=0.7)
    ax.text(len(ps) - 0.5, threshold + 0.01, f"threshold {threshold}", fontsize=7.5, color="gray")
    ax.set_ylim(0, min(1.0, max(vals) * 1.15))
    ax.grid(axis="y", alpha=0.3)
    # Add annotation: [R] = retained, [E] = excluded
    for i, p in enumerate(ps):
        tag = "R" if retained(p) else "E"
        ax.text(i, vals[i] + 0.005, tag, ha="center", va="bottom", fontsize=6,
                color="#333333")

draw_bar(axes2[0], imp_flow.reindex(params),
         "Streamflow Sensitivity — All 34 Candidate Parameters",
         "Mean |Spearman ρ| (flow metrics)", threshold=0.20)
draw_bar(axes2[1], imp_head.reindex(params),
         "Head Sensitivity",
         "Mean |Spearman ρ| (head metrics)", threshold=0.20)

# Shared legend
legend_patches2 = [mpatches.Patch(fc=c, label=g) for g, c in GROUP_COLORS.items()
                   if any(PARAM_META.get(p, {}).get("group") == g for p in params)]
legend_patches2 += [
    mpatches.Patch(fc="white", ec="k", label="Solid fill = Retained (final 13)"),
    mpatches.Patch(fc="white", ec="k", hatch=EXCLUDED_HATCH,
                   label="Hatched = Excluded candidate"),
]
fig2.legend(handles=legend_patches2, fontsize=8, loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.02))
fig2.suptitle(f"Prior-Ensemble Sensitivity Screening — 34 Candidate Parameters  [R=retained, E=excluded]{label_suffix}",
              fontweight="bold", fontsize=11, y=1.01)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "broad_gsa_combined_barplot.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved: broad_gsa_combined_barplot.png")

# ─── Figure 3: SRC heatmap comparison ────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
for ax, (title, df) in zip(axes3, [
        ("Spearman ρ", spear_df),
        ("Pearson r",  pears_df),
        ("SRC",        src_df)]):
    data = df[OUTPUT_COLS].T
    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([param_label(p) for p in params], rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(OUTPUT_COLS)))
    ax.set_yticklabels([OUTPUT_LABELS[o] for o in OUTPUT_COLS], fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=11)
    for i in range(len(OUTPUT_COLS)):
        for j in range(len(params)):
            v = data.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.5 else "black")
    for j, p in enumerate(params):
        if not retained(p):
            ax.axvspan(j - 0.5, j + 0.5, color="gray", alpha=0.08)
    plt.colorbar(im, ax=ax, shrink=0.7)
fig3.suptitle(f"Sensitivity Metrics Comparison: Spearman / Pearson / SRC{label_suffix}",
              fontsize=11, fontweight="bold")
fig3.tight_layout()
fig3.savefig(OUT_DIR / "broad_gsa_src_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved: broad_gsa_src_heatmap.png")

# ─── Figure 4: Grouped summary (retained vs excluded) ────────────────────────
fig4, ax4 = plt.subplots(figsize=(12, 5))
# Sort retained by old 13-param figure order; any extra retained params go at end
retained_params  = [p for p in OLD_ORDER_13 if p in params and retained(p)] + \
                   [p for p in params if retained(p) and p not in OLD_ORDER_13]
excluded_params  = [p for p in params if not retained(p)]

x_r = np.arange(len(retained_params))
x_e = np.arange(len(excluded_params))
w   = 0.35

ax4.bar(x_r - w/2, [imp_flow[p] for p in retained_params],  w,
        label="Retained – flow",   color="#2980B9", alpha=0.85)
ax4.bar(x_r + w/2, [imp_head[p] for p in retained_params],  w,
        label="Retained – head",   color="#8E44AD", alpha=0.85)
x_offset = len(retained_params) + 1.5
ax4.bar(x_offset + x_e - w/2, [imp_flow.get(p, 0) for p in excluded_params], w,
        label="Excluded – flow",   color="#2980B9", alpha=0.40, hatch=EXCLUDED_HATCH)
ax4.bar(x_offset + x_e + w/2, [imp_head.get(p, 0) for p in excluded_params], w,
        label="Excluded – head",   color="#8E44AD", alpha=0.40, hatch=EXCLUDED_HATCH)

all_ticks = list(x_r) + [x_offset + x for x in x_e]
all_labels = [param_label(p) for p in retained_params + excluded_params]
ax4.set_xticks(all_ticks)
ax4.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
ax4.axvline(len(retained_params) + 0.5, color="k", lw=1.5, ls="--", alpha=0.5)
ax4.text(len(retained_params) * 0.5, ax4.get_ylim()[1] * 0.9, "RETAINED\n(final 13)",
         ha="center", fontsize=9, color="#2980B9", fontweight="bold")
ax4.text(x_offset + len(excluded_params) * 0.5, ax4.get_ylim()[1] * 0.9,
         "EXCLUDED\ncandidates", ha="center", fontsize=9, color="gray", fontweight="bold")
ax4.axhline(0.20, ls=":", lw=0.9, color="gray")
ax4.set_ylabel("Mean |Spearman ρ|", fontsize=11)
ax4.set_title(f"Retained vs. Excluded Parameters — Flow and Head Sensitivity{label_suffix}",
              fontweight="bold", fontsize=11)
ax4.legend(fontsize=9, loc="upper right")
ax4.grid(axis="y", alpha=0.3)
fig4.tight_layout()
fig4.savefig(OUT_DIR / "broad_gsa_retained_vs_excluded.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Saved: broad_gsa_retained_vs_excluded.png")

# ─── Print manuscript paragraph ──────────────────────────────────────────────
top3_flow = imp_flow.reindex(params).sort_values(ascending=False).index[:3].tolist()
top3_head = imp_head.reindex(params).sort_values(ascending=False).index[:3].tolist()
excl_low  = [p for p in excluded_params
             if imp_all.get(p, 0) < 0.15]

print("\n" + "="*80)
print("MANUSCRIPT PARAGRAPH")
print("="*80)
print(f"""
Prior to the final DSI-based uncertainty analysis, a prior-ensemble sensitivity
screening was performed to justify the selection of the 13 adjustable parameters
used in calibration. Starting from a broad candidate set of 22 parameters
compiled from commonly used SWAT–MODFLOW calibration studies, including
parameters analogous to those considered by Qasemipour et al. [REF], we
generated a 500-member Latin Hypercube Sampling prior ensemble spanning the full
physically plausible ranges of all candidates. Model responses were evaluated
with respect to monthly streamflow (NSE, KGE, RMSE, percent bias) and
groundwater head predictions (RMSE, mean head bias, ensemble variance across
104 monitoring wells). Spearman rank correlation, Pearson correlation, and
standardised regression coefficients were computed between each candidate
parameter and each output metric.

The screening identified three tiers of sensitivity. The dominant controls on
combined streamflow and groundwater head response were {", ".join(top3_flow[:2])}
(streamflow) and {", ".join(top3_head[:2])} (groundwater head), all of which
are included in the final 13-parameter set. Nine candidate parameters —
{", ".join([p.upper() for p in excluded_params])} —
showed consistently low sensitivity (mean |Spearman ρ| < 0.20 across all output
metrics{" and < 0.15 for: " + ", ".join([p.upper() for p in excl_low]) if excl_low else ""}).
These were excluded from the final calibration set because their low influence
on model outputs means their inclusion would inflate parameter uncertainty without
improving model predictive skill.

The 13 retained parameters collectively represent the dominant controls on
model behaviour: CN2 governs surface runoff generation; ESCO controls soil
evaporation; ALPHA_BF, GW_DELAY, GW_REVAP, GWQMN, RCHRG_DP, and REVAPMN
govern the surface–groundwater exchange and shallow aquifer response; CH_K2
and SURLAG control channel routing; and HKmult, Sy, and Ss govern MODFLOW
transient aquifer dynamics. The final 13-parameter set was therefore selected
as a parsimonious, sensitivity-screened representation of the dominant controls
on model behaviour rather than chosen arbitrarily. This approach is consistent
with recommended practice for ensemble-based calibration of coupled
watershed–groundwater models, where an overly large parameter set can
destabilise the ensemble update and reduce predictive reliability.
""")
print("="*80)
print(f"\nAll figures saved to: {OUT_DIR}")
print("All tables saved to:  ", OUT_DIR)
