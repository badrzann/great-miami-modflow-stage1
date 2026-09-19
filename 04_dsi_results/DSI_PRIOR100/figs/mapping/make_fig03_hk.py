"""
make_fig03_hk.py
================
Generate Fig. 3 for the GMRW paper:
  (A) Before Calibration — Prior Ensemble Mean HK map
  (B) After Calibration  — Posterior HK map (hk_mult = 1.00)
  (C) Distribution of hk_mult (prior ensemble histogram + KDE)

Output:
  D:/GMRW/finalresult/fig/fig03_hk_map.png   (300 DPI)
  D:/GMRW/finalresult/fig/fig03_hk_map.pdf
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# PATHS AND GRID CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE = "D:/GMRW/finalresult/swatmf_run"
OUT  = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

NROW, NCOL = 197, 135

LON_MIN = -84.855
LON_MAX = -83.585
LAT_MIN =  39.185
LAT_MAX =  40.710

DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW

INACTIVE_COLOR = "#aec6d6"   # light blue shown for inactive MODFLOW cells

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def load_ibound():
    vals = []
    with open(os.path.join(BASE, "modflow_GMRW.bas")) as f:
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


def load_hk():
    vals = []
    capturing = False
    with open(os.path.join(BASE, "modflow_GMRW.upw")) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "INTERNAL" in s.upper() and "HK" in s.upper().replace("HDRY", ""):
                capturing = True
                continue
            if capturing:
                if re.match(r'(INTERNAL|EXTERNAL|CONSTANT)\b', s, re.I) and len(vals) > 0:
                    break
                for tok in s.split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
                if len(vals) >= NROW * NCOL:
                    break
    return np.array(vals[:NROW * NCOL], dtype=float).reshape(NROW, NCOL)


def load_multipliers():
    """Load prior100 ensemble HK multipliers (n=100, DSI prior)."""
    prior_csv = "D:/GMRW/FINAL_PROJECT_RESULTS/PRIOR_ENSEMBLE/swat_modflow_ies_prior100.0.par.csv"
    prior_df = pd.read_csv(prior_csv, index_col=0)
    return prior_df["hk_mult"].values


def load_wells():
    """Return (lon, lat) for each observation well from modflow.obs."""
    wells = []
    obs_file = os.path.join(BASE, "modflow.obs")
    if not os.path.exists(obs_file):
        return wells
    with open(obs_file) as f:
        f.readline()          # header comment
        n = int(f.readline().strip())
        for _ in range(n):
            parts = f.readline().split()
            row0 = int(parts[0]) - 1   # 0-based
            col0 = int(parts[1]) - 1
            lon = LON_MIN + (col0 + 0.5) * DLON
            lat = LAT_MAX - (row0 + 0.5) * DLAT
            wells.append((lon, lat))
    return wells


def load_stream_cells():
    """Return (lon, lat) centre-point of each river cell from modflow_GMRW.riv."""
    riv_file = os.path.join(BASE, "modflow_GMRW.riv")
    pts = []
    seen = set()
    with open(riv_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                lay = int(parts[0])
                row = int(parts[1]) - 1   # 0-based
                col = int(parts[2]) - 1
                if lay >= 1 and 0 <= row < NROW and 0 <= col < NCOL:
                    key = (row, col)
                    if key not in seen:
                        seen.add(key)
                        lon = LON_MIN + (col + 0.5) * DLON
                        lat = LAT_MAX - (row + 0.5) * DLAT
                        pts.append((lon, lat))
            except (ValueError, IndexError):
                pass
    return pts


def make_coord_grids():
    cols = np.arange(NCOL + 1)
    rows = np.arange(NROW + 1)
    lon_corners = LON_MIN + cols * DLON
    lat_corners = LAT_MAX - rows * DLAT
    LON, LAT = np.meshgrid(lon_corners, lat_corners)
    return LON, LAT


def watershed_boundary_segs(ibound):
    active = ibound != 0
    segs = []
    diff_h = np.diff(active.astype(int), axis=0)
    for r, c in zip(*np.where(diff_h != 0)):
        lat_y = LAT_MAX - (r + 1) * DLAT
        segs.append([(LON_MIN + c * DLON, lat_y),
                     (LON_MIN + (c + 1) * DLON, lat_y)])
    diff_v = np.diff(active.astype(int), axis=1)
    for r, c in zip(*np.where(diff_v != 0)):
        lon_x = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x, LAT_MAX - r * DLAT),
                     (lon_x, LAT_MAX - (r + 1) * DLAT)])
    for c in range(NCOL):
        if active[0, c]:
            segs.append([(LON_MIN + c * DLON, LAT_MAX),
                         (LON_MIN + (c+1) * DLON, LAT_MAX)])
        if active[NROW-1, c]:
            segs.append([(LON_MIN + c * DLON, LAT_MIN),
                         (LON_MIN + (c+1) * DLON, LAT_MIN)])
    for r in range(NROW):
        if active[r, 0]:
            segs.append([(LON_MIN, LAT_MAX - r * DLAT),
                         (LON_MIN, LAT_MAX - (r+1) * DLAT)])
        if active[r, NCOL-1]:
            segs.append([(LON_MAX, LAT_MAX - r * DLAT),
                         (LON_MAX, LAT_MAX - (r+1) * DLAT)])
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: draw one HK map panel
# ─────────────────────────────────────────────────────────────────────────────
def draw_hk_panel(ax, hk_data, ibound, LON_G, LAT_G, bdy_segs,
                  title_label, cmap, norm,
                  show_ylabel=True):
    """Draw a single HK map panel. Inactive cells are transparent (white bg)."""
    active = (ibound != 0) & (hk_data > 0)

    ax.set_facecolor("white")

    # HK pcolormesh (active cells only — inactive remain white)
    hk_disp = np.ma.masked_where(~active, hk_data)
    pcm = ax.pcolormesh(LON_G, LAT_G, hk_disp,
                        cmap=cmap, norm=norm,
                        shading="flat", linewidth=0,
                        edgecolors="none", zorder=2)

    # Watershed boundary
    lc = LineCollection(bdy_segs, colors="black", linewidths=0.8, zorder=6)
    ax.add_collection(lc)

    # Stats box (top-right)
    hk_active = hk_data[active]
    stats_txt = (f"Mean: {hk_active.mean():.1f} m/d\n"
                 f"Median: {np.median(hk_active):.1f} m/d\n"
                 f"Range: {hk_active.min():.1f}\u2013{hk_active.max():.0f} m/d")
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="#888888",
                      alpha=0.85, pad=3, boxstyle="round,pad=0.3"))

    # Tighten to active-cell bounding box
    act_r, act_c = np.where(ibound != 0)
    pad = 0.005
    x0 = LON_MIN + act_c.min() * DLON - pad
    x1 = LON_MIN + (act_c.max() + 1) * DLON + pad
    y0 = LAT_MAX - (act_r.max() + 1) * DLAT - pad
    y1 = LAT_MAX - act_r.min() * DLAT + pad
    # Set aspect first, then override limits so no extra whitespace is added
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Longitude (\u00b0)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Latitude (\u00b0)", fontsize=8)

    lon_ticks = np.arange(-84.8, -83.5, 0.4)
    lat_ticks = np.arange(39.2, 40.8, 0.4)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([f"{v:.1f}\u00b0" for v in lon_ticks], fontsize=7)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([f"{v:.1f}\u00b0" for v in lat_ticks], fontsize=7)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_title(title_label, fontsize=9.5, fontweight="bold", loc="left", pad=4)

    return pcm


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_fig03(ibound, hk_base, prior_mult):
    POST_MULT        = 1.00
    PRIOR_MAX_MULT   = float(prior_mult.max())   # 2.0 — upper bound of prior range

    hk_prior = hk_base * PRIOR_MAX_MULT
    hk_post  = hk_base * POST_MULT

    active_pos = (ibound != 0) & (hk_base > 0)
    print(f"  Prior MAX multiplier  : {PRIOR_MAX_MULT:.3f}")
    print(f"  Posterior multiplier  : {POST_MULT:.2f}")
    print(f"  Prior HK range (active>0): "
          f"{hk_prior[active_pos].min():.1f} \u2013 {hk_prior[active_pos].max():.0f} m/d")
    print(f"  Post  HK range (active>0): "
          f"{hk_post[active_pos].min():.1f} \u2013 {hk_post[active_pos].max():.0f} m/d")

    # Separate colormaps with separate ranges
    vmax_prior = round(np.percentile(hk_prior[active_pos], 98) / 10) * 10
    vmax_post  = round(np.percentile(hk_post[active_pos],  98) / 10) * 10

    cmap_prior = plt.cm.YlGnBu
    norm_prior = mcolors.Normalize(vmin=0, vmax=vmax_prior)

    cmap_post = plt.cm.YlOrRd
    norm_post = mcolors.Normalize(vmin=0, vmax=vmax_post)

    LON_G, LAT_G = make_coord_grids()
    bdy_segs     = watershed_boundary_segs(ibound)

    # ── Figure layout ──────────────────────────────────────────────────────
    # Row 0: two map panels  (A) and (B)  with own vertical colorbars
    # Row 1: multiplier distribution  (C)
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 0.52],
        hspace=0.50, wspace=0.18,
        left=0.05, right=0.97,
        top=0.93, bottom=0.07
    )
    ax_prior = fig.add_subplot(gs[0, 0])
    ax_post  = fig.add_subplot(gs[0, 1])
    ax_dist  = fig.add_subplot(gs[1, :])

    # ── (A) Prior ensemble mean HK ─────────────────────────────────────────
    pcm_prior = draw_hk_panel(ax_prior, hk_prior, ibound, LON_G, LAT_G, bdy_segs,
                  "(A)  Before Calibration \u2014 Prior Ensemble Mean HK",
                  cmap_prior, norm_prior,
                  show_ylabel=True)
    cbar_a = fig.colorbar(pcm_prior, ax=ax_prior, fraction=0.046, pad=0.03, shrink=0.92)
    cbar_a.set_label("HK (m/day)", fontsize=8)
    cbar_a.ax.tick_params(labelsize=7)
    ax_prior.text(
        0.5, -0.11,
        f"hk_mult range: {prior_mult.min():.2f}\u2013{prior_mult.max():.2f}  |  "
        f"Map shown at max multiplier = {PRIOR_MAX_MULT:.2f}  |  "
        f"n = {len(prior_mult)} realizations",
        transform=ax_prior.transAxes, ha="center", fontsize=7.5,
        style="italic", color="#444444"
    )

    # ── (B) Posterior HK (hk_mult = 1.00, all converged) ──────────────────
    pcm_post = draw_hk_panel(ax_post, hk_post, ibound, LON_G, LAT_G, bdy_segs,
                  "(B)  After Calibration \u2014 Posterior HK",
                  cmap_post, norm_post,
                  show_ylabel=False)
    cbar_b = fig.colorbar(pcm_post, ax=ax_post, fraction=0.046, pad=0.03, shrink=0.92)
    cbar_b.set_label("HK (m/day)", fontsize=8)
    cbar_b.ax.tick_params(labelsize=7)
    ax_post.text(
        0.5, -0.11,
        f"Posterior hk_mult: {POST_MULT:.2f}  (all realizations converged)",
        transform=ax_post.transAxes, ha="center", fontsize=7.5,
        style="italic", color="#444444"
    )

    # ── (C) Distribution of hk_mult ────────────────────────────────────────
    prior_p5, prior_p95 = np.percentile(prior_mult, [5, 95])
    prior_mean_m        = float(np.mean(prior_mult))

    bins = np.arange(0.60, 2.15, 0.08)
    ax_dist.hist(prior_mult, bins=bins, density=True, alpha=0.50,
                 color="#6baed6", edgecolor="white", linewidth=0.5,
                 label=f"Prior ensemble (n\u202f=\u202f{len(prior_mult)})",
                 zorder=3)

    x_kde = np.linspace(0.50, 2.20, 400)
    kde   = gaussian_kde(prior_mult)
    ax_dist.plot(x_kde, kde(x_kde), color="#2171b5", lw=2.0,
                 label="KDE", zorder=5)

    # P5\u2013P95 shaded band
    mask_p = (x_kde >= prior_p5) & (x_kde <= prior_p95)
    ax_dist.fill_between(x_kde[mask_p], 0, kde(x_kde[mask_p]),
                         alpha=0.20, color="#2171b5",
                         label=f"Prior P5\u2013P95 [{prior_p5:.2f}\u2013{prior_p95:.2f}]",
                         zorder=4)

    # Vertical lines
    ax_dist.axvline(prior_mean_m, color="#d4a017", ls="--", lw=2.0,
                    label=f"Prior mean = {prior_mean_m:.2f}", zorder=6)
    ax_dist.axvline(POST_MULT, color="red", ls="-", lw=2.0,
                    label=f"Posterior = {POST_MULT:.2f}", zorder=7)

    ax_dist.set_xlabel("hk_mult  (HK scaling factor)", fontsize=10)
    ax_dist.set_ylabel("Density", fontsize=10)
    ax_dist.set_title(
        "(C)  Distribution of hk_mult  (Calibration parameter)",
        fontsize=10, fontweight="bold", loc="left"
    )
    ax_dist.set_xlim(0.60, 2.10)
    ax_dist.tick_params(labelsize=8.5)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.legend(fontsize=8.5, framealpha=0.9,
                   edgecolor="#888888", loc="upper right")

    # Main title
    fig.suptitle(
        "Aquifer Hydraulic Conductivity \u2014 Before and After Calibration",
        fontsize=13, fontweight="bold", y=0.985
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading IBOUND …")
    ibound = load_ibound()
    print(f"  Active cells: {(ibound != 0).sum()}")

    print("Loading base HK …")
    hk = load_hk()
    print(f"  Base HK range (active): "
          f"{hk[ibound!=0].min():.1f} \u2013 {hk[ibound!=0].max():.1f} m/day")

    print("Loading prior multipliers \u2026")
    prior_mult = load_multipliers()
    print(f"  n={len(prior_mult)}, range=[{prior_mult.min():.3f}, {prior_mult.max():.3f}], "
          f"mean={prior_mult.mean():.3f}")

    print("Generating figure \u2026")
    fig = make_fig03(ibound, hk, prior_mult)

    for ext in ("png", "pdf"):
        path = os.path.join(OUT, f"fig03_hk_map.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\u2713 Saved: {path}")
    plt.close(fig)
    print("Done.")
