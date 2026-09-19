"""
make_odnr_hk_single.py
======================
Single-panel figure of the original ODNR hydraulic-conductivity map
in exactly the same format as the panels in fig03_hk_map.png.

Output:
  D:/GMRW/finalresult/fig/fig03_odnr_hk.png   (300 DPI)
  D:/GMRW/finalresult/fig/fig03_odnr_hk.pdf
"""

import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib_scalebar.scalebar import ScaleBar

# ─────────────────────────────────────────────────────────────────────────────
# PATHS AND GRID CONSTANTS  (identical to make_fig03_hk.py)
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


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (identical to make_fig03_hk.py)
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


def make_coord_grids():
    lon_corners = LON_MIN + np.arange(NCOL + 1) * DLON
    lat_corners = LAT_MAX - np.arange(NROW + 1) * DLAT
    return np.meshgrid(lon_corners, lat_corners)


def watershed_boundary_segs(active):
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
                         (LON_MIN + (c + 1) * DLON, LAT_MAX)])
        if active[NROW - 1, c]:
            segs.append([(LON_MIN + c * DLON, LAT_MIN),
                         (LON_MIN + (c + 1) * DLON, LAT_MIN)])
    for r in range(NROW):
        if active[r, 0]:
            segs.append([(LON_MIN, LAT_MAX - r * DLAT),
                         (LON_MIN, LAT_MAX - (r + 1) * DLAT)])
        if active[r, NCOL - 1]:
            segs.append([(LON_MAX, LAT_MAX - r * DLAT),
                         (LON_MAX, LAT_MAX - (r + 1) * DLAT)])
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# NORTH ARROW  (same style as fig_4panel_fixed.py)
# ─────────────────────────────────────────────────────────────────────────────
def add_north_arrow(ax, x=0.055, y=0.965):
    ax.annotate('', xy=(x, y - 0.01), xycoords='axes fraction',
                xytext=(x, y - 0.068), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=2.0, color='black',
                                mutation_scale=14), zorder=22)
    ax.text(x, y + 0.003, 'N', transform=ax.transAxes,
            fontsize=11, fontweight='bold', ha='center', va='bottom', zorder=22)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading IBOUND …")
    ibound = load_ibound()
    print(f"  Active cells: {(ibound != 0).sum()}")

    print("Loading base HK …")
    hk = load_hk()
    active_pos = (ibound != 0) & (hk > 0)
    hk_active  = hk[active_pos]
    print(f"  ODNR HK range (active): {hk_active.min():.1f} – {hk_active.max():.1f} m/day")

    LON_G, LAT_G = make_coord_grids()
    bdy_segs     = watershed_boundary_segs(active_pos)

    # Colormap and normalization  — YlGnBu, same as panel (a) in fig03_hk_map
    vmax = round(np.percentile(hk_active, 98) / 10) * 10
    cmap = plt.cm.YlGnBu
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    # Compute tight axis limits from DATA cells (hk > 0)
    act_r, act_c = np.where(active_pos)
    pad = 0.005
    x0 = LON_MIN + act_c.min() * DLON - pad
    x1 = LON_MIN + (act_c.max() + 1) * DLON + pad
    y0 = LAT_MAX - (act_r.max() + 1) * DLAT - pad
    y1 = LAT_MAX - act_r.min() * DLAT + pad

    # Figure size: match data aspect ratio so aspect='equal' causes no letterboxing
    data_w = x1 - x0   # degrees lon
    data_h = y1 - y0   # degrees lat
    fig_h = 7.8
    fig_w = fig_h * (data_w / data_h) + 1.6   # +1.6" for colorbar + labels

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_facecolor("white")

    # HK pcolormesh — active data cells only
    hk_disp = np.ma.masked_where(~active_pos, hk)
    pcm = ax.pcolormesh(LON_G, LAT_G, hk_disp,
                        cmap=cmap, norm=norm,
                        shading="flat", linewidth=0, edgecolors="none", zorder=2)

    # Watershed boundary
    lc = LineCollection(bdy_segs, colors="black", linewidths=0.9, zorder=6)
    ax.add_collection(lc)

    # Stats box — placed below the colorbar, outside the map axes
    stats_txt = (f"Mean: {hk_active.mean():.1f} m/d    "
                 f"Median: {np.median(hk_active):.1f} m/d    "
                 f"Range: {hk_active.min():.1f}–{hk_active.max():.0f} m/d")
    ax.text(0.5, -0.07, stats_txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="#888888",
                      alpha=0.90, pad=4, boxstyle="round,pad=0.4"), zorder=20)

    # Axis limits tight to active-cell bounding box
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")

    # Axis labels and ticks
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    lon_ticks = np.arange(-84.8, -83.5, 0.4)
    lat_ticks = np.arange(39.2,   40.8,  0.4)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([f"{v:.1f}°" for v in lon_ticks], fontsize=7.5)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([f"{v:.1f}°" for v in lat_ticks], fontsize=7.5)
    ax.tick_params(direction="in", top=True, right=True)

    # Panel title  (same style as draw_hk_panel)
    ax.set_title("(a)  Original ODNR HK", fontsize=9.5, fontweight="bold",
                 loc="left", pad=4)

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03, shrink=0.92)
    cbar.set_label("HK (m/day)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # North arrow
    add_north_arrow(ax, x=0.055, y=0.965)

    # Scale bar  (matplotlib-scalebar; 1° ≈ 111 km, so 1 m_per_deg ≈ 111000)
    scalebar = ScaleBar(111000, "m", length_fraction=0.11,
                        location="lower left",
                        border_pad=0.5, sep=3,
                        frameon=True,
                        color="black", box_color="white", box_alpha=0.8,
                        font_properties={"size": 7.5})
    ax.add_artist(scalebar)

    # Figure title
    fig.suptitle("Spatial distribution of hydraulic conductivity — ODNR base map",
                 fontsize=11, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in ("png", "pdf"):
        path = os.path.join(OUT, f"fig03_odnr_hk.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Saved: {path}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
