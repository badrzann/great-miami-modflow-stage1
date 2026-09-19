"""
make_hru_map.py
===============
HRU/DHRU map matching the reference image:
  - Black background
  - Orange HRU polygons with dark-orange edges (creating the HRU grid pattern)
  - Watershed boundary (white, bold)
  - Left locator panel
  - lat/lon axes, legend below
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
HRU_SHP = r"D:\GMRW_intraction\linkage\swat model\GMRW\Watershed\Shapes\hru1.shp"
OUT     = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710

COL_BG      = "white"      # white background
COL_HRU     = "#C8E6B0"    # light mint-green HRU fill (matches reference)
COL_EDGE    = "#111111"    # near-black edges between HRUs
COL_SUB_BDY = "#111111"    # subbasin boundary lines
COL_BDY     = "black"      # watershed outer boundary


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOCATOR PANEL
# ─────────────────────────────────────────────────────────────────────────────
def draw_locator(ax):
    us_lon = [-124.7, -124.7, -67.0, -67.0, -124.7]
    us_lat = [  24.5,   49.0,  49.0,  24.5,   24.5]
    ax.fill(us_lon, us_lat, color="#c8efb8", alpha=0.9, zorder=1)
    ax.plot(us_lon, us_lat, "k-", lw=0.4, zorder=2)
    states = [
        ([-80.52,-80.52,-81.6,-82.5,-83.1,-83.47,-84.82,-84.82,-83.75,-82.3,-81.5,-80.52],
         [41.98, 40.0,  38.6, 38.4, 38.6,  39.1,  39.1,  41.7,  41.98, 42.32,42.32,41.98],
         "#b7dab3"),
    ]
    for lons, lats, col in states:
        ax.fill(lons, lats, color=col, alpha=1.0, zorder=2)
        ax.plot(lons + [lons[0]], lats + [lats[0]], "k-", lw=0.5, zorder=3)
    rect = mpatches.Rectangle((LON_MIN, LAT_MIN),
                               LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
                               lw=1.5, edgecolor="red",
                               facecolor="#ff8888", alpha=0.55, zorder=5)
    ax.add_patch(rect)
    ax.set_xlim(-125, -65)
    ax.set_ylim(24, 50)
    ax.set_facecolor("#dbeeff")
    ax.set_xlabel("Longitude (°)", fontsize=8)
    ax.set_ylabel("Latitude (°)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(a)", fontsize=10, fontweight="bold", loc="left")
    sb_lon0, sb_lat0 = -116, 25.5
    ax.plot([sb_lon0, sb_lon0 + 3], [sb_lat0, sb_lat0], "k-", lw=2, zorder=6)
    for x in [sb_lon0, sb_lon0+1, sb_lon0+2, sb_lon0+3]:
        ax.plot([x], [sb_lat0], "k|", lw=1.5, ms=4, zorder=6)
    ax.text(sb_lon0 + 1.5, sb_lat0 - 0.8, "0  345  690  1380 km",
            ha="center", fontsize=5.5, zorder=6)
    ax.annotate("N", xy=(-67.5, 47.2), fontsize=8, ha="center", fontweight="bold", zorder=6)
    ax.annotate("", xy=(-67.5, 48.5), xytext=(-67.5, 47.5),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2), zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MAIN FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_hru_figure(hru_wgs, subs_wgs):
    fig, ax_hru = plt.subplots(figsize=(9, 11))
    fig.subplots_adjust(bottom=0.14)
    fig.patch.set_facecolor("white")
    ax_hru.set_facecolor(COL_BG)

    # Plot HRU polygons — light green fill, dark edge
    hru_wgs.plot(
        ax=ax_hru,
        facecolor=COL_HRU,
        edgecolor=COL_EDGE,
        linewidth=0.2,
        zorder=2,
    )

    # Subbasin boundaries (thicker dark lines = visible grid pattern)
    subs_wgs.boundary.plot(
        ax=ax_hru, color=COL_SUB_BDY, linewidth=0.8, zorder=3
    )

    # Watershed outline
    basin = hru_wgs.dissolve()
    basin.boundary.plot(ax=ax_hru, color=COL_BDY, linewidth=1.8, zorder=5)

    # Lat/lon axes — use actual data bounds + padding to avoid clipping
    bounds = hru_wgs.total_bounds  # [xmin, ymin, xmax, ymax]
    pad_x = (bounds[2] - bounds[0]) * 0.02
    pad_y = (bounds[3] - bounds[1]) * 0.02
    xmin, xmax = bounds[0] - pad_x, bounds[2] + pad_x
    ymin, ymax = bounds[1] - pad_y, bounds[3] + pad_y
    ax_hru.set_xlim(xmin, xmax)
    ax_hru.set_ylim(ymin, ymax)
    ax_hru.set_aspect("equal")
    import math
    lon_ticks = np.arange(math.ceil(xmin * 5) / 5, xmax, 0.2)
    lat_ticks = np.arange(math.ceil(ymin * 5) / 5, ymax, 0.2)
    ax_hru.set_xticks(lon_ticks)
    ax_hru.set_xticklabels([f"{v:.1f}°" for v in lon_ticks], fontsize=9)
    ax_hru.set_yticks(lat_ticks)
    ax_hru.set_yticklabels([f"{v:.1f}°" for v in lat_ticks], fontsize=9)
    ax_hru.tick_params(direction="in", top=True, right=True)
    ax_hru.set_xlabel("Longitude (°)", fontsize=10)
    ax_hru.set_ylabel("Latitude (°)",  fontsize=10)

    # Legend below map
    legend_handles = [
        mpatches.Patch(facecolor=COL_HRU, edgecolor="#333", lw=0.5,
                       label=f"SWAT HRU  (n = {len(hru_wgs):,})"),
        mpatches.Patch(facecolor="none", edgecolor="#333", lw=0.8,
                       label=f"Subbasin boundary  (n = {len(subs_wgs):,})"),
        Line2D([0], [0], color=COL_BDY, lw=1.8,
               label="Watershed boundary"),
    ]
    ax_hru.legend(handles=legend_handles,
                  loc="upper center", bbox_to_anchor=(0.5, -0.09),
                  ncol=3, fontsize=9, framealpha=0.95,
                  borderpad=0.6, edgecolor="#888888",
                  handlelength=1.4, handleheight=1.0)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading HRU shapefile …")
    hru = gpd.read_file(HRU_SHP)
    print(f"  HRUs: {len(hru):,}  |  CRS: {hru.crs}")

    print("Reprojecting to WGS84 …")
    hru_wgs  = hru.to_crs("EPSG:4326")

    print("Loading subbasin shapefile …")
    subs = gpd.read_file(r"D:\GMRW_intraction\linkage\swat model\GMRW\Watershed\Shapes\subs1.shp")
    subs_wgs = subs.to_crs("EPSG:4326")
    print(f"  Subbasins: {len(subs_wgs):,}")

    print("Generating figure …")
    fig = make_hru_figure(hru_wgs, subs_wgs)

    out_png = os.path.join(OUT, "fig_hru_map.png")
    out_pdf = os.path.join(OUT, "fig_hru_map.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,          bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved:\n   {out_png}\n   {out_pdf}")
