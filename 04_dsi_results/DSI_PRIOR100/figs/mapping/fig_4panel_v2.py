"""
Figure – 4-panel publication-quality maps for the GMRW manuscript.

(a) Study area: MODFLOW grid, river cells, monitoring wells, flow gauge
(b) Land use: grouped SWAT categories (stochastic HRU-fraction sampling)
(c) Soil hydrologic group: C and D
(d) DEM / Topography with river network

All geometry derived from the MODFLOW grid (geographic coordinates).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
NROW, NCOL = 197, 135
LON_MIN, LAT_MIN = -84.855, 39.185
DX, DY = 0.01, 0.01

SWATMF_DIR = Path(r"D:\GMRW\finalresult\swatmf_run")
MAP_DIR    = Path(r"D:\GMRW\mapping")
OUT_DIR    = Path(r"D:\GMRW\finalresult\fig")

GAUGE_LON, GAUGE_LAT = -84.2851, 39.6368

# ── Geographic helpers ────────────────────────────────────────────────────────
lon_centres = LON_MIN + (np.arange(NCOL) + 0.5) * DX
lat_centres = LAT_MIN + (NROW - np.arange(NROW) - 0.5) * DY
lon_grid, lat_grid = np.meshgrid(lon_centres, lat_centres)
lon_edges = np.linspace(LON_MIN, LON_MIN + NCOL * DX, NCOL + 1)
lat_edges = np.linspace(LAT_MIN, LAT_MIN + NROW * DY, NROW + 1)

def rc2lonlat(row, col):
    return LON_MIN + (col + 0.5) * DX, LAT_MIN + (NROW - row - 0.5) * DY

# ── Load data ────────────────────────────────────────────────────────────────
ibound        = np.load(MAP_DIR / "ibound.npy")
luse_grid     = np.load(MAP_DIR / "luse_fine_grid.npy")
soilgrp_grid  = np.load(MAP_DIR / "soilgrp_fine_grid.npy")
top_grid      = np.load(MAP_DIR / "top_grid.npy")

active_mask = ibound > 0

# ── Load IBOUND from BAS ─────────────────────────────────────────────────────
def load_ibound_from_bas():
    with open(SWATMF_DIR / "modflow_GMRW.bas") as fh:
        lines = fh.readlines()
    start = 0
    for i, line in enumerate(lines):
        if "IBOUND" in line.upper():
            start = i + 1
            break
    vals = []
    for line in lines[start:]:
        parts = [x for x in line.split() if x.lstrip('-').isdigit()]
        if not parts:
            continue
        vals.extend(int(x) for x in parts)
        if len(vals) >= NROW * NCOL:
            break
    return np.array(vals[:NROW*NCOL]).reshape((NROW, NCOL))

ibound_bas = load_ibound_from_bas()

# ── Load river cells ─────────────────────────────────────────────────────────
def load_river_cells():
    cells = []
    with open(SWATMF_DIR / "modflow_GMRW.riv") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                r, c = int(parts[1]) - 1, int(parts[2]) - 1
                cells.append((r, c))
    return cells

riv_cells = load_river_cells()
riv_lons = np.array([rc2lonlat(r, c)[0] for r, c in riv_cells])
riv_lats = np.array([rc2lonlat(r, c)[1] for r, c in riv_cells])

# ── Load wells ───────────────────────────────────────────────────────────────
wells = pd.read_csv(SWATMF_DIR / "well_name_mapping.csv")

# ── Land use grouping ────────────────────────────────────────────────────────
group_names = ['Agriculture', 'Forest', 'Urban', 'Water', 'Wetland', 'Rangeland']
group_to_int = {g: i+1 for i, g in enumerate(group_names)}

# ── Crop extents to the active area with padding ─────────────────────────────
active_rows, active_cols = np.where(active_mask)
r_min, r_max = active_rows.min(), active_rows.max()
c_min, c_max = active_cols.min(), active_cols.max()

# Convert to geographic bounds with padding
pad_cells = 3
x_lo = LON_MIN + max(0, c_min - pad_cells) * DX
x_hi = LON_MIN + min(NCOL, c_max + pad_cells + 1) * DX
y_lo = LAT_MIN + (NROW - min(NROW, r_max + pad_cells + 1)) * DY
y_hi = LAT_MIN + (NROW - max(0, r_min - pad_cells)) * DY

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 15), dpi=300)
((ax_a, ax_b), (ax_c, ax_d)) = axes

# ── Common functions ─────────────────────────────────────────────────────────
def add_boundary(ax, lw=1.4):
    ax.contour(lon_grid, lat_grid, ibound_bas.astype(float),
               levels=[0.5], colors="black", linewidths=lw, zorder=10)

def set_extent(ax):
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7.5, direction='in', length=3)
    ax.set_xlabel('')
    ax.set_ylabel('')

def add_scalebar(ax, loc='lower left'):
    sb = ScaleBar(
        dx=111_320, units="m", location=loc,
        length_fraction=0.12, font_properties={"size": 7},
        box_alpha=0.85, sep=2, pad=0.5,
        scale_loc='bottom', label_loc='top',
    )
    ax.add_artist(sb)

def add_north_arrow(ax, x=0.04, y=0.96):
    # Arrow shaft
    ax.annotate(
        "", xy=(x, y - 0.005), xycoords="axes fraction",
        xytext=(x, y - 0.065), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"),
        zorder=20,
    )
    # "N" label
    ax.text(x, y + 0.005, "N", transform=ax.transAxes,
            fontsize=10, fontweight="bold", ha="center", va="bottom",
            zorder=20)

def add_label(ax, label, x=0.07, y=0.94):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.85),
            zorder=25)

# ═════════════════════════════════════════════════════════════════════════════
# (a) Study area – grid, rivers, wells, gauge
# ═════════════════════════════════════════════════════════════════════════════
# Background fill
active_fill = np.where(active_mask, 1.0, np.nan)
ax_a.pcolormesh(
    lon_edges, lat_edges, active_fill[::-1],
    cmap=ListedColormap(["#ececec"]), edgecolors="none", zorder=1,
)

# Thin MODFLOW grid lines (only inside active area)
for r in range(r_min, r_max + 2):
    y = LAT_MIN + (NROW - r) * DY
    ax_a.plot([x_lo, x_hi], [y, y], color='#d0d0d0', linewidth=0.06, zorder=1)
for c in range(c_min, c_max + 2):
    x = LON_MIN + c * DX
    ax_a.plot([x, x], [y_lo, y_hi], color='#d0d0d0', linewidth=0.06, zorder=1)

add_boundary(ax_a)

# River cells
ax_a.scatter(riv_lons, riv_lats, s=2, c="#4a90d9", marker="s",
             linewidths=0, zorder=4, label="River cell")

# Monitoring wells
ax_a.scatter(wells["lon"], wells["lat"], s=18, c="#2d5e2d",
             marker="s", linewidths=0.3, edgecolors='black',
             zorder=7, label="Monitoring well")

# Flow gauge
ax_a.scatter(GAUGE_LON, GAUGE_LAT, s=100, c="red", marker="*",
             edgecolors="black", linewidths=0.5, zorder=8,
             label="Flow gauge")

set_extent(ax_a)
add_boundary(ax_a)
add_scalebar(ax_a)
add_north_arrow(ax_a)
add_label(ax_a, '(a)')

leg_a = ax_a.legend(
    loc="lower right", fontsize=7.5, framealpha=0.92,
    edgecolor="#999999", fancybox=False,
    handletextpad=0.5, borderpad=0.5, labelspacing=0.4,
)
leg_a.get_frame().set_linewidth(0.5)

# ═════════════════════════════════════════════════════════════════════════════
# (b) Land use (fine-resolution, grouped)
# ═════════════════════════════════════════════════════════════════════════════
luse_colors = {
    'Agriculture': '#c8b464',   # golden/wheat
    'Forest':      '#2d6a2d',   # dark green
    'Urban':       '#b0b0b0',   # gray
    'Water':       '#3a7abf',   # blue
    'Wetland':     '#7ecfc0',   # light teal
    'Rangeland':   '#d4a843',   # tan
}

# Build colormap: index 0 = white (inactive), then 1-6 for groups
cmap_colors = ['white'] + [luse_colors[g] for g in group_names]
luse_cmap = ListedColormap(cmap_colors)
luse_bounds = np.arange(-0.5, len(group_names) + 1.5, 1)
luse_norm = BoundaryNorm(luse_bounds, luse_cmap.N)

plot_lu = np.ma.masked_where(luse_grid == 0, luse_grid)
ax_b.pcolormesh(
    lon_edges, lat_edges, plot_lu[::-1],
    cmap=luse_cmap, norm=luse_norm, edgecolors="none", zorder=2,
)

add_boundary(ax_b)
set_extent(ax_b)
add_scalebar(ax_b)
add_north_arrow(ax_b)
add_label(ax_b, '(b)')

# Legend (only groups that appear)
present_groups = sorted(set(luse_grid[luse_grid > 0]))
lu_patches = [mpatches.Patch(facecolor=luse_colors[group_names[g-1]],
                             edgecolor='black', linewidth=0.4,
                             label=group_names[g-1])
              for g in present_groups]
leg_b = ax_b.legend(
    handles=lu_patches, title="Land use", title_fontsize=8,
    loc="lower right", fontsize=7.5, framealpha=0.92,
    edgecolor="#999999", fancybox=False,
    handletextpad=0.5, borderpad=0.5, labelspacing=0.35,
)
leg_b.get_frame().set_linewidth(0.5)

# ═════════════════════════════════════════════════════════════════════════════
# (c) Soil hydrologic group
# ═════════════════════════════════════════════════════════════════════════════
sg_colors = {1: '#c89664', 2: '#a0a0a0'}   # C → orange-brown, D → gray
sg_labels = {1: 'C', 2: 'D'}

sg_cmap = ListedColormap(['white', sg_colors[1], sg_colors[2]])
sg_bounds = [-0.5, 0.5, 1.5, 2.5]
sg_norm = BoundaryNorm(sg_bounds, sg_cmap.N)

plot_sg = np.ma.masked_where(soilgrp_grid == 0, soilgrp_grid)
ax_c.pcolormesh(
    lon_edges, lat_edges, plot_sg[::-1],
    cmap=sg_cmap, norm=sg_norm, edgecolors="none", zorder=2,
)

add_boundary(ax_c)
set_extent(ax_c)
add_scalebar(ax_c)
add_north_arrow(ax_c)
add_label(ax_c, '(c)')

sg_patches = [mpatches.Patch(facecolor=sg_colors[i], edgecolor='black',
                              linewidth=0.4, label=sg_labels[i])
              for i in [1, 2]]
leg_c = ax_c.legend(
    handles=sg_patches, title="Soil group", title_fontsize=8,
    loc="lower right", fontsize=7.5, framealpha=0.92,
    edgecolor="#999999", fancybox=False,
    handletextpad=0.5, borderpad=0.5, labelspacing=0.35,
)
leg_c.get_frame().set_linewidth(0.5)

# ═════════════════════════════════════════════════════════════════════════════
# (d) DEM / Topography
# ═════════════════════════════════════════════════════════════════════════════
top_masked = np.ma.masked_where(~active_mask, top_grid)

im_d = ax_d.pcolormesh(
    lon_edges, lat_edges, top_masked[::-1],
    cmap='terrain', edgecolors="none", zorder=2,
    vmin=140, vmax=460,
)

add_boundary(ax_d)

# River cells on DEM
ax_d.scatter(riv_lons, riv_lats, s=1.2, c="#1a5fb4", marker="s",
             linewidths=0, zorder=4, label="River cell")

set_extent(ax_d)
add_scalebar(ax_d)
add_north_arrow(ax_d)
add_label(ax_d, '(d)')

# Colorbar inside the panel (right side, within axes)
from mpl_toolkits.axes_grid1 import make_axes_locatable
cax = fig.add_axes([
    ax_d.get_position().x1 - 0.025,  # right edge of panel
    ax_d.get_position().y0 + 0.03,
    0.012,
    ax_d.get_position().height * 0.5
])
cbar = fig.colorbar(im_d, cax=cax)
cbar.set_label("Elevation (m)", fontsize=7.5, labelpad=3)
cbar.ax.tick_params(labelsize=6.5)

leg_d = ax_d.legend(
    loc="lower right", fontsize=7.5, framealpha=0.92,
    edgecolor="#999999", fancybox=False,
    handletextpad=0.5, borderpad=0.5,
)
leg_d.get_frame().set_linewidth(0.5)

# ═════════════════════════════════════════════════════════════════════════════
# Layout & save
# ═════════════════════════════════════════════════════════════════════════════
fig.subplots_adjust(wspace=0.12, hspace=0.10, left=0.05, right=0.95,
                    top=0.97, bottom=0.03)

out_png = OUT_DIR / "fig_4panel_maps.png"
out_pdf = OUT_DIR / "fig_4panel_maps.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved:\n  {out_png}\n  {out_pdf}")
