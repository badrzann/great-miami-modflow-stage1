"""
Figure – 4-panel publication-quality maps for the GMRW manuscript.
Final version using fine-resolution stochastic HRU-fraction data.

(a) Study area: MODFLOW grid, river cells, monitoring wells, flow gauge
(b) Land use: grouped SWAT categories
(c) Soil hydrologic group
(d) DEM / Topography with river network
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

# ── Configuration ─────────────────────────────────────────────────────────────
NROW, NCOL = 197, 135
LON_MIN, LAT_MIN = -84.855, 39.185
DX, DY = 0.01, 0.01

SWATMF_DIR = Path(r"D:\GMRW\finalresult\swatmf_run")
MAP_DIR    = Path(r"D:\GMRW\mapping")
OUT_DIR    = Path(r"D:\GMRW\finalresult\fig")

GAUGE_LON, GAUGE_LAT = -84.2851, 39.6368

# ── Coordinate helpers ───────────────────────────────────────────────────────
lon_centres = LON_MIN + (np.arange(NCOL) + 0.5) * DX
lat_centres = LAT_MIN + (NROW - np.arange(NROW) - 0.5) * DY
lon_grid, lat_grid = np.meshgrid(lon_centres, lat_centres)
lon_edges = np.linspace(LON_MIN, LON_MIN + NCOL * DX, NCOL + 1)
lat_edges = np.linspace(LAT_MIN, LAT_MIN + NROW * DY, NROW + 1)

def rc2lonlat(row, col):
    return LON_MIN + (col + 0.5) * DX, LAT_MIN + (NROW - row - 0.5) * DY

# ── Load data ────────────────────────────────────────────────────────────────
ibound        = np.load(MAP_DIR / "ibound.npy")
luse_grid     = np.load(MAP_DIR / "luse_fine_grid.npy")      # fine stochastic
soilgrp_grid  = np.load(MAP_DIR / "soilgrp_fine_grid.npy")   # fine stochastic
top_grid      = np.load(MAP_DIR / "top_grid.npy")
active_mask   = ibound > 0

# IBOUND from BAS
def load_ibound_from_bas():
    with open(SWATMF_DIR / "modflow_GMRW.bas") as fh:
        lines = fh.readlines()
    start = 0
    for i, line in enumerate(lines):
        if "IBOUND" in line.upper():
            start = i + 1; break
    vals = []
    for line in lines[start:]:
        parts = [x for x in line.split() if x.lstrip('-').isdigit()]
        if not parts: continue
        vals.extend(int(x) for x in parts)
        if len(vals) >= NROW * NCOL: break
    return np.array(vals[:NROW*NCOL]).reshape((NROW, NCOL))

ibound_bas = load_ibound_from_bas()

# River cells
def load_river_cells():
    cells = []
    with open(SWATMF_DIR / "modflow_GMRW.riv") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                cells.append((int(parts[1])-1, int(parts[2])-1))
    return cells

riv_cells = load_river_cells()
riv_lons = np.array([rc2lonlat(r,c)[0] for r,c in riv_cells])
riv_lats = np.array([rc2lonlat(r,c)[1] for r,c in riv_cells])

# Wells
wells = pd.read_csv(SWATMF_DIR / "well_name_mapping.csv")

# ── Active area extent ───────────────────────────────────────────────────────
ar, ac = np.where(active_mask)
pad = 3
x_lo = LON_MIN + max(0, ac.min()-pad) * DX
x_hi = LON_MIN + min(NCOL, ac.max()+pad+1) * DX
y_lo = LAT_MIN + (NROW - min(NROW, ar.max()+pad+1)) * DY
y_hi = LAT_MIN + (NROW - max(0, ar.min()-pad)) * DY

# ── Category setup ───────────────────────────────────────────────────────────
group_names = ['Agriculture', 'Forest', 'Urban', 'Water', 'Wetland', 'Rangeland']

luse_colors = {
    'Agriculture': '#C8B464',
    'Forest':      '#2D6A2D',
    'Urban':       '#A8A8A8',
    'Water':       '#3A7ABF',
    'Wetland':     '#7ECFC0',
    'Rangeland':   '#D4A843',
}

sg_colors = {1: '#C89664', 2: '#A0A0A0'}
sg_labels = {1: 'C', 2: 'D'}

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 16), dpi=300)

# Create axes with specific positions for better control
# [left, bottom, width, height]
w, h = 0.42, 0.42
gap_x, gap_y = 0.06, 0.04
x0, y1 = 0.06, 0.96

ax_a = fig.add_axes([x0,           y1 - h,        w, h])
ax_b = fig.add_axes([x0 + w + gap_x, y1 - h,      w, h])
ax_c = fig.add_axes([x0,           y1 - 2*h - gap_y, w, h])
ax_d = fig.add_axes([x0 + w + gap_x, y1 - 2*h - gap_y, w, h])

all_axes = [ax_a, ax_b, ax_c, ax_d]

# ── Common helpers ───────────────────────────────────────────────────────────
def add_boundary(ax, lw=1.5):
    ax.contour(lon_grid, lat_grid, ibound_bas.astype(float),
               levels=[0.5], colors="black", linewidths=lw, zorder=10)

def set_extent(ax):
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7.5, direction='in', length=3)

def add_scalebar(ax, loc='lower left'):
    ax.add_artist(ScaleBar(
        dx=111_320, units="m", location=loc,
        length_fraction=0.12, font_properties={"size": 7},
        box_alpha=0.85, sep=2, pad=0.6, border_pad=0.4,
        scale_loc='bottom', label_loc='top',
    ))

def add_north_arrow(ax, x=0.04, y=0.97):
    # Filled arrow
    ax.annotate(
        "", xy=(x, y - 0.01), xycoords="axes fraction",
        xytext=(x, y - 0.07), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black",
                        mutation_scale=12),
        zorder=20,
    )
    ax.text(x, y, "N", transform=ax.transAxes,
            fontsize=10, fontweight="bold", ha="center", va="bottom",
            zorder=20)

def add_label(ax, label):
    ax.text(0.06, 0.95, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left',
            zorder=25)

# ═════════════════════════════════════════════════════════════════════════════
# (a) Study area
# ═════════════════════════════════════════════════════════════════════════════
active_fill = np.where(active_mask, 1.0, np.nan)
ax_a.pcolormesh(lon_edges, lat_edges, active_fill[::-1],
                cmap=ListedColormap(["#ECECEC"]), edgecolors="none", zorder=1)

# Thin grid lines
for r in range(max(0,ar.min()-2), min(NROW,ar.max()+3)):
    y = LAT_MIN + (NROW - r) * DY
    ax_a.plot([x_lo, x_hi], [y, y], color='#D8D8D8', linewidth=0.05, zorder=1)
for c in range(max(0,ac.min()-2), min(NCOL,ac.max()+3)):
    x = LON_MIN + c * DX
    ax_a.plot([x, x], [y_lo, y_hi], color='#D8D8D8', linewidth=0.05, zorder=1)

add_boundary(ax_a)

# River cells
ax_a.scatter(riv_lons, riv_lats, s=2.5, c="#4A90D9", marker="s",
             linewidths=0, zorder=4, label="River cell")

# Monitoring wells
ax_a.scatter(wells["lon"], wells["lat"], s=22, c="#2D5E2D",
             marker="s", linewidths=0.4, edgecolors='black',
             zorder=7, label=f"Monitoring well (n={len(wells)})")

# Flow gauge
ax_a.scatter(GAUGE_LON, GAUGE_LAT, s=100, c="red", marker="*",
             edgecolors="black", linewidths=0.5, zorder=8,
             label="Flow gauge")

set_extent(ax_a)
add_scalebar(ax_a)
add_north_arrow(ax_a)
add_label(ax_a, '(a)')

leg_a = ax_a.legend(
    loc="lower right", fontsize=7, framealpha=0.92,
    edgecolor="#999", fancybox=False, handletextpad=0.5,
    borderpad=0.5, labelspacing=0.4,
)
leg_a.get_frame().set_linewidth(0.5)

# ═════════════════════════════════════════════════════════════════════════════
# (b) Land use
# ═════════════════════════════════════════════════════════════════════════════
cmap_colors = ['white'] + [luse_colors[g] for g in group_names]
luse_cmap = ListedColormap(cmap_colors)
luse_bounds = np.arange(-0.5, len(group_names) + 1.5, 1)
luse_norm = BoundaryNorm(luse_bounds, luse_cmap.N)

plot_lu = np.ma.masked_where(luse_grid == 0, luse_grid)
ax_b.pcolormesh(lon_edges, lat_edges, plot_lu[::-1],
                cmap=luse_cmap, norm=luse_norm, edgecolors="none", zorder=2)

add_boundary(ax_b)

# River overlay as thin blue line for context
ax_b.scatter(riv_lons, riv_lats, s=0.5, c="#6CB4EE", marker="s",
             linewidths=0, zorder=3, alpha=0.5)

set_extent(ax_b)
add_scalebar(ax_b)
add_north_arrow(ax_b)
add_label(ax_b, '(b)')

# Legend (only present groups)
present_lu = sorted(set(luse_grid[luse_grid > 0]))
lu_patches = [mpatches.Patch(facecolor=luse_colors[group_names[g-1]],
                             edgecolor='black', linewidth=0.4,
                             label=group_names[g-1])
              for g in present_lu]
leg_b = ax_b.legend(
    handles=lu_patches, title="Land use", title_fontsize=8,
    loc="lower right", fontsize=7, framealpha=0.92,
    edgecolor="#999", fancybox=False, handletextpad=0.5,
    borderpad=0.5, labelspacing=0.3,
)
leg_b.get_frame().set_linewidth(0.5)

# ═════════════════════════════════════════════════════════════════════════════
# (c) Soil hydrologic group
# ═════════════════════════════════════════════════════════════════════════════
sg_cmap = ListedColormap(['white', sg_colors[1], sg_colors[2]])
sg_bounds_arr = [-0.5, 0.5, 1.5, 2.5]
sg_norm = BoundaryNorm(sg_bounds_arr, sg_cmap.N)

plot_sg = np.ma.masked_where(soilgrp_grid == 0, soilgrp_grid)
ax_c.pcolormesh(lon_edges, lat_edges, plot_sg[::-1],
                cmap=sg_cmap, norm=sg_norm, edgecolors="none", zorder=2)

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
    loc="lower right", fontsize=7, framealpha=0.92,
    edgecolor="#999", fancybox=False, handletextpad=0.5,
    borderpad=0.5, labelspacing=0.3,
)
leg_c.get_frame().set_linewidth(0.5)

# ═════════════════════════════════════════════════════════════════════════════
# (d) DEM / Topography
# ═════════════════════════════════════════════════════════════════════════════
top_masked = np.ma.masked_where(~active_mask, top_grid)

im_d = ax_d.pcolormesh(lon_edges, lat_edges, top_masked[::-1],
                        cmap='terrain', edgecolors="none", zorder=2,
                        vmin=140, vmax=460)

add_boundary(ax_d)

# River on DEM
ax_d.scatter(riv_lons, riv_lats, s=1, c="#1A5FB4", marker="s",
             linewidths=0, zorder=4, label="River cell")

set_extent(ax_d)
add_scalebar(ax_d)
add_north_arrow(ax_d)
add_label(ax_d, '(d)')

# Colorbar positioned inside the panel
pos_d = ax_d.get_position()
cax = fig.add_axes([pos_d.x1 + 0.008, pos_d.y0 + pos_d.height*0.15,
                    0.012, pos_d.height * 0.6])
cbar = fig.colorbar(im_d, cax=cax)
cbar.set_label("Elevation (m)", fontsize=8, labelpad=4)
cbar.ax.tick_params(labelsize=6.5)

leg_d = ax_d.legend(
    loc="lower right", fontsize=7, framealpha=0.92,
    edgecolor="#999", fancybox=False, handletextpad=0.5, borderpad=0.5,
)
leg_d.get_frame().set_linewidth(0.5)

# ═════════════════════════════════════════════════════════════════════════════
# Remove redundant axis labels (only outer edges)
# ═════════════════════════════════════════════════════════════════════════════
for ax in [ax_a, ax_b]:
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelbottom=False)

for ax in [ax_b, ax_d]:
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelleft=False)

ax_c.set_xlabel('Longitude (°W)', fontsize=9)
ax_d.set_xlabel('Longitude (°W)', fontsize=9)
ax_a.set_ylabel('Latitude (°N)', fontsize=9)
ax_c.set_ylabel('Latitude (°N)', fontsize=9)

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
out_png = OUT_DIR / "fig_4panel_maps.png"
out_pdf = OUT_DIR / "fig_4panel_maps.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved:\n  {out_png}\n  {out_pdf}")
