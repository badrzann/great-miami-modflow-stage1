"""
Figure – 4-panel publication-quality maps for the GMRW manuscript.

(a) Study area: MODFLOW grid, river cells, monitoring wells, flow gauge
(b) Land use: grouped SWAT categories
(c) Soil hydrologic group: C and D
(d) DEM / Topography

All geometry derived from the MODFLOW grid (geographic coordinates).
No external shapefiles are modified or reprojected.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
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
    lon = LON_MIN + (col + 0.5) * DX
    lat = LAT_MIN + (NROW - row - 0.5) * DY
    return lon, lat

# ── Load data ────────────────────────────────────────────────────────────────
ibound       = np.load(MAP_DIR / "ibound.npy")
luse_grid    = np.load(MAP_DIR / "luse_int_grid.npy")
soilgrp_grid = np.load(MAP_DIR / "soilgrp_int_grid.npy")
top_grid     = np.load(MAP_DIR / "top_grid.npy")
sub_grid     = np.load(MAP_DIR / "sub_grid.npy")

with open(MAP_DIR / "grid_lookup.json") as f:
    lookup = json.load(f)

active_mask = ibound > 0

# ── Load IBOUND from BAS for boundary contour ───────────────────────────────
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

# ── Load river cells from .riv ───────────────────────────────────────────────
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

# ═════════════════════════════════════════════════════════════════════════════
# GROUP LAND USE into 5 categories (like reference image)
# ═════════════════════════════════════════════════════════════════════════════
# SWAT codes -> grouped categories
luse_grouping = {
    'AGRL': 'Agriculture',
    'PAST': 'Agriculture',
    'RNGB': 'Agriculture',
    'RNGE': 'Agriculture',
    'FRSD': 'Forest',
    'FRSE': 'Forest',
    'FRST': 'Forest',
    'UIDU': 'Urban',
    'URHD': 'Urban',
    'URLD': 'Urban',
    'URMD': 'Urban',
    'BARR': 'Urban',
    'WATR': 'Water',
    'WETL': 'Wetland',
    'WETN': 'Wetland',
}

# Build grouped land use grid
group_names = ['Agriculture', 'Forest', 'Urban', 'Water', 'Wetland']
group_to_int = {g: i+1 for i, g in enumerate(group_names)}

int_to_luse_raw = lookup['int_to_luse']
grouped_luse_grid = np.zeros_like(luse_grid)
for r in range(NROW):
    for c in range(NCOL):
        code = luse_grid[r, c]
        if code > 0:
            raw_name = int_to_luse_raw[str(code)]
            group = luse_grouping.get(raw_name, 'Unknown')
            if group in group_to_int:
                grouped_luse_grid[r, c] = group_to_int[group]

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=300)
((ax_a, ax_b), (ax_c, ax_d)) = axes

panel_labels = ['(a)', '(b)', '(c)', '(d)']

# ── Common plot functions ────────────────────────────────────────────────────
def add_boundary(ax):
    """Add watershed boundary contour."""
    ax.contour(
        lon_grid, lat_grid, ibound_bas.astype(float),
        levels=[0.5], colors="black", linewidths=1.2, zorder=5,
    )

def set_map_extent(ax):
    """Set common extent and labels."""
    pad = 0.03
    ax.set_xlim(LON_MIN + 0.1 - pad, LON_MIN + NCOL * DX - 0.05 + pad)
    ax.set_ylim(LAT_MIN + 0.1 - pad, LAT_MIN + NROW * DY - 0.05 + pad)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)

def add_scalebar(ax):
    """Add scale bar."""
    sb = ScaleBar(
        dx=111_320, units="m",
        location="lower left",
        length_fraction=0.15,
        font_properties={"size": 7},
        box_alpha=0.85,
        sep=2,
        pad=0.4,
    )
    ax.add_artist(sb)

def add_north_arrow(ax, x=0.05, y=0.95):
    """Add north arrow."""
    ax.annotate(
        "", xy=(x, y), xycoords="axes fraction",
        xytext=(x, y - 0.06), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
        zorder=20,
    )
    ax.annotate(
        "N", xy=(x, y + 0.01), xycoords="axes fraction",
        fontsize=9, fontweight="bold", ha="center", va="bottom", zorder=20,
    )

def add_panel_label(ax, label, x=0.05, y=0.93):
    """Add panel label like (a), (b), etc."""
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='none', alpha=0.85),
        zorder=25,
    )

# ═════════════════════════════════════════════════════════════════════════════
# PANEL (a): Study area – grid, rivers, wells, gauge
# ═════════════════════════════════════════════════════════════════════════════
# Background fill for active cells
active_fill = np.where(active_mask, 1.0, np.nan)
ax_a.pcolormesh(
    lon_edges, lat_edges, active_fill[::-1],
    cmap=ListedColormap(["#f0f0f0"]), edgecolors="none", zorder=1,
)

# MODFLOW grid lines (thin)
for r in range(NROW + 1):
    y = LAT_MIN + r * DY
    ax_a.axhline(y, color='#cccccc', linewidth=0.08, zorder=1)
for c in range(NCOL + 1):
    x = LON_MIN + c * DX
    ax_a.axvline(x, color='#cccccc', linewidth=0.08, zorder=1)

add_boundary(ax_a)

# River cells
ax_a.scatter(riv_lons, riv_lats, s=2.5, c="#4493c7", marker="s",
             linewidths=0, zorder=3, label="River cell")

# Observation wells
ax_a.scatter(wells["lon"], wells["lat"], s=20, c="#2d5e2d",
             marker="s", linewidths=0.3, edgecolors='black',
             zorder=6, label="Monitoring well")

# Flow gauge
ax_a.scatter(GAUGE_LON, GAUGE_LAT, s=100, c="red", marker="*",
             edgecolors="black", linewidths=0.5, zorder=7,
             label="Flow gauge")

# Legend
leg_a = ax_a.legend(
    loc="lower right", fontsize=7, framealpha=0.92,
    edgecolor="gray", fancybox=False, handletextpad=0.4,
    borderpad=0.4, labelspacing=0.35,
)

set_map_extent(ax_a)
add_scalebar(ax_a)
add_north_arrow(ax_a)
add_panel_label(ax_a, '(a)')

# ═════════════════════════════════════════════════════════════════════════════
# PANEL (b): Land use (grouped)
# ═════════════════════════════════════════════════════════════════════════════
# Colors matching reference image style
luse_colors = {
    'Agriculture': '#c8b464',   # golden/wheat
    'Forest':      '#2d6a2d',   # dark green
    'Urban':       '#b0b0b0',   # gray
    'Water':       '#3a7abf',   # blue
    'Wetland':     '#7ecfc0',   # light teal
}

luse_cmap_list = ['white'] + [luse_colors[g] for g in group_names]
luse_cmap = ListedColormap(luse_cmap_list)
luse_bounds = np.arange(-0.5, len(group_names) + 1.5, 1)
luse_norm = BoundaryNorm(luse_bounds, luse_cmap.N)

# Plot
plot_data = np.ma.masked_where(grouped_luse_grid == 0, grouped_luse_grid)
ax_b.pcolormesh(
    lon_edges, lat_edges, plot_data[::-1],
    cmap=luse_cmap, norm=luse_norm, edgecolors="none", zorder=2,
)

add_boundary(ax_b)

# Legend patches
luse_patches = [mpatches.Patch(facecolor=luse_colors[g], edgecolor='black',
                               linewidth=0.4, label=g)
                for g in group_names if group_to_int[g] in grouped_luse_grid]
leg_b = ax_b.legend(
    handles=luse_patches, title="Land use", title_fontsize=8,
    loc="lower right", fontsize=7, framealpha=0.92,
    edgecolor="gray", fancybox=False, handletextpad=0.4,
    borderpad=0.4, labelspacing=0.35,
)

set_map_extent(ax_b)
add_scalebar(ax_b)
add_north_arrow(ax_b)
add_panel_label(ax_b, '(b)')

# ═════════════════════════════════════════════════════════════════════════════
# PANEL (c): Soil hydrologic group
# ═════════════════════════════════════════════════════════════════════════════
soil_group_colors = {
    'C': '#c89664',    # orange/brown
    'D': '#a0a0a0',    # gray
}

int_to_soilgrp = lookup['int_to_soilgrp']
soilgrp_names_present = sorted(set(int_to_soilgrp[str(k)] for k in range(1, len(int_to_soilgrp)+1)))

sg_cmap_list = ['white'] + [soil_group_colors.get(int_to_soilgrp[str(i+1)], '#ffffff')
                             for i in range(len(int_to_soilgrp))]
sg_cmap = ListedColormap(sg_cmap_list)
sg_bounds = np.arange(-0.5, len(int_to_soilgrp) + 1.5, 1)
sg_norm = BoundaryNorm(sg_bounds, sg_cmap.N)

plot_sg = np.ma.masked_where(soilgrp_grid == 0, soilgrp_grid)
ax_c.pcolormesh(
    lon_edges, lat_edges, plot_sg[::-1],
    cmap=sg_cmap, norm=sg_norm, edgecolors="none", zorder=2,
)

add_boundary(ax_c)

# Legend
sg_patches = [mpatches.Patch(facecolor=soil_group_colors[g], edgecolor='black',
                              linewidth=0.4, label=g)
              for g in soilgrp_names_present if g in soil_group_colors]
leg_c = ax_c.legend(
    handles=sg_patches, title="Soil group", title_fontsize=8,
    loc="lower right", fontsize=7, framealpha=0.92,
    edgecolor="gray", fancybox=False, handletextpad=0.4,
    borderpad=0.4, labelspacing=0.35,
)

set_map_extent(ax_c)
add_scalebar(ax_c)
add_north_arrow(ax_c)
add_panel_label(ax_c, '(c)')

# ═════════════════════════════════════════════════════════════════════════════
# PANEL (d): DEM / Topography
# ═════════════════════════════════════════════════════════════════════════════
top_masked = np.ma.masked_where(~active_mask, top_grid)

im_d = ax_d.pcolormesh(
    lon_edges, lat_edges, top_masked[::-1],
    cmap='terrain', edgecolors="none", zorder=2,
    vmin=140, vmax=460,
)

add_boundary(ax_d)

# River cells on top of DEM
ax_d.scatter(riv_lons, riv_lats, s=1.5, c="#1a5fb4", marker="s",
             linewidths=0, zorder=3, label="River cell")

# Colorbar
cbar = fig.colorbar(im_d, ax=ax_d, shrink=0.6, pad=0.02, aspect=25)
cbar.set_label("Elevation (m)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Legend for river cells
leg_d = ax_d.legend(
    loc="lower right", fontsize=7, framealpha=0.92,
    edgecolor="gray", fancybox=False, handletextpad=0.4,
    borderpad=0.4,
)

set_map_extent(ax_d)
add_scalebar(ax_d)
add_north_arrow(ax_d)
add_panel_label(ax_d, '(d)')

# ═════════════════════════════════════════════════════════════════════════════
# Final layout adjustments
# ═════════════════════════════════════════════════════════════════════════════
fig.subplots_adjust(wspace=0.15, hspace=0.08, left=0.05, right=0.95,
                    top=0.97, bottom=0.03)

# Save
out_png = OUT_DIR / "fig_4panel_maps.png"
out_pdf = OUT_DIR / "fig_4panel_maps.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved:\n  {out_png}\n  {out_pdf}")
