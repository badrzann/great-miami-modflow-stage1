"""
FIXED 4-panel publication-quality maps for the GMRW manuscript.

Fixes applied:
1. Wells filtered to ONLY those inside active IBOUND boundary
2. Land use: deterministic dominant-per-subbasin (clean solid patches)
3. Soil: deterministic dominant-per-subbasin (clean solid patches)
4. Panel (a): clear MODFLOW grid with visible cell lines

(a) MODFLOW grid + river cells + wells (inside boundary only) + gauge
(b) Land use: grouped SWAT categories (clean solid subbasin patches)
(c) Soil hydrologic group (clean solid patches)
(d) Elevation / DEM with river network
"""

import json, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, LightSource
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.linewidth': 0.7,
})

# ── Configuration ─────────────────────────────────────────────────────────────
NROW, NCOL   = 197, 135
LON_MIN      = -84.855
LAT_MIN      = 39.185
DX = DY      = 0.01

SWATMF_DIR   = Path(r"D:\GMRW\finalresult\swatmf_run")
MAP_DIR      = Path(r"D:\GMRW\mapping")
OUT_DIR      = Path(r"D:\GMRW\finalresult\fig")
GAUGE_LON, GAUGE_LAT = -84.2851, 39.6368

# ── Coordinate helpers ───────────────────────────────────────────────────────
lon_edges = np.linspace(LON_MIN, LON_MIN + NCOL*DX, NCOL+1)
lat_edges = np.linspace(LAT_MIN, LAT_MIN + NROW*DY, NROW+1)
lon_cen   = LON_MIN + (np.arange(NCOL) + 0.5)*DX
lat_cen   = LAT_MIN + (NROW - np.arange(NROW) - 0.5)*DY
lon_grid, lat_grid = np.meshgrid(lon_cen, lat_cen)

def rc2lonlat(row, col):
    return LON_MIN+(col+0.5)*DX, LAT_MIN+(NROW-row-0.5)*DY

def lonlat2rc(lon, lat):
    col = int((lon - LON_MIN) / DX)
    row = NROW - 1 - int((lat - LAT_MIN) / DY)
    return row, col

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load IBOUND
# ══════════════════════════════════════════════════════════════════════════════
ibound     = np.load(MAP_DIR / "ibound.npy")
sub_grid   = np.load(MAP_DIR / "sub_grid.npy")
top_grid   = np.load(MAP_DIR / "top_grid.npy")
active_mask = ibound > 0

# IBOUND from BAS for contour boundary
def load_ibound_bas():
    with open(SWATMF_DIR / "modflow_GMRW.bas") as fh:
        lines = fh.readlines()
    start = next(i+1 for i, l in enumerate(lines) if "IBOUND" in l.upper())
    vals = []
    for line in lines[start:]:
        parts = [x for x in line.split() if x.lstrip('-').isdigit()]
        if not parts: continue
        vals.extend(int(x) for x in parts)
        if len(vals) >= NROW*NCOL: break
    return np.array(vals[:NROW*NCOL]).reshape((NROW, NCOL))

ibound_bas = load_ibound_bas()

# ══════════════════════════════════════════════════════════════════════════════
# 2. Build DETERMINISTIC land use and soil grids (clean patches)
# ══════════════════════════════════════════════════════════════════════════════
luse_grouping = {
    'AGRL': 'Agriculture', 'PAST': 'Agriculture',
    'RNGB': 'Rangeland',   'RNGE': 'Rangeland',
    'FRSD': 'Forest',      'FRSE': 'Forest',  'FRST': 'Forest',
    'UIDU': 'Urban',       'URHD': 'Urban',   'URLD': 'Urban',   'URMD': 'Urban', 'BARR': 'Urban',
    'WATR': 'Water',
    'WETL': 'Wetland',     'WETN': 'Wetland',
}
group_names_lu  = ['Agriculture', 'Forest', 'Urban', 'Water', 'Wetland', 'Rangeland']
group_names_sg  = ['Af17-1-2a-2 (Group C)', 'Af32-2ab-3 (Group D)']
soil_label_map  = {'Af17-1-2a-2': 'Group C (Af17)', 'Af32-2ab-3': 'Group D (Af32)', 'Ao39-2b-4': 'Group C (Ao39)'}
sg_simple       = {'Af17-1-2a-2': 'C', 'Af32-2ab-3': 'D', 'Ao39-2b-4': 'C'}

group_to_int_lu = {g: i+1 for i, g in enumerate(group_names_lu)}
sg_to_int       = {'Af17-1-2a-2': 1, 'Af32-2ab-3': 2, 'Ao39-2b-4': 1}  # C=1, D=2

# Parse HRU headers to get per-subbasin dominant properties
sub_luse_fracs = defaultdict(lambda: defaultdict(float))
sub_soil_fracs = defaultdict(lambda: defaultdict(float))
hru_files = glob.glob(r'D:\GMRW\swat-modflow files\*.hru')
for fpath in hru_files:
    with open(fpath) as fh:
        header = fh.readline().strip()
        line2  = fh.readline().strip()
    m = re.search(r'Subbasin:(\d+)\s+HRU:\d+\s+Luse:(\S+)\s+Soil:\s*(\S+)', header)
    if m:
        sub  = int(m.group(1))
        lu   = m.group(2)
        soil = m.group(3)
        fr   = float(line2.split('|')[0].strip()) if '|' in line2 else 0.0
        grp  = luse_grouping.get(lu, 'Agriculture')
        sub_luse_fracs[sub][grp]  += fr
        sub_soil_fracs[sub][soil] += fr

# Compute dominant per subbasin
sub_dom_lu   = {s: max(d, key=d.get) for s, d in sub_luse_fracs.items()}
sub_dom_soil = {s: max(d, key=d.get) for s, d in sub_soil_fracs.items()}

# Build deterministic grids
luse_det_grid = np.zeros((NROW, NCOL), dtype=int)
soil_det_grid = np.zeros((NROW, NCOL), dtype=int)
for r in range(NROW):
    for c in range(NCOL):
        sub = sub_grid[r, c]
        if sub == 0: continue
        if sub in sub_dom_lu:
            luse_det_grid[r, c] = group_to_int_lu.get(sub_dom_lu[sub], 0)
        if sub in sub_dom_soil:
            soil_det_grid[r, c] = sg_to_int.get(sub_dom_soil[sub], 0)

print("Land use distribution:")
from collections import Counter
lc = Counter(luse_det_grid[luse_det_grid > 0])
for code, n in sorted(lc.items(), key=lambda x: -x[1]):
    print(f"  {group_names_lu[code-1]:15s}: {n:5d} ({100*n/sum(lc.values()):.1f}%)")
print("Soil group distribution:")
sc = Counter(soil_det_grid[soil_det_grid > 0])
for code, n in sorted(sc.items(), key=lambda x: -x[1]):
    print(f"  {'C' if code==1 else 'D':5s}: {n:5d} ({100*n/sum(sc.values()):.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. River cells
# ══════════════════════════════════════════════════════════════════════════════
def load_river_cells():
    cells = []
    with open(SWATMF_DIR / "modflow_GMRW.riv") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 3 and p[0].isdigit():
                cells.append((int(p[1])-1, int(p[2])-1))
    return cells

riv_cells = load_river_cells()
riv_lons  = np.array([rc2lonlat(r,c)[0] for r,c in riv_cells])
riv_lats  = np.array([rc2lonlat(r,c)[1] for r,c in riv_cells])

# ══════════════════════════════════════════════════════════════════════════════
# 4. Wells: filter to INSIDE active zone only
# ══════════════════════════════════════════════════════════════════════════════
wells_all = pd.read_csv(SWATMF_DIR / "well_name_mapping.csv")
inside_mask = []
for _, w in wells_all.iterrows():
    r, c = lonlat2rc(w['lon'], w['lat'])
    inside_mask.append(0 <= r < NROW and 0 <= c < NCOL and ibound[r, c] > 0)
wells = wells_all[inside_mask].reset_index(drop=True)
print(f"\nWells inside active boundary: {len(wells)} / {len(wells_all)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Figure setup
# ══════════════════════════════════════════════════════════════════════════════
# Active area extent
ar, ac = np.where(active_mask)
pad = 4
x_lo = LON_MIN + max(0, ac.min()-pad)*DX
x_hi = LON_MIN + min(NCOL, ac.max()+pad+1)*DX
y_lo = LAT_MIN + (NROW - min(NROW, ar.max()+pad+1))*DY
y_hi = LAT_MIN + (NROW - max(0, ar.min()-pad))*DY

fig = plt.figure(figsize=(14, 15.5), dpi=300)
W, H   = 0.42, 0.435
gap_x  = 0.06
gap_y  = 0.04
x0, y1 = 0.055, 0.97

ax_a = fig.add_axes([x0,              y1-H,            W, H])
ax_b = fig.add_axes([x0+W+gap_x,      y1-H,            W, H])
ax_c = fig.add_axes([x0,              y1-2*H-gap_y,    W, H])
ax_d = fig.add_axes([x0+W+gap_x,      y1-2*H-gap_y,    W, H])

# ── Shared helpers ───────────────────────────────────────────────────────────
def add_boundary(ax, lw=1.5):
    ax.contour(lon_grid, lat_grid, ibound_bas.astype(float),
               levels=[0.5], colors='black', linewidths=lw, zorder=15)

def set_extent(ax):
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7.5, direction='in', length=3, width=0.5)

def add_scalebar(ax, loc='lower left'):
    ax.add_artist(ScaleBar(dx=111_320, units='m', location=loc,
                           length_fraction=0.12, font_properties={'size': 7.5},
                           box_alpha=0.88, sep=2, pad=0.5,
                           border_pad=0.4, scale_loc='bottom', label_loc='top'))

def add_north_arrow(ax, x=0.055, y=0.965):
    ax.annotate('', xy=(x, y-0.01), xycoords='axes fraction',
                xytext=(x, y-0.068), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=2.0, color='black',
                                mutation_scale=14), zorder=22)
    ax.text(x, y+0.003, 'N', transform=ax.transAxes,
            fontsize=11, fontweight='bold', ha='center', va='bottom', zorder=22)

def add_label(ax, label, x=0.16, y=0.975):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left', zorder=25)

def style_legend(leg):
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor('#888888')

# ══════════════════════════════════════════════════════════════════════════════
# (a) Study area: active grid + river + wells (filtered) + gauge
# ══════════════════════════════════════════════════════════════════════════════
# Light fill on active cells
ax_a.pcolormesh(lon_edges, lat_edges,
                np.where(active_mask, 1.0, np.nan)[::-1],
                cmap=ListedColormap(['#EBEBEB']), zorder=1)

# MODFLOW grid lines (clearly visible, only inside active region)
r0, r1 = max(0, ar.min()-1), min(NROW+1, ar.max()+2)
c0, c1 = max(0, ac.min()-1), min(NCOL+1, ac.max()+2)
for r in range(r0, r1+1):
    y = LAT_MIN + (NROW - r)*DY
    ax_a.plot([x_lo, x_hi], [y, y], color='#C8C8C8', lw=0.12, zorder=2)
for c in range(c0, c1+1):
    x = LON_MIN + c*DX
    ax_a.plot([x, x], [y_lo, y_hi], color='#C8C8C8', lw=0.12, zorder=2)

add_boundary(ax_a)

# River cells (light blue filled squares)
ax_a.scatter(riv_lons, riv_lats, s=2.8, c='#4A90D9', marker='s',
             linewidths=0, zorder=4, label='River cell')

# Monitoring wells (dark green squares, inside boundary only)
ax_a.scatter(wells['lon'], wells['lat'], s=22, c='#1B5E20',
             marker='s', linewidths=0.4, edgecolors='black',
             zorder=8, label=f'Monitoring well (n={len(wells)})')

# Flow gauge (red star)
ax_a.scatter(GAUGE_LON, GAUGE_LAT, s=120, c='#D32F2F', marker='*',
             edgecolors='black', linewidths=0.5, zorder=9,
             label='Flow gauge')

set_extent(ax_a)
add_scalebar(ax_a)
add_north_arrow(ax_a)
add_label(ax_a, '(a)')

leg_a = ax_a.legend(loc='lower right', fontsize=7.5, framealpha=0.92,
                    fancybox=False, handletextpad=0.5,
                    borderpad=0.5, labelspacing=0.4)
style_legend(leg_a)
ax_a.set_ylabel('Latitude (°N)', fontsize=9)
ax_a.tick_params(axis='x', labelbottom=False)

# ══════════════════════════════════════════════════════════════════════════════
# (b) Land use (deterministic dominant per subbasin → clean solid patches)
# ══════════════════════════════════════════════════════════════════════════════
lu_colors = {
    'Agriculture': '#C8B45A',
    'Forest':      '#2B6B2B',
    'Urban':       '#ADADAD',
    'Water':       '#3A7ABF',
    'Wetland':     '#6ABFAD',
    'Rangeland':   '#D4A843',
}
cmap_lu = ListedColormap(['white'] + [lu_colors[g] for g in group_names_lu])
norm_lu = BoundaryNorm(np.arange(-0.5, len(group_names_lu)+1.5, 1), cmap_lu.N)

ax_b.pcolormesh(lon_edges, lat_edges,
                np.ma.masked_where(luse_det_grid==0, luse_det_grid)[::-1],
                cmap=cmap_lu, norm=norm_lu, zorder=2)

# River overlay (slightly more visible)
ax_b.scatter(riv_lons, riv_lats, s=1.2, c='#1565C0',
             marker='s', linewidths=0, zorder=3, alpha=0.85)

add_boundary(ax_b)
set_extent(ax_b)
add_scalebar(ax_b)
add_north_arrow(ax_b)
add_label(ax_b, '(b)')

present_lu = sorted(set(luse_det_grid[luse_det_grid > 0]))
lu_patches = [mpatches.Patch(facecolor=lu_colors[group_names_lu[g-1]],
                             edgecolor='black', lw=0.4,
                             label=group_names_lu[g-1])
              for g in present_lu]
leg_b = ax_b.legend(handles=lu_patches, title='Land use', title_fontsize=8,
                    loc='lower right', fontsize=7.5, framealpha=0.92,
                    fancybox=False, handletextpad=0.5,
                    borderpad=0.5, labelspacing=0.3)
style_legend(leg_b)
ax_b.tick_params(axis='x', labelbottom=False)
ax_b.tick_params(axis='y', labelleft=False)

# ══════════════════════════════════════════════════════════════════════════════
# (c) Soil hydrologic group (deterministic dominant per subbasin)
# ══════════════════════════════════════════════════════════════════════════════
sg_colors = {1: '#E0A060', 2: '#9A9A9A'}   # C = warm tan, D = gray
sg_labels  = {1: 'Group C', 2: 'Group D'}

cmap_sg = ListedColormap(['white', sg_colors[1], sg_colors[2]])
norm_sg = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_sg.N)

ax_c.pcolormesh(lon_edges, lat_edges,
                np.ma.masked_where(soil_det_grid==0, soil_det_grid)[::-1],
                cmap=cmap_sg, norm=norm_sg, zorder=2)

add_boundary(ax_c)
set_extent(ax_c)
add_scalebar(ax_c)
add_north_arrow(ax_c)
add_label(ax_c, '(c)')

sg_patches = [mpatches.Patch(facecolor=sg_colors[i], edgecolor='black',
                              lw=0.4, label=sg_labels[i])
              for i in [1, 2] if i in Counter(soil_det_grid[soil_det_grid > 0])]
leg_c = ax_c.legend(handles=sg_patches, title='Soil group', title_fontsize=8,
                    loc='lower right', fontsize=7.5, framealpha=0.92,
                    fancybox=False, handletextpad=0.5,
                    borderpad=0.5, labelspacing=0.35)
style_legend(leg_c)
ax_c.set_xlabel('Longitude (°W)', fontsize=9)
ax_c.set_ylabel('Latitude (°N)', fontsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# (d) DEM / Topography with river network
# ══════════════════════════════════════════════════════════════════════════════
top_ma = np.ma.masked_where(~active_mask, top_grid)

# DEM with hillshade blending
from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)

# Normalize terrain for shade computation (fill inactive with mean to avoid edge effects)
top_filled = np.where(active_mask, top_grid, np.nanmean(top_grid[active_mask]))
hill = ls.hillshade(top_filled, vert_exag=5, dx=913, dy=913)
hill_ma = np.ma.masked_where(~active_mask, hill)

# Terrain coloring
im_ref = ax_d.pcolormesh(lon_edges, lat_edges, top_ma[::-1],
                          cmap='terrain', vmin=140, vmax=460, zorder=2)

# Overlay hillshade as a semi-transparent gray shading
hill_plot = np.ma.masked_where(~active_mask, hill)
ax_d.pcolormesh(lon_edges, lat_edges, hill_plot[::-1],
                cmap='gray', vmin=0, vmax=1,
                alpha=0.3, zorder=3)

add_boundary(ax_d)

# River cells - only inside active boundary
riv_mask_d = [active_mask[r, c] if 0 <= r < NROW and 0 <= c < NCOL else False
              for r, c in riv_cells]
ax_d.scatter(riv_lons[riv_mask_d], riv_lats[riv_mask_d], s=1.5, c='#1565C0',
             marker='s', linewidths=0, zorder=6, label='River cell')

set_extent(ax_d)
add_scalebar(ax_d)
add_north_arrow(ax_d)
add_label(ax_d, '(d)')

# Colorbar
pos_d = ax_d.get_position()
cax = fig.add_axes([pos_d.x0 + pos_d.width + 0.01,
                    pos_d.y0 + pos_d.height*0.15,
                    0.012, pos_d.height*0.6])
cbar = fig.colorbar(im_ref, cax=cax)
cbar.set_label('Elevation (m)', fontsize=8, labelpad=4)
cbar.ax.tick_params(labelsize=7)

leg_d = ax_d.legend(loc='lower right', fontsize=7.5, framealpha=0.92,
                    fancybox=False, handletextpad=0.5, borderpad=0.5)
style_legend(leg_d)
ax_d.set_xlabel('Longitude (°W)', fontsize=9)
ax_d.tick_params(axis='y', labelleft=False)

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
out_png = OUT_DIR / "fig_4panel_maps.png"
out_pdf = OUT_DIR / "fig_4panel_maps.pdf"
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved:\n  {out_png}\n  {out_pdf}")
