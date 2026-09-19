"""
map_wells_three_categories.py
==============================
Publication-quality map showing all MCD groundwater monitoring wells in and
around the GMRW watershed, distinguished into three categories:

  Category A  [blue filled circle]
    85 wells currently used in the SWAT-MF model calibration

  Category B  [orange filled triangle]
    Inside-boundary MCD wells with recorded DTW data, NOT yet in the model
    (23 wells with enough data for head estimation)

  Category C  [light-grey open circle]
    All other MCD Depth-to-Water wells (outside or on the boundary,
    not included in the model)

Background: MODFLOW DIS TOP array (DEM) with watershed IBOUND boundary.

Outputs:
  D:/GMRW/finalresult/fig/fig_wells_three_categories.png  (300 DPI)
  D:/GMRW/finalresult/fig/fig_wells_three_categories.pdf
"""

import os, re, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE      = "D:/GMRW/finalresult/swatmf_run"
OBS_CSV   = "D:/GMRW/finalresult/obs_data/mcd_well_name_location_matches.csv"
ALL_CSV   = "D:/GMRW/finalresult/obs_data/GW_head_all_wells.csv"
GEOJSON   = "D:/GMRW/_tmp/all_indicators_geojson.json"
OUT       = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRID CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NROW, NCOL   = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD IBOUND
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DEM (TOP array)
# ─────────────────────────────────────────────────────────────────────────────
def load_dem():
    vals = []
    in_top = False
    with open(os.path.join(BASE, "modflow_GMRW.dis")) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if 'TOP' in s.upper() and not in_top:
                in_top = True
                continue
            if in_top:
                for p in s.split():
                    try:
                        vals.append(float(p))
                    except ValueError:
                        pass
                if len(vals) >= NROW * NCOL:
                    break
    return np.array(vals[:NROW * NCOL], dtype=float).reshape(NROW, NCOL)


# ─────────────────────────────────────────────────────────────────────────────
# 3. WATERSHED BOUNDARY SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
def watershed_boundary_segments(ibound):
    active = ibound != 0
    segs = []
    diff_h = np.diff(active.astype(int), axis=0)
    for r, c in zip(*np.where(diff_h != 0)):
        lat_y  = LAT_MAX - (r + 1) * DLAT
        lon_x0 = LON_MIN + c * DLON
        lon_x1 = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x0, lat_y), (lon_x1, lat_y)])
    diff_v = np.diff(active.astype(int), axis=1)
    for r, c in zip(*np.where(diff_v != 0)):
        lat_y0 = LAT_MAX - r * DLAT
        lat_y1 = LAT_MAX - (r + 1) * DLAT
        lon_x  = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x, lat_y0), (lon_x, lat_y1)])
    for c in range(NCOL):
        if active[0, c]:
            segs.append([(LON_MIN + c * DLON, LAT_MAX), (LON_MIN + (c + 1) * DLON, LAT_MAX)])
        if active[NROW - 1, c]:
            segs.append([(LON_MIN + c * DLON, LAT_MIN), (LON_MIN + (c + 1) * DLON, LAT_MIN)])
    for r in range(NROW):
        if active[r, 0]:
            segs.append([(LON_MIN, LAT_MAX - r * DLAT), (LON_MIN, LAT_MAX - (r + 1) * DLAT)])
        if active[r, NCOL - 1]:
            segs.append([(LON_MAX, LAT_MAX - r * DLAT), (LON_MAX, LAT_MAX - (r + 1) * DLAT)])
    return segs


def is_inside(lon, lat, ibound):
    row = int((LAT_MAX - lat) / DLAT)
    col = int((lon - LON_MIN) / DLON)
    if 0 <= row < NROW and 0 <= col < NCOL:
        return ibound[row, col] != 0
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD WELL DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_model_wells():
    """Return dict of mcd_id -> (lon, lat, local_name) using grid-snapped CSV coordinates."""
    wells = {}
    with open(OBS_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # Cell-centre coordinate from the model grid (guaranteed inside active boundary)
            r = int(row['row']) - 1
            c = int(row['col']) - 1
            lon = LON_MIN + (c + 0.5) * DLON
            lat = LAT_MAX - (r + 0.5) * DLAT
            wells[row['mcd_location_id']] = (lon, lat, row['local_well_name'])
    return wells


def load_model_well_ids():
    """Return set of MCD IDs for the 85 model wells."""
    return set(load_model_wells().keys())


def load_new_inside_ids():
    """Return set of MCD IDs flagged as new_inside_boundary in GW_head_all_wells.csv."""
    ids = set()
    with open(ALL_CSV, newline='', encoding='utf-8-sig', errors='replace') as f:
        for row in csv.DictReader(f):
            if row.get('status', '') == 'new_inside_boundary':
                ids.add(row['mcd_id'])
    return ids


def load_geojson_gw_wells():
    """
    Extract all GW-style wells from the MCD GeoJSON.
    Returns dict: mcd_id -> (lon, lat, display_name)
    """
    with open(GEOJSON, encoding='utf-8') as f:
        data = json.load(f)

    GW_PREFIXES = ('BUT', 'SHE', 'MON', 'WAR', 'CLA', 'MIA', 'PRE', 'CHA',
                   'HAM', 'BU-', 'W-9', 'MI-', 'LO-', 'CL-', 'BU-')

    wells = {}
    for feat in data['features']:
        p   = feat['properties']
        lid = p['locationIdentifier']
        if not any(lid.startswith(pf) for pf in GW_PREFIXES):
            continue
        # also catch plain numeric IDs that appear in CLA, MIA etc.
        coords = feat['geometry']['coordinates']
        wells[lid] = (float(coords[0]), float(coords[1]), p['location'])
    return wells


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCALE BAR
# ─────────────────────────────────────────────────────────────────────────────
def add_scale_bar(ax, lon0, lat0, length_km=50):
    """Draw a 50-km scale bar at given lon/lat anchor (bottom-left of bar)."""
    # 1 degree lon at this latitude ≈ 111.32 * cos(lat) km
    import math
    km_per_deg_lon = 111.32 * math.cos(math.radians((lat0 + lat0) / 2))
    dlon = length_km / km_per_deg_lon

    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color='black', lw=2,
            transform=ax.transData, solid_capstyle='butt')
    ax.plot([lon0, lon0],             [lat0 - 0.005, lat0 + 0.005], color='black', lw=2)
    ax.plot([lon0 + dlon, lon0 + dlon], [lat0 - 0.005, lat0 + 0.005], color='black', lw=2)
    ax.text(lon0 + dlon / 2, lat0 - 0.018, f'{length_km} km',
            ha='center', va='top', fontsize=8, fontweight='bold')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading IBOUND...")
    ibound = load_ibound()

    print("Loading well data...")
    model_wells    = load_model_wells()        # mcd_id -> (snapped_lon, snapped_lat, name)
    model_ids      = set(model_wells.keys())
    new_inside_ids = load_new_inside_ids()

    print("Loading GeoJSON GW wells...")
    gw_wells = load_geojson_gw_wells()         # all MCD GW wells (true coords)

    boundary_segs = watershed_boundary_segments(ibound)

    # ── Categorise every MCD GW well ──────────────────────────────────────
    # Category A: use grid-snapped coordinates (always inside active boundary)
    cat_A = [(lon, lat, mid, name) for mid, (lon, lat, name) in model_wells.items()]

    # Category B and C: use true GeoJSON coordinates
    cat_B = []   # inside boundary, not in model, has data
    cat_C = []   # other (outside boundary or no data)
    for mid, (lon, lat, name) in gw_wells.items():
        if mid in model_ids:
            continue   # already in cat_A with snapped coords
        elif mid in new_inside_ids:
            cat_B.append((lon, lat, mid, name))
        else:
            cat_C.append((lon, lat, mid, name))

    print(f"  Category A (model wells):           {len(cat_A)}")
    print(f"  Category B (inside, not in model):  {len(cat_B)}")
    print(f"  Category C (other MCD GW wells):    {len(cat_C)}")

    # ── Active-cell fill (flat, no DEM colorbar) ────────────────────────
    active_fill = np.where(ibound != 0, 1.0, np.nan)
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 10), dpi=300)
    fig.subplots_adjust(bottom=0.15)   # space below axes for legend

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Watershed interior — solid light fill so it is clearly distinct from outside
    from matplotlib.colors import LinearSegmentedColormap
    cmap_fill = matplotlib.colors.ListedColormap(['#e8f4e8'])   # very light green
    cmap_fill.set_bad(color='white', alpha=0)   # NaN (inactive) → white
    ax.imshow(
        active_fill,
        extent=extent,
        origin='upper',
        cmap=cmap_fill,
        vmin=0, vmax=1,
        aspect='auto',
        zorder=1,
    )

    # Watershed boundary — thicker, dark
    lc = LineCollection(boundary_segs, linewidths=1.8, colors='#222222', zorder=5)
    ax.add_collection(lc)

    # ── Plot wells ────────────────────────────────────────────────────────
    # Category C first (background)
    if cat_C:
        xC = [p[0] for p in cat_C]
        yC = [p[1] for p in cat_C]
        ax.scatter(xC, yC, s=22, c='white', edgecolors='#999999',
                   linewidths=0.8, marker='o', zorder=6,
                   label=f'MCD wells – outside/unassigned (n={len(cat_C)})')

    # Category B (inside, not in model)
    if cat_B:
        xB = [p[0] for p in cat_B]
        yB = [p[1] for p in cat_B]
        ax.scatter(xB, yB, s=60, c='#FF7F0E', edgecolors='#cc5500',
                   linewidths=0.8, marker='^', zorder=8,
                   label=f'Inside boundary – not in model (n={len(cat_B)})')

    # Category A (model wells) — on top
    if cat_A:
        xA = [p[0] for p in cat_A]
        yA = [p[1] for p in cat_A]
        ax.scatter(xA, yA, s=55, c='#1F77B4', edgecolors='#0a3d6b',
                   linewidths=0.8, marker='o', zorder=10,
                   label=f'Model calibration wells (n={len(cat_A)})')

    # ── Scale bar (bottom-left, outside the active watershed) ─────────────
    add_scale_bar(ax, lon0=LON_MIN + 0.04, lat0=LAT_MIN + 0.05, length_km=50)

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel('Longitude (°W)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
    ax.tick_params(labelsize=9)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title(
        'MCD Groundwater Monitoring Wells\nGreat Miami River Watershed',
        fontsize=12, fontweight='bold', pad=10
    )

    # ── Legend — placed BELOW the map, 2 columns ─────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4',
               markeredgecolor='#0a3d6b', markeredgewidth=0.8,
               markersize=10, label=f'Model calibration wells  (n={len(cat_A)})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF7F0E',
               markeredgecolor='#b85000', markeredgewidth=0.8,
               markersize=10, label=f'Inside boundary – not in model  (n={len(cat_B)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#888888', markeredgewidth=0.9,
               markersize=8, label=f'MCD wells outside watershed  (n={len(cat_C)})'),
        Line2D([0], [0], color='#222222', linewidth=1.8,
               label='Watershed boundary'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),   # below the axes
        ncol=2,
        fontsize=9,
        framealpha=0.95,
        edgecolor='#aaaaaa',
        handletextpad=0.6,
        handlelength=1.8,
        borderpad=0.8,
        columnspacing=1.2,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    for ext in ('png', 'pdf'):
        out_path = os.path.join(OUT, f'fig_wells_three_categories.{ext}')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out_path}")

    plt.close(fig)
    print("\nDone.")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n=== CATEGORY SUMMARY ===")
    print(f"  A – Model calibration wells:      {len(cat_A)}")
    print(f"  B – Inside boundary, not in model:{len(cat_B)}")
    print(f"  C – Other MCD GW wells:           {len(cat_C)}")
    total = len(cat_A) + len(cat_B) + len(cat_C)
    print(f"  Total GW wells in GeoJSON:        {total}")
    print()
    print("  Category B wells (candidates for future model use):")
    for lon, lat, mid, name in sorted(cat_B, key=lambda x: x[2]):
        print(f"    {mid:<15} {name:<30} ({lon:.4f}, {lat:.4f})")


if __name__ == '__main__':
    main()
