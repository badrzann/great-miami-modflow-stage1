"""
map_wells_findings.py  (v2 — publication map with coordinate conflict highlights)
==========================================================================
Same three-category base as map_wells_three_categories.py, plus:

  OVERLAY D  [red star]
    6 model wells whose MCD portal (GeoJSON) coordinates plot OUTSIDE
    the active model boundary (coordinate discrepancy / transcription error)

  OVERLAY E  [black diamond]
    1 model well (LO-3) with extreme coordinate gap (82.8 km) and
    confirmed IBOUND = 0 cell assignment — excluded from valid calibration

Outputs:
  D:/GMRW/finalresult/fig/fig_wells_findings.png  (300 DPI)
  D:/GMRW/finalresult/fig/fig_wells_findings.pdf
"""

import os, csv, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE    = "D:/GMRW/finalresult/swatmf_run"
OBS_CSV = "D:/GMRW/finalresult/obs_data/mcd_well_name_location_matches.csv"
ALL_CSV = "D:/GMRW/finalresult/obs_data/GW_head_all_wells.csv"
GEOJSON = "D:/GMRW/_tmp/all_indicators_geojson.json"
CMP_CSV = "D:/GMRW/finalresult/obs_data/well_coord_head_comparison.csv"
OUT     = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

NROW, NCOL = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW

# Wells whose portal GeoJSON coords plot outside model boundary
OUTSIDE_PORTAL_IDS = {
    'SHE00039', 'HAM00002', 'BUT00013', 'BUT00179', 'BU-12', 'CLA10010'
}
# Well with extreme 82.8 km coord gap AND inactive-cell assignment
EXTREME_COORD_ID = 'LO-3'

# ─────────────────────────────────────────────────────────────────────────────
def load_ibound():
    vals = []
    with open(os.path.join(BASE, "modflow_GMRW.bas")) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.upper() == "FREE":
                continue
            for tok in s.split():
                try: vals.append(int(float(tok)))
                except ValueError: pass
            if len(vals) >= NROW * NCOL: break
    return np.array(vals[:NROW * NCOL], dtype=int).reshape(NROW, NCOL)


def watershed_boundary_segments(ibound):
    active = ibound != 0
    segs = []
    dh = np.diff(active.astype(int), axis=0)
    for r, c in zip(*np.where(dh != 0)):
        y  = LAT_MAX - (r + 1) * DLAT
        x0 = LON_MIN + c * DLON;  x1 = LON_MIN + (c + 1) * DLON
        segs.append([(x0, y), (x1, y)])
    dv = np.diff(active.astype(int), axis=1)
    for r, c in zip(*np.where(dv != 0)):
        y0 = LAT_MAX - r * DLAT;   y1 = LAT_MAX - (r + 1) * DLAT
        x  = LON_MIN + (c + 1) * DLON
        segs.append([(x, y0), (x, y1)])
    for c in range(NCOL):
        if active[0, c]:
            segs.append([(LON_MIN + c*DLON, LAT_MAX), (LON_MIN + (c+1)*DLON, LAT_MAX)])
        if active[NROW-1, c]:
            segs.append([(LON_MIN + c*DLON, LAT_MIN), (LON_MIN + (c+1)*DLON, LAT_MIN)])
    for r in range(NROW):
        if active[r, 0]:
            segs.append([(LON_MIN, LAT_MAX - r*DLAT), (LON_MIN, LAT_MAX - (r+1)*DLAT)])
        if active[r, NCOL-1]:
            segs.append([(LON_MAX, LAT_MAX - r*DLAT), (LON_MAX, LAT_MAX - (r+1)*DLAT)])
    return segs


def cell_centre(row, col):
    """Convert 1-based row/col to cell-centre lon/lat."""
    r, c = int(row) - 1, int(col) - 1
    return LON_MIN + (c + 0.5) * DLON, LAT_MAX - (r + 0.5) * DLAT


def add_scale_bar(ax, lon0, lat0, length_km=50):
    km_per_deg = 111.32 * math.cos(math.radians(lat0))
    dlon = length_km / km_per_deg
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color='black', lw=2,
            solid_capstyle='butt', zorder=20)
    ax.plot([lon0, lon0],               [lat0 - 0.005, lat0 + 0.005], 'k-', lw=2, zorder=20)
    ax.plot([lon0+dlon, lon0+dlon],     [lat0 - 0.005, lat0 + 0.005], 'k-', lw=2, zorder=20)
    ax.text(lon0 + dlon/2, lat0 - 0.02, f'{length_km} km',
            ha='center', va='top', fontsize=8, fontweight='bold', zorder=21)


def add_north_arrow(ax, lon, lat):
    """Simple north arrow."""
    dy = 0.04
    ax.annotate('', xy=(lon, lat + dy), xytext=(lon, lat),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                zorder=25)
    ax.text(lon, lat + dy + 0.005, 'N', ha='center', va='bottom',
            fontsize=9, fontweight='bold', zorder=25)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Data ──────────────────────────────────────────────────────────────
    ibound = load_ibound()
    bsegs  = watershed_boundary_segments(ibound)
    active_fill = np.where(ibound != 0, 1.0, np.nan)

    # Model wells → cell-centre coords
    model_wells = {}   # id -> (lon, lat, local_name, row, col)
    with open(OBS_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lon, lat = cell_centre(row['row'], row['col'])
            model_wells[row['mcd_location_id']] = (
                lon, lat, row['local_well_name'],
                int(row['row']), int(row['col'])
            )

    # New inside-boundary candidate wells
    new_inside_ids = set()
    with open(ALL_CSV, newline='', encoding='utf-8-sig', errors='replace') as f:
        for row in csv.DictReader(f):
            if row.get('status', '') == 'new_inside_boundary':
                new_inside_ids.add(row['mcd_id'])

    # GeoJSON wells
    gw_geo = {}
    with open(GEOJSON, encoding='utf-8') as f:
        data = json.load(f)
    GW_PFX = ('BUT', 'SHE', 'MON', 'WAR', 'CLA', 'MIA', 'PRE', 'CHA',
              'HAM', 'BU-', 'W-9', 'MI-', 'LO-', 'CL-')
    for feat in data['features']:
        p   = feat['properties']
        lid = p['locationIdentifier']
        if any(lid.startswith(px) for px in GW_PFX):
            c = feat['geometry']['coordinates']
            gw_geo[lid] = (float(c[0]), float(c[1]), p['location'])

    model_ids = set(model_wells.keys())

    # ── Separate model wells into "ok" vs "flagged" vs "extreme" ──────────
    cat_A_ok   = []   # normal calibration wells (blue)
    cat_A_out  = []   # 6 portal-outside wells (red star) — plotted at CELL CENTRE
    cat_A_ext  = []   # LO-3 extreme gap (black diamond) — plotted at CELL CENTRE
    for mid, (lon, lat, name, r, c) in model_wells.items():
        if mid == EXTREME_COORD_ID:
            cat_A_ext.append((lon, lat, mid, name))
        elif mid in OUTSIDE_PORTAL_IDS:
            cat_A_out.append((lon, lat, mid, name))
        else:
            cat_A_ok.append((lon, lat, mid, name))

    cat_B, cat_C = [], []
    for mid, (lon, lat, name) in gw_geo.items():
        if mid in model_ids: continue
        (cat_B if mid in new_inside_ids else cat_C).append((lon, lat, mid, name))

    print(f"Cat A-ok  : {len(cat_A_ok)}")
    print(f"Cat A-out : {len(cat_A_out)}  (portal outside boundary)")
    print(f"Cat A-ext :  {len(cat_A_ext)}  ({EXTREME_COORD_ID}: 82.8 km gap)")
    print(f"Cat B     : {len(cat_B)}  (inside, new candidates)")
    print(f"Cat C     : {len(cat_C)}  (other MCD GW wells)")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 10.5), dpi=300)
    fig.subplots_adjust(bottom=0.20)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Watershed fill
    cmap_f = mcolors.ListedColormap(['#e6f2e6'])
    cmap_f.set_bad(color='white', alpha=0)
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    ax.imshow(active_fill, extent=extent, origin='upper',
              cmap=cmap_f, vmin=0, vmax=1, aspect='auto', zorder=1)

    # Watershed boundary
    lc = LineCollection(bsegs, linewidths=1.6, colors='#1a1a1a', zorder=5)
    ax.add_collection(lc)

    # Cat C — background grey
    if cat_C:
        ax.scatter([p[0] for p in cat_C], [p[1] for p in cat_C],
                   s=20, c='white', edgecolors='#aaaaaa',
                   linewidths=0.7, marker='o', zorder=6)

    # Cat B — orange triangles
    if cat_B:
        ax.scatter([p[0] for p in cat_B], [p[1] for p in cat_B],
                   s=55, c='#FF7F0E', edgecolors='#b85000',
                   linewidths=0.8, marker='^', zorder=8)

    # Cat A normal — blue circles
    if cat_A_ok:
        ax.scatter([p[0] for p in cat_A_ok], [p[1] for p in cat_A_ok],
                   s=50, c='#2171b5', edgecolors='#084594',
                   linewidths=0.8, marker='o', zorder=10)

    # Cat A portal-outside — red stars (SIZE larger, plotted at cell-centre)
    if cat_A_out:
        ax.scatter([p[0] for p in cat_A_out], [p[1] for p in cat_A_out],
                   s=140, c='#d62728', edgecolors='#800000',
                   linewidths=0.9, marker='*', zorder=12)
        # Small annotation for each
        for lon, lat, mid, name in cat_A_out:
            short = mid.replace('BUT0', 'BU').replace('SHE0', 'SH').replace(
                    'CLA1', 'CL').replace('HAM0', 'H').replace('BU-', 'BU')
            ax.annotate(short, (lon, lat),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=5.5, color='#800000', zorder=13,
                        fontweight='bold')

    # Cat A extreme (LO-3) — black diamond
    if cat_A_ext:
        ax.scatter([p[0] for p in cat_A_ext], [p[1] for p in cat_A_ext],
                   s=120, c='black', edgecolors='#333333',
                   linewidths=0.9, marker='D', zorder=14)
        lo3 = cat_A_ext[0]
        ax.annotate('LO-3\n(82 km gap)', (lo3[0], lo3[1]),
                    textcoords='offset points', xytext=(8, 6),
                    fontsize=6, color='black', fontweight='bold', zorder=15,
                    bbox=dict(boxstyle='round,pad=0.2', fc='#ffffcc',
                              ec='#aaaaaa', alpha=0.85))

    # ── Counties outline (light grey dotted) from GeoJSON if available ────
    # (skipped if file not found)

    # ── Scale bar ─────────────────────────────────────────────────────────
    add_scale_bar(ax, LON_MIN + 0.04, LAT_MIN + 0.05, 50)

    # ── North arrow ───────────────────────────────────────────────────────
    add_north_arrow(ax, LON_MAX - 0.10, LAT_MIN + 0.08)

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel('Longitude (°W)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
    ax.tick_params(labelsize=9)
    ax.set_aspect('equal')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

    ax.set_title(
        'Great Miami River Watershed — Groundwater Monitoring Wells\n'
        'MCD Portal Coordinates vs SWAT-MF Model Cell Assignment',
        fontsize=11, fontweight='bold', pad=10
    )

    # ── Legend ────────────────────────────────────────────────────────────
    n_total = len(cat_A_ok) + len(cat_A_out) + len(cat_A_ext)
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2171b5',
               markeredgecolor='#084594', markeredgewidth=0.8, markersize=9,
               label=f'Model calibration wells — good assignment  (n={len(cat_A_ok)})'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='#d62728',
               markeredgecolor='#800000', markeredgewidth=0.8, markersize=11,
               label=f'Model well — portal coord OUTSIDE boundary  (n={len(cat_A_out)})'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='black',
               markeredgecolor='#333333', markeredgewidth=0.8, markersize=8,
               label=f'LO-3 — extreme coord gap (82.8 km) / inactive cell  (n=1)'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#FF7F0E',
               markeredgecolor='#b85000', markeredgewidth=0.8, markersize=9,
               label=f'Inside boundary — candidate wells, not in model  (n={len(cat_B)})'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#aaaaaa', markeredgewidth=0.7, markersize=7,
               label=f'Other MCD DTW wells  (n={len(cat_C)})'),
        Line2D([0],[0], color='#1a1a1a', linewidth=1.6,
               label='Watershed boundary'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=8.5,
        framealpha=0.95,
        edgecolor='#aaaaaa',
        handletextpad=0.5,
        handlelength=1.5,
        borderpad=0.8,
        columnspacing=1.0,
    )

    # ── Inset note ────────────────────────────────────────────────────────
    note = (
        "Red stars plotted at model-assigned cell centres.\n"
        "Portal (web) coordinates for these 6 wells fall outside active domain."
    )
    fig.text(0.12, 0.025, note, ha='left', va='bottom',
             fontsize=7.5, color='#555555', style='italic')

    # ── Save ──────────────────────────────────────────────────────────────
    for ext in ('png', 'pdf'):
        fp = os.path.join(OUT, f'fig_wells_findings.{ext}')
        fig.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"Saved: {fp}")
    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    main()
