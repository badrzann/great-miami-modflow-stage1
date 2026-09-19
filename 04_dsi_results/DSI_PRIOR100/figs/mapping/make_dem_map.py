"""
make_dem_map.py
===============
Digital Elevation Model figure with USA locator map.

Title: "Digital elevation model of the study watershed with its geographic location"

Layout:
  Left  – USA locator map  (US states from Natural Earth, red dashed GMRW box,
           two red diagonal connecting lines to the right panel)
  Right – DEM map with:
            • Proper longitude / latitude axes
            • jet-style DEM colormap (blue=low → red=high)
            • Red watershed boundary
            • Black filled triangles = monitoring wells
            • Scale bar (50 km)
            • Legend inside upper-right overlapping the map

DEM source: MODFLOW DIS file TOP array (ground-surface elevation, metres AMSL)

Outputs:
  D:/GMRW/finalresult/fig/fig02_study_area_map.png   (300 DPI)
  D:/GMRW/finalresult/fig/fig02_study_area_map.pdf
"""

import os, re, json
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE = "D:/GMRW/finalresult/swatmf_run"
OUT  = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

NROW, NCOL = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
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


def load_dem():
    """Read TOP array from MODFLOW DIS file (ground-surface elevation, m)."""
    vals = []
    capturing = False
    with open(os.path.join(BASE, "modflow_GMRW.dis")) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "TOP" in s.upper() and "INTERNAL" in s.upper():
                capturing = True
                continue
            if capturing:
                if re.match(r"(INTERNAL|EXTERNAL|CONSTANT)\b", s, re.I) and len(vals) > 0:
                    break
                for tok in s.split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
                if len(vals) >= NROW * NCOL:
                    break
    return np.array(vals[:NROW * NCOL], dtype=float).reshape(NROW, NCOL)


def load_wells():
    wells = []
    with open(os.path.join(BASE, "modflow.obs")) as f:
        f.readline()
        n = int(f.readline().strip())
        for _ in range(n):
            parts = f.readline().split()
            row0 = int(parts[0]) - 1
            col0 = int(parts[1]) - 1
            lon = LON_MIN + (col0 + 0.5) * DLON
            lat = LAT_MAX - (row0 + 0.5) * DLAT
            wells.append((lon, lat))
    return wells


# ─────────────────────────────────────────────────────────────────────────────
# 2. WATERSHED BOUNDARY SEGMENTS
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
            segs.append([(LON_MIN + c*DLON, LAT_MAX), (LON_MIN + (c+1)*DLON, LAT_MAX)])
        if active[NROW-1, c]:
            segs.append([(LON_MIN + c*DLON, LAT_MIN), (LON_MIN + (c+1)*DLON, LAT_MIN)])
    for r in range(NROW):
        if active[r, 0]:
            segs.append([(LON_MIN, LAT_MAX - r*DLAT), (LON_MIN, LAT_MAX - (r+1)*DLAT)])
        if active[r, NCOL-1]:
            segs.append([(LON_MAX, LAT_MAX - r*DLAT), (LON_MAX, LAT_MAX - (r+1)*DLAT)])
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# 3. DOWNLOAD / LOAD US STATES
# ─────────────────────────────────────────────────────────────────────────────
_STATES_CACHE = "D:/GMRW/_tmp_us_states.json"
_STATES_URL   = ("https://raw.githubusercontent.com/"
                 "PublicaMundi/MappingAPI/master/data/geojson/us-states.json")

def load_us_states():
    """Return list of (polygons, multipolygons) as lists of (lons, lats)."""
    # Try cache first
    if os.path.exists(_STATES_CACHE):
        with open(_STATES_CACHE) as f:
            data = json.load(f)
    else:
        print("  Downloading US states GeoJSON …")
        with urllib.request.urlopen(_STATES_URL, timeout=12) as r:
            data = json.loads(r.read())
        with open(_STATES_CACHE, "w") as f:
            json.dump(data, f)

    polys = []
    for feat in data["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords_list = geom["coordinates"]
        elif geom["type"] == "MultiPolygon":
            # flatten all rings
            coords_list = [ring
                           for part in geom["coordinates"]
                           for ring in part]
        else:
            continue
        for ring in coords_list:
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            polys.append((lons, lats))
    return polys


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCALE BAR HELPER
# ─────────────────────────────────────────────────────────────────────────────
def add_scale_bar(ax, length_km=50):
    km_per_deg_lon = 111.32 * np.cos(np.radians(39.7))
    bar_deg = length_km / km_per_deg_lon
    x0 = LON_MIN + 0.04 * (LON_MAX - LON_MIN)
    y0 = LAT_MIN + 0.04 * (LAT_MAX - LAT_MIN)
    ax.plot([x0, x0 + bar_deg], [y0, y0], "k-", lw=3, solid_capstyle="butt",
            transform=ax.transData, zorder=10)
    for x in [x0, x0 + bar_deg]:
        ax.plot(x, y0, "k|", lw=2, ms=6, zorder=10)
    ax.text(x0 + bar_deg / 2, y0 - 0.025, f"{length_km} km",
            ha="center", va="top", fontsize=9, fontweight="bold", zorder=10)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(ibound, dem, wells):
    active = ibound != 0

    # DEM display: mask inactive cells — ONLY active cells get DEM colour
    dem_disp = dem.astype(float).copy()
    dem_disp[~active] = np.nan
    dem_disp[dem_disp <= 10] = np.nan   # remove stray zeros outside domain

    # Fixed colour range matching reference figure (197–342 m)
    dem_vals = dem[active & (dem > 10)]
    vmin = 197.0
    vmax = 342.0

    # ── Figure and axes ───────────────────────────────────────────────────────
    fig, (ax_loc, ax_dem) = plt.subplots(
        1, 2,
        figsize=(16, 7),
        facecolor="white",
        gridspec_kw={"width_ratios": [0.42, 0.58], "wspace": 0.06}
    )
    fig.subplots_adjust(left=0.02, right=0.93, top=0.91, bottom=0.06)

    # ── LEFT PANEL: USA LOCATOR ───────────────────────────────────────────────
    ax_loc.set_facecolor("#C4E4FF")   # ocean blue

    # Draw US states
    print("  Drawing US states …")
    state_polys = load_us_states()
    # Filter to contiguous US (exclude Hawaii/Alaska by longitude)
    for lons, lats in state_polys:
        cx = np.mean(lons)
        if cx < -128 or cx > -65 or np.mean(lats) < 24:
            continue
        ax_loc.fill(lons, lats, color="#D4EDCC", edgecolor="#8CB88C",
                    linewidth=0.4, zorder=2)

    # Red dashed bounding box for GMRW
    pad = 0.15
    bx0, bx1 = LON_MIN - pad, LON_MAX + pad
    by0, by1 = LAT_MIN - pad, LAT_MAX + pad
    rect = mpatches.FancyBboxPatch(
        (bx0, by0), bx1 - bx0, by1 - by0,
        boxstyle="square,pad=0",
        linewidth=1.5, edgecolor="#CC0000",
        facecolor="#FF888844", zorder=5
    )
    ax_loc.add_patch(rect)

    ax_loc.set_xlim(-125, -65)
    ax_loc.set_ylim(24, 50)
    ax_loc.set_aspect("equal", adjustable="datalim")
    ax_loc.tick_params(labelsize=7.5)
    ax_loc.set_xlabel("Longitude (°)", fontsize=8.5)
    ax_loc.set_ylabel("Latitude (°)",  fontsize=8.5)

    # USA scale bar (approx 345 km = 3° at 38° lat)
    sb0_lon, sb0_lat = -122, 25.5
    ax_loc.annotate("", xy=(sb0_lon + 12, sb0_lat),
                    xytext=(sb0_lon, sb0_lat),
                    arrowprops=dict(arrowstyle="-", color="k", lw=2.5))
    for xv in [sb0_lon, sb0_lon+4, sb0_lon+8, sb0_lon+12]:
        ax_loc.plot(xv, sb0_lat, "k|", lw=1.5, ms=5)
    ax_loc.text(sb0_lon + 6, sb0_lat - 1.0,
                "0      345      690     1380 km",
                ha="center", fontsize=6.5)

    # ── RIGHT PANEL: DEM MAP ──────────────────────────────────────────────────
    ax_dem.set_facecolor("white")  # inactive cells show as white

    lon_edges = np.linspace(LON_MIN, LON_MAX, NCOL + 1)
    lat_edges = np.linspace(LAT_MAX, LAT_MIN, NROW + 1)
    LON_G, LAT_G = np.meshgrid(lon_edges, lat_edges)

    # Build colourmap with NaN → white (so inactive cells are pure white)
    jet_cmap = plt.cm.jet.copy()
    jet_cmap.set_bad(color="white")

    # DEM fill — inactive cells are NaN → white
    pcm = ax_dem.pcolormesh(LON_G, LAT_G, dem_disp,
                             cmap=jet_cmap,
                             vmin=vmin, vmax=vmax,
                             shading="flat", zorder=2)

    # Watershed boundary (red)
    bdy_segs = watershed_boundary_segments(ibound)
    lc = LineCollection(bdy_segs, colors="red", linewidths=1.2, zorder=6)
    ax_dem.add_collection(lc)

    # Observation wells (filled black triangles)
    wx = [w[0] for w in wells]
    wy = [w[1] for w in wells]
    ax_dem.scatter(wx, wy, s=20, marker="^",
                   facecolors="black", edgecolors="white", linewidths=0.3,
                   zorder=8)

    # Scale bar
    add_scale_bar(ax_dem, length_km=50)

    # Axes formatting
    ax_dem.set_xlim(LON_MIN, LON_MAX)
    ax_dem.set_ylim(LAT_MIN, LAT_MAX)
    ax_dem.set_aspect("equal")
    lon_ticks = np.arange(-84.8, -83.5, 0.2)
    lat_ticks = np.arange(39.2, 40.8, 0.2)
    ax_dem.set_xticks(lon_ticks)
    ax_dem.set_xticklabels([f"{v:.1f}°" for v in lon_ticks], fontsize=8)
    ax_dem.set_yticks(lat_ticks)
    ax_dem.set_yticklabels([f"{v:.1f}°" for v in lat_ticks], fontsize=8)
    ax_dem.tick_params(direction="in", top=True, right=True, labelsize=8)
    ax_dem.set_xlabel("Longitude", fontsize=9)
    ax_dem.set_ylabel("Latitude",  fontsize=9)

    # Colorbar (right side of DEM panel)
    cbar = fig.colorbar(pcm, ax=ax_dem, fraction=0.04, pad=0.03, shrink=0.88)
    cbar.set_label("DEM (m)", fontsize=9)
    # Fixed ticks matching reference: 197, 226, 255, 284, 313, 342
    cb_ticks = [197, 226, 255, 284, 313, 342]
    cbar.set_ticks(cb_ticks)
    cbar.ax.tick_params(labelsize=8)

    # Legend overlapping inside upper-right of DEM panel
    legend_elements = [
        Line2D([0], [0], color="red", lw=1.5, label="Watershed boundary"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="black", markeredgecolor="white",
               markersize=7, lw=0, label="Observation well"),
    ]
    ax_dem.legend(handles=legend_elements,
                  loc="upper right",
                  fontsize=8.5, framealpha=0.88, borderpad=0.5,
                  edgecolor="#888888")

    # ── Connecting lines from locator box to DEM panel (Figure coordinates) ──
    # Map the locator box corners in axes coords → Figure coords
    # Then draw lines using fig.lines (transFigure)
    fig.canvas.draw()   # needed to set renderer so we can use transData

    def ax_to_fig(ax, x_data, y_data):
        """Convert data coords in ax → Figure coords."""
        pt = ax.transData.transform((x_data, y_data))
        return fig.transFigure.inverted().transform(pt)

    # Two connecting corners: (top-right of box → top-left of dem)
    #                          (bottom-right of box → bottom-left of dem)
    corners_loc = [(bx1, by1), (bx1, by0)]  # right edge of locator box
    corners_dem = [(LON_MIN, LAT_MAX), (LON_MIN, LAT_MIN)]  # left edge of DEM

    for (xl, yl), (xd, yd) in zip(corners_loc, corners_dem):
        xf_l, yf_l = ax_to_fig(ax_loc, xl, yl)
        xf_d, yf_d = ax_to_fig(ax_dem, xd, yd)
        line = mlines.Line2D(
            [xf_l, xf_d], [yf_l, yf_d],
            transform=fig.transFigure,
            color="red", linewidth=1.0, linestyle="-",
            clip_on=False, zorder=10
        )
        fig.add_artist(line)

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Digital elevation model of the study watershed with its geographic location",
        fontsize=12, fontweight="bold", y=0.97
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data …")
    ibound = load_ibound()
    dem    = load_dem()
    wells  = load_wells()

    active = ibound != 0
    dem_active = dem[active & (dem > 10)]
    print(f"  Active cells  : {active.sum()}")
    print(f"  DEM range     : {dem_active.min():.1f}–{dem_active.max():.1f} m")
    print(f"  Wells         : {len(wells)}")

    print("\nGenerating figure …")
    fig = make_figure(ibound, dem, wells)

    out_png = os.path.join(OUT, "fig02_study_area_map.png")
    out_pdf = os.path.join(OUT, "fig02_study_area_map.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,          bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved:\n   {out_png}\n   {out_pdf}")
