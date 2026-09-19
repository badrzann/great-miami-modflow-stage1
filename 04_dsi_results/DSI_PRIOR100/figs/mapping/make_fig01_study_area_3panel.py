"""
make_fig01_study_area_3panel.py
===============================
Publication-quality 3-panel study area figure for the GMRW paper.

  Panel (a):  USA overview — state boundaries, Ohio highlighted, GMRW bbox
  Panel (b):  Regional zoom — Ohio / western Ohio, glacial aquifer indication, GMRW outline
  Panel (c):  Detailed watershed — DEM hillshade, river network, monitoring wells, boundary

Data sources
------------
  * Natural Earth 110m admin-0 (countries) + 50m admin-1 (states) — downloaded via geopandas
  * MODFLOW DIS file  → TOP elevation (DEM proxy at 913 m)
  * MODFLOW BAS file  → IBOUND active-cell mask → watershed boundary polygon
  * swatmf_river2grid.txt  → river cells
  * modflow.obs       → 85 monitoring-well positions
  * USGS gauge 03274000 location (hard-coded)

Output
------
  D:/GMRW/finalresult/fig/fig01_study_area_3panel.png  (300 dpi)
  D:/GMRW/finalresult/fig/fig01_study_area_3panel.pdf  (vector)

Requirements
------------
  geopandas, shapely, numpy, scipy, matplotlib
"""

import os, sys, re, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  &  GRID CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE   = "D:/GMRW/finalresult/swatmf_run"
OUT    = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

NROW, NCOL = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW

# USGS gauge 03274000 (Great Miami River at Miamisburg)
GAUGE_LAT, GAUGE_LON = 39.6284, -84.2714

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD MODFLOW DATA
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
    return np.array(vals[: NROW * NCOL], dtype=int).reshape(NROW, NCOL)


def load_top_elevation():
    """Read TOP array from MODFLOW DIS file (ground surface elevation, m)."""
    vals = []
    capturing = False
    with open(os.path.join(BASE, "modflow_GMRW.dis")) as f:
        for line in f:
            s = line.strip()
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
    arr = np.array(vals[: NROW * NCOL], dtype=float).reshape(NROW, NCOL)
    print(f"  TOP: min={arr[arr>0].min():.1f}  max={arr.max():.1f} m")
    return arr


def load_river_mask():
    mask = np.zeros((NROW, NCOL), dtype=bool)
    rpath = os.path.join(BASE, "swatmf_river2grid.txt")
    if not os.path.exists(rpath):
        return mask
    with open(rpath) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 2:
            try:
                cell_num = int(parts[1])
                r0 = (cell_num - 1) // NCOL
                c0 = (cell_num - 1) %  NCOL
                if 0 <= r0 < NROW and 0 <= c0 < NCOL:
                    mask[r0, c0] = True
            except (ValueError, IndexError):
                pass
        i += 3
    print(f"  River cells: {mask.sum()}")
    return mask


def load_well_positions():
    """Returns list of (lat, lon) for each well."""
    wells = []
    path = os.path.join(BASE, "modflow.obs")
    with open(path) as f:
        f.readline()
        n = int(f.readline().strip())
        for _ in range(n):
            parts = f.readline().split()
            row0 = int(parts[0]) - 1
            col0 = int(parts[1]) - 1
            lat = LAT_MAX - (row0 + 0.5) * DLAT
            lon = LON_MIN + (col0 + 0.5) * DLON
            wells.append((lat, lon))
    print(f"  Wells: {len(wells)}")
    return wells


def ibound_to_polygon(ibound):
    """Convert IBOUND active mask to a shapely Polygon in lon/lat coordinates."""
    from shapely.geometry import box as sbox
    active = ibound != 0
    polys = []
    for r in range(NROW):
        for c in range(NCOL):
            if active[r, c]:
                x0 = LON_MIN + c * DLON
                y0 = LAT_MAX - (r + 1) * DLAT
                polys.append(sbox(x0, y0, x0 + DLON, y0 + DLAT))
    merged = unary_union(polys)
    # Simplify for cleaner boundary display
    merged = merged.buffer(0.001).buffer(-0.001)
    print(f"  Watershed polygon area: {merged.area:.4f} deg²")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DOWNLOAD NATURAL EARTH DATA  (admin-0, admin-1)
# ─────────────────────────────────────────────────────────────────────────────
def get_usa_states():
    """Download Natural Earth 110m admin-1 states+provinces."""
    url = ("https://naciscdn.org/naturalearth/110m/cultural/"
           "ne_110m_admin_1_states_provinces.zip")
    try:
        states = gpd.read_file(url)
        print(f"  States loaded: {len(states)} features (110m)")
    except Exception:
        # fallback: 50m
        url50 = ("https://naciscdn.org/naturalearth/50m/cultural/"
                 "ne_50m_admin_1_states_provinces.zip")
        states = gpd.read_file(url50)
        print(f"  States loaded: {len(states)} features (50m fallback)")
    return states


def get_countries():
    """Get Natural Earth low-res (~110m) countries from geopandas built-in."""
    return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS: north arrow, scale bar, hillshade
# ─────────────────────────────────────────────────────────────────────────────
def add_north_arrow(ax, x=0.95, y=0.95, size=14):
    ax.annotate("N", xy=(x, y), xycoords="axes fraction",
                ha="center", va="top", fontsize=size, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.annotate("", xy=(x, y - 0.01), xycoords="axes fraction",
                xytext=(x, y - 0.09), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))


def add_scale_bar(ax, lon0, lat0, length_km, label):
    """Draw a simple scale bar at geographic coords (lon0, lat0)."""
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat0)))
    x1 = lon0 + length_km * deg_per_km
    ax.plot([lon0, x1], [lat0, lat0], "k-", lw=2.5, solid_capstyle="butt",
            transform=ax.transData)
    ax.plot([lon0, lon0], [lat0 - 0.01, lat0 + 0.01], "k-", lw=1.5)
    ax.plot([x1, x1],     [lat0 - 0.01, lat0 + 0.01], "k-", lw=1.5)
    ax.text((lon0 + x1) / 2, lat0 + 0.025, label, ha="center", va="bottom",
            fontsize=7, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])


def compute_hillshade(elev, azimuth=315, altitude=45):
    """Compute hillshade from elevation array."""
    elev = elev.astype(float)
    elev[elev == 0] = np.nan
    dx = ndimage.sobel(elev, axis=1)
    dy = ndimage.sobel(elev, axis=0)
    slope = np.sqrt(dx**2 + dy**2)
    aspect = np.arctan2(-dy, dx)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    hs = (np.sin(alt_rad) * np.cos(slope) +
          np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    hs = np.clip(hs, 0, 1)
    return hs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading MODFLOW data ...")
    ibound = load_ibound()
    top    = load_top_elevation()
    rivers = load_river_mask()
    wells  = load_well_positions()
    ws_poly = ibound_to_polygon(ibound)

    print("Downloading Natural Earth data ...")
    states = get_usa_states()
    # Filter to CONUS
    us_states = states[states["admin"] == "United States of America"].copy()
    us_states = us_states.to_crs(epsg=4326)

    # Exclude non-CONUS (AK, HI, PR, etc.)
    exclude = ["Alaska", "Hawaii", "Puerto Rico", "United States Virgin Islands",
               "American Samoa", "Guam", "Northern Mariana Islands"]
    conus = us_states[~us_states["name"].isin(exclude)].copy()

    ohio = conus[conus["name"] == "Ohio"]
    indiana = conus[conus["name"] == "Indiana"]

    # Watershed GeoDataFrame for overlay
    ws_gdf = gpd.GeoDataFrame(geometry=[ws_poly], crs="EPSG:4326")
    ws_bbox = ws_poly.bounds  # (minx, miny, maxx, maxy)

    # ── FIGURE LAYOUT ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6.5), facecolor="white")

    # GridSpec: 1 row, 3 cols with width ratios
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.3],
                           left=0.03, right=0.97, bottom=0.06, top=0.94,
                           wspace=0.08)

    ax_usa    = fig.add_subplot(gs[0])
    ax_region = fig.add_subplot(gs[1])
    ax_detail = fig.add_subplot(gs[2])

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (a):  USA Overview
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (a): USA overview ...")
    conus.plot(ax=ax_usa, color="#F0EDE4", edgecolor="#888888", linewidth=0.3)
    ohio.plot(ax=ax_usa, color="#B8D4E3", edgecolor="#333333", linewidth=0.6)
    indiana.plot(ax=ax_usa, color="#D4E3B8", edgecolor="#333333", linewidth=0.6)

    # GMRW bounding box
    bx = box(ws_bbox[0] - 0.1, ws_bbox[1] - 0.1,
             ws_bbox[2] + 0.1, ws_bbox[3] + 0.1)
    gpd.GeoDataFrame(geometry=[bx], crs="EPSG:4326").boundary.plot(
        ax=ax_usa, color="red", linewidth=1.8, linestyle="--")

    # GMRW filled polygon
    ws_gdf.plot(ax=ax_usa, color="red", alpha=0.5, edgecolor="red", linewidth=0.8)

    ax_usa.set_xlim(-127, -66)
    ax_usa.set_ylim(24, 50)
    ax_usa.set_xlabel("Longitude (°W)", fontsize=8)
    ax_usa.set_ylabel("Latitude (°N)", fontsize=8)
    ax_usa.tick_params(labelsize=7)
    ax_usa.set_title("(a)", fontsize=11, fontweight="bold", loc="left")

    # Legend for Panel (a)
    leg_patches = [
        mpatches.Patch(facecolor="#B8D4E3", edgecolor="#333", label="Ohio"),
        mpatches.Patch(facecolor="#D4E3B8", edgecolor="#333", label="Indiana"),
        mpatches.Patch(facecolor="red", alpha=0.5, edgecolor="red", label="GMRW"),
    ]
    ax_usa.legend(handles=leg_patches, loc="lower left", fontsize=6,
                   frameon=True, fancybox=False, edgecolor="#999",
                   framealpha=0.9)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (b):  Regional Zoom (Ohio + surrounding)
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (b): Regional zoom ...")
    regional_states = conus[conus["name"].isin(
        ["Ohio", "Indiana", "Kentucky", "West Virginia", "Michigan", "Pennsylvania"])]

    regional_states.plot(ax=ax_region, color="#F0EDE4", edgecolor="#666666",
                         linewidth=0.5)
    ohio.plot(ax=ax_region, color="#B8D4E3", edgecolor="#333333", linewidth=0.8)
    indiana.plot(ax=ax_region, color="#D4E3B8", edgecolor="#333333", linewidth=0.8)

    # GMRW polygon
    ws_gdf.plot(ax=ax_region, color="#E8A848", alpha=0.55,
                edgecolor="#CC3300", linewidth=2.0)

    # Label major cities / landmarks
    cities = {
        "Dayton":     (-84.19, 39.76),
        "Cincinnati": (-84.51, 39.10),
        "Columbus":   (-82.99, 39.96),
        "Indianapolis": (-86.16, 39.77),
    }
    for name, (cx, cy) in cities.items():
        ax_region.plot(cx, cy, "ko", markersize=3, zorder=5)
        ax_region.annotate(name, (cx, cy), xytext=(4, 4),
                           textcoords="offset points", fontsize=5.5,
                           fontstyle="italic",
                           path_effects=[pe.withStroke(linewidth=1.5,
                                                       foreground="white")])

    ax_region.set_xlim(-87.5, -80.0)
    ax_region.set_ylim(37.5, 42.5)
    ax_region.set_xlabel("Longitude (°W)", fontsize=8)
    ax_region.set_ylabel("Latitude (°N)", fontsize=8)
    ax_region.tick_params(labelsize=7)
    ax_region.set_title("(b)", fontsize=11, fontweight="bold", loc="left")

    # Legend for Panel (b)
    leg_b = [
        mpatches.Patch(facecolor="#B8D4E3", edgecolor="#333", label="Ohio"),
        mpatches.Patch(facecolor="#D4E3B8", edgecolor="#333", label="Indiana"),
        mpatches.Patch(facecolor="#E8A848", alpha=0.55, edgecolor="#CC3300",
                       linewidth=1.5, label="GMRW"),
    ]
    ax_region.legend(handles=leg_b, loc="lower right", fontsize=6,
                      frameon=True, fancybox=False, edgecolor="#999",
                      framealpha=0.9)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (c):  Detailed Watershed Map
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (c): Detailed watershed ...")

    # --- DEM hillshade + elevation colour ---
    active = ibound != 0
    elev = top.copy()
    elev[~active] = np.nan

    hs = compute_hillshade(top)
    hs[~active] = np.nan

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    # Elevation colourmap
    vmin, vmax = np.nanmin(elev[active]), np.nanmax(elev[active])
    cmap_elev = plt.cm.terrain.copy()
    cmap_elev.set_bad(color="white")
    norm_elev = mcolors.Normalize(vmin=vmin, vmax=vmax)

    ax_detail.imshow(elev, extent=extent, origin="upper",
                     cmap=cmap_elev, norm=norm_elev, alpha=0.7,
                     interpolation="bilinear", aspect="auto")

    # Hillshade overlay
    cmap_hs = plt.cm.gray.copy()
    cmap_hs.set_bad(color="white", alpha=0)
    ax_detail.imshow(hs, extent=extent, origin="upper",
                     cmap=cmap_hs, alpha=0.35,
                     interpolation="bilinear", aspect="auto")

    # --- River network ---
    rr, rc = np.where(rivers)
    r_lons = LON_MIN + (rc + 0.5) * DLON
    r_lats = LAT_MAX - (rr + 0.5) * DLAT
    ax_detail.scatter(r_lons, r_lats, s=1.2, c="#2171B5", alpha=0.8,
                      marker="s", linewidths=0, zorder=3)

    # --- Monitoring wells ---
    w_lats = [w[0] for w in wells]
    w_lons = [w[1] for w in wells]
    ax_detail.scatter(w_lons, w_lats, s=22, c="white", edgecolors="black",
                      linewidths=0.6, marker="^", zorder=5, label="Monitoring wells")

    # --- USGS gauge ---
    ax_detail.scatter([GAUGE_LON], [GAUGE_LAT], s=60, c="red",
                      edgecolors="black", linewidths=0.8, marker="*",
                      zorder=6, label="USGS gauge 03274000")

    # --- Watershed boundary ---
    if isinstance(ws_poly, MultiPolygon):
        for geom in ws_poly.geoms:
            xs, ys = geom.exterior.xy
            ax_detail.plot(xs, ys, color="black", linewidth=1.8, zorder=4)
    else:
        xs, ys = ws_poly.exterior.xy
        ax_detail.plot(xs, ys, color="black", linewidth=1.8, zorder=4)

    # Labels for major rivers
    river_labels = {
        "Great Miami R.":  (-84.30, 40.20),
        "Stillwater R.":   (-84.42, 40.00),
        "Mad R.":          (-83.88, 39.98),
        "Twin Creek":      (-84.58, 39.58),
    }
    for name, (rx, ry) in river_labels.items():
        ax_detail.text(rx, ry, name, fontsize=5.5, color="#0B3D91",
                       fontstyle="italic", fontweight="bold", rotation=0,
                       path_effects=[pe.withStroke(linewidth=1.5,
                                                   foreground="white")],
                       zorder=7)

    # --- Elevation colorbar ---
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_elev, cmap=cmap_elev),
        ax=ax_detail, orientation="vertical", fraction=0.025, pad=0.02,
        shrink=0.8)
    cbar.set_label("Elevation (m asl)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Scale bar & north arrow
    add_scale_bar(ax_detail, LON_MIN + 0.08, LAT_MIN + 0.08, 25, "25 km")
    add_north_arrow(ax_detail, x=0.93, y=0.96, size=11)

    ax_detail.set_xlim(LON_MIN - 0.02, LON_MAX + 0.02)
    ax_detail.set_ylim(LAT_MIN - 0.02, LAT_MAX + 0.02)
    ax_detail.set_xlabel("Longitude (°W)", fontsize=8)
    ax_detail.set_ylabel("Latitude (°N)", fontsize=8)
    ax_detail.tick_params(labelsize=7)
    ax_detail.set_title("(c)", fontsize=11, fontweight="bold", loc="left")

    # Legend for Panel (c)
    leg_c = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=6, label="Monitoring wells (85)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=8, label="USGS gauge"),
        Line2D([0], [0], color="#2171B5", lw=2, label="River network"),
        Line2D([0], [0], color="black", lw=1.8, label="Watershed boundary"),
    ]
    ax_detail.legend(handles=leg_c, loc="lower left", fontsize=5.5,
                     frameon=True, fancybox=False, edgecolor="#999",
                     framealpha=0.92, handletextpad=0.5)

    # ── SAVE ──────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT, "fig01_study_area_3panel.png")
    out_pdf = os.path.join(OUT, "fig01_study_area_3panel.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
