"""
make_fig01_study_area_pro.py
============================
Publication-quality 3-panel study area figure for the GMRW paper.

  Panel (a):  USA overview with rivers, lakes, state boundaries, GMRW highlighted
  Panel (b):  Regional zoom — Ohio / western Ohio with rivers, GMRW boundary
  Panel (c):  Detailed watershed — DEM + hillshade, river network, wells, boundary

Data sources (all open / authoritative)
----------------------------------------
  Natural Earth 10m  — rivers (North America), lakes (North America)
  Natural Earth 50m  — admin-1 states, admin-0 countries
  MODFLOW DIS/BAS    — TOP elevation, IBOUND, river cells, observation wells
"""

import os, re, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import box, MultiPolygon
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

GAUGE_LAT, GAUGE_LON = 39.6284, -84.2714   # USGS 03274000

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
    merged = merged.buffer(0.002).buffer(-0.002)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DOWNLOAD NATURAL EARTH DATA
# ─────────────────────────────────────────────────────────────────────────────
def download_data():
    print("  Downloading NE 10m NA rivers ...")
    rivers = gpd.read_file(
        "https://naciscdn.org/naturalearth/10m/physical/"
        "ne_10m_rivers_north_america.zip")

    print("  Downloading NE 10m NA lakes ...")
    lakes = gpd.read_file(
        "https://naciscdn.org/naturalearth/10m/physical/"
        "ne_10m_lakes_north_america.zip")

    # Also get the global 50m rivers for fallback / Great Miami context
    print("  Downloading NE 50m global rivers ...")
    rivers50 = gpd.read_file(
        "https://naciscdn.org/naturalearth/50m/physical/"
        "ne_50m_rivers_lake_centerlines.zip")

    print("  Downloading NE 50m lakes ...")
    lakes50 = gpd.read_file(
        "https://naciscdn.org/naturalearth/50m/physical/"
        "ne_50m_lakes.zip")

    print("  Downloading NE 50m states ...")
    states = gpd.read_file(
        "https://naciscdn.org/naturalearth/50m/cultural/"
        "ne_50m_admin_1_states_provinces.zip")

    print("  Downloading NE 50m countries ...")
    countries = gpd.read_file(
        "https://naciscdn.org/naturalearth/50m/cultural/"
        "ne_50m_admin_0_countries.zip")

    return rivers, lakes, rivers50, lakes50, states, countries


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def add_north_arrow(ax, x=0.95, y=0.95, size=12):
    ax.annotate("N", xy=(x, y), xycoords="axes fraction",
                ha="center", va="top", fontsize=size, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.annotate("", xy=(x, y - 0.01), xycoords="axes fraction",
                xytext=(x, y - 0.08), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="fancy", lw=1.2, color="black",
                                mutation_scale=12))


def add_scale_bar(ax, lon0, lat0, length_km, label):
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat0)))
    x1 = lon0 + length_km * deg_per_km
    ax.plot([lon0, x1], [lat0, lat0], "k-", lw=3, solid_capstyle="butt")
    ax.plot([lon0, lon0], [lat0 - 0.015, lat0 + 0.015], "k-", lw=1.5)
    ax.plot([x1, x1],     [lat0 - 0.015, lat0 + 0.015], "k-", lw=1.5)
    ax.text((lon0 + x1) / 2, lat0 + 0.035, label, ha="center", va="bottom",
            fontsize=7.5, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def compute_hillshade(elev, azimuth=315, altitude=45):
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
    return np.clip(hs, 0, 1)


def plot_polygon_boundary(ax, poly, **kwargs):
    """Plot exterior boundary of a shapely Polygon or MultiPolygon."""
    if isinstance(poly, MultiPolygon):
        for geom in poly.geoms:
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, **kwargs)
    else:
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load local data ───────────────────────────────────────────────────
    print("Loading MODFLOW data ...")
    ibound  = load_ibound()
    top     = load_top_elevation()
    rivers_mf = load_river_mask()
    wells   = load_well_positions()
    ws_poly = ibound_to_polygon(ibound)
    ws_gdf  = gpd.GeoDataFrame(geometry=[ws_poly], crs="EPSG:4326")
    ws_bbox = ws_poly.bounds

    # ── Download external data ────────────────────────────────────────────
    print("Downloading Natural Earth datasets ...")
    rivers10, lakes10, rivers50, lakes50, states_all, countries = download_data()

    # Filter to CONUS
    us_states = states_all[states_all["admin"] == "United States of America"].copy()
    us_states = us_states.to_crs(epsg=4326)
    exclude = ["Alaska", "Hawaii", "Puerto Rico",
               "United States Virgin Islands",
               "American Samoa", "Guam", "Northern Mariana Islands"]
    conus = us_states[~us_states["name"].isin(exclude)].copy()
    ohio    = conus[conus["name"] == "Ohio"]
    indiana = conus[conus["name"] == "Indiana"]

    # USA outline (CONUS dissolve)
    usa_outline = countries[countries["NAME"] == "United States of America"].to_crs(epsg=4326)

    # Clip rivers & lakes to CONUS extent
    conus_box = box(-130, 23, -64, 51)
    from shapely.validation import make_valid
    rivers10 = rivers10.to_crs(epsg=4326)
    rivers10["geometry"] = rivers10.geometry.apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g)
    lakes10 = lakes10.to_crs(epsg=4326)
    lakes10["geometry"] = lakes10.geometry.apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g)
    rivers50 = rivers50.to_crs(epsg=4326)
    lakes50 = lakes50.to_crs(epsg=4326)
    lakes50["geometry"] = lakes50.geometry.apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g)

    rivers_conus = rivers10.clip(conus_box)
    lakes_conus  = lakes10.clip(conus_box)
    rivers50_conus = rivers50.clip(conus_box)
    lakes50_conus  = lakes50.clip(conus_box)

    # Merge 10m + 50m rivers for completeness
    all_rivers = gpd.GeoDataFrame(
        geometry=list(rivers_conus.geometry) + list(rivers50_conus.geometry),
        crs="EPSG:4326")

    # Great Lakes (large lakes)
    great_lakes = lakes50_conus[lakes50_conus.geometry.area > 0.5]

    # Regional rivers (clip to Ohio region)
    region_box = box(-87.5, 37.5, -79.5, 43.0)
    rivers_regional = all_rivers.clip(region_box)
    lakes_regional  = lakes50_conus.clip(region_box)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE LAYOUT
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 7), facecolor="white")
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 0.9, 1.1],
                           left=0.02, right=0.98, bottom=0.04, top=0.93,
                           wspace=0.06)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ── Colour constants ──────────────────────────────────────────────────
    C_LAND       = "#FAFAFA"       # very light grey land
    C_STATE_EDGE = "#9E9E9E"       # state boundary
    C_COUNTRY    = "#555555"        # country boundary
    C_RIVER      = "#7FBCE6"        # light blue rivers
    C_RIVER_DARK = "#4A90C4"        # slightly darker for major rivers
    C_LAKE       = "#B3D9F2"        # lake fill
    C_LAKE_EDGE  = "#7FBCE6"        # lake edge
    C_OHIO       = "#D6E8F5"        # Ohio highlight (light blue tint)
    C_GMRW_FILL  = "#E85050"        # GMRW red fill
    C_GMRW_EDGE  = "#CC0000"        # GMRW red edge
    C_OCEAN_TXT  = "#7FBCE6"        # ocean label colour

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (a):  CONUS overview  — rivers, lakes, GMRW
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (a): USA overview ...")

    # Land (all states)
    conus.plot(ax=ax_a, color=C_LAND, edgecolor=C_STATE_EDGE, linewidth=0.35)

    # Great Lakes
    great_lakes.plot(ax=ax_a, color=C_LAKE, edgecolor=C_LAKE_EDGE, linewidth=0.3)

    # Rivers — thin light blue
    all_rivers.plot(ax=ax_a, color=C_RIVER, linewidth=0.3, alpha=0.7)
    # Major rivers (50m) slightly thicker
    rivers50_conus.plot(ax=ax_a, color=C_RIVER_DARK, linewidth=0.5, alpha=0.6)

    # Country border (dashed)
    usa_outline.boundary.plot(ax=ax_a, color=C_COUNTRY, linewidth=0.8,
                              linestyle="--", alpha=0.6)

    # Ohio highlight
    ohio.plot(ax=ax_a, color=C_OHIO, edgecolor="#666666", linewidth=0.5)

    # GMRW polygon — bold red
    ws_gdf.plot(ax=ax_a, color=C_GMRW_FILL, alpha=0.85,
                edgecolor=C_GMRW_EDGE, linewidth=1.2, zorder=5)

    # Annotation arrow pointing to GMRW
    gmrw_cx = (ws_bbox[0] + ws_bbox[2]) / 2
    gmrw_cy = (ws_bbox[1] + ws_bbox[3]) / 2
    ax_a.annotate("GMRW", xy=(gmrw_cx, gmrw_cy),
                  xytext=(gmrw_cx + 6, gmrw_cy + 3),
                  fontsize=8, fontweight="bold", color=C_GMRW_EDGE,
                  arrowprops=dict(arrowstyle="->,head_width=0.3",
                                  color=C_GMRW_EDGE, lw=1.2),
                  path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                  zorder=6)

    # Ocean labels
    ax_a.text(-125, 36, "P A C I F I C\nO C E A N", fontsize=7,
              color=C_OCEAN_TXT, fontstyle="italic", rotation=90,
              ha="center", va="center", alpha=0.7)
    ax_a.text(-67.5, 36, "A T L A N T I C\n  O C E A N", fontsize=7,
              color=C_OCEAN_TXT, fontstyle="italic", rotation=270,
              ha="center", va="center", alpha=0.7)

    # State name: Ohio
    ax_a.text(-82.7, 40.3, "Ohio", fontsize=6.5, fontstyle="italic",
              color="#444444",
              path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    ax_a.set_xlim(-127, -65)
    ax_a.set_ylim(24, 50)
    ax_a.set_xlabel("Longitude (°W)", fontsize=8)
    ax_a.set_ylabel("Latitude (°N)", fontsize=8)
    ax_a.tick_params(labelsize=6.5)
    ax_a.set_title("(a)", fontsize=12, fontweight="bold", loc="left")
    ax_a.set_facecolor("white")

    # Thin frame
    for spine in ax_a.spines.values():
        spine.set_linewidth(0.6)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (b):  Regional zoom  — Ohio + surrounding states + rivers
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (b): Regional zoom ...")

    regional_names = ["Ohio", "Indiana", "Kentucky", "West Virginia",
                      "Michigan", "Pennsylvania", "Illinois", "Virginia"]
    regional_states = conus[conus["name"].isin(regional_names)]

    regional_states.plot(ax=ax_b, color=C_LAND, edgecolor=C_STATE_EDGE,
                         linewidth=0.5)
    ohio.plot(ax=ax_b, color=C_OHIO, edgecolor="#555555", linewidth=0.7)

    # Rivers in region
    rivers_regional.plot(ax=ax_b, color=C_RIVER, linewidth=0.45, alpha=0.7)

    # Lakes
    lakes_regional.plot(ax=ax_b, color=C_LAKE, edgecolor=C_LAKE_EDGE,
                        linewidth=0.3)

    # GMRW polygon
    ws_gdf.plot(ax=ax_b, color=C_GMRW_FILL, alpha=0.7,
                edgecolor=C_GMRW_EDGE, linewidth=2.2, zorder=5)

    # Label GMRW
    ax_b.text(gmrw_cx - 0.15, gmrw_cy + 0.55, "GMRW", fontsize=9,
              fontweight="bold", color=C_GMRW_EDGE, ha="center",
              path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
              zorder=6)

    # Key cities
    cities = {
        "Dayton":       (-84.19, 39.76),
        "Cincinnati":   (-84.51, 39.10),
        "Columbus":     (-82.99, 39.96),
        "Indianapolis": (-86.16, 39.77),
    }
    for name, (cx, cy) in cities.items():
        ax_b.plot(cx, cy, "ko", markersize=3, zorder=5)
        ax_b.annotate(name, (cx, cy), xytext=(4, 4),
                       textcoords="offset points", fontsize=6,
                       fontstyle="italic",
                       path_effects=[pe.withStroke(linewidth=2,
                                                   foreground="white")])

    # State labels
    for sname, (sx, sy) in [("Ohio", (-82.5, 40.8)),
                              ("Indiana", (-86.2, 40.4)),
                              ("Kentucky", (-85.5, 38.2)),
                              ("W. Virginia", (-80.5, 38.5))]:
        ax_b.text(sx, sy, sname, fontsize=6, color="#777777",
                  fontstyle="italic", ha="center",
                  path_effects=[pe.withStroke(linewidth=1.5,
                                             foreground="white")])

    # Label Ohio River
    ax_b.text(-84.8, 38.75, "Ohio River", fontsize=5.5, color=C_RIVER_DARK,
              fontstyle="italic", rotation=-15,
              path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    ax_b.set_xlim(-87.5, -79.5)
    ax_b.set_ylim(37.5, 43.0)
    ax_b.set_xlabel("Longitude (°W)", fontsize=8)
    ax_b.set_ylabel("Latitude (°N)", fontsize=8)
    ax_b.tick_params(labelsize=6.5)
    ax_b.set_title("(b)", fontsize=12, fontweight="bold", loc="left")
    ax_b.set_facecolor("white")
    for spine in ax_b.spines.values():
        spine.set_linewidth(0.6)

    # Bounding box for Panel (c) extent shown in Panel (b)
    detail_rect = mpatches.Rectangle(
        (LON_MIN, LAT_MIN), LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
        linewidth=1.5, edgecolor="black", facecolor="none",
        linestyle="-", zorder=6)
    ax_b.add_patch(detail_rect)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (c):  Detailed watershed map  — DEM + hillshade + rivers + wells
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (c): Detailed watershed ...")

    active = ibound != 0
    elev = top.copy()
    elev[~active] = np.nan
    hs = compute_hillshade(top)
    hs[~active] = np.nan

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    # Elevation colour
    vmin = np.nanpercentile(elev[active], 1)
    vmax = np.nanpercentile(elev[active], 99)
    cmap_elev = plt.cm.terrain.copy()
    cmap_elev.set_bad(color="white")
    norm_elev = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot elevation
    ax_c.imshow(elev, extent=extent, origin="upper",
                cmap=cmap_elev, norm=norm_elev, alpha=0.75,
                interpolation="bilinear", aspect="auto")

    # Hillshade overlay
    cmap_hs = plt.cm.gray.copy()
    cmap_hs.set_bad(color="white", alpha=0)
    ax_c.imshow(hs, extent=extent, origin="upper",
                cmap=cmap_hs, alpha=0.30,
                interpolation="bilinear", aspect="auto")

    # River network (MODFLOW river cells)
    rr, rc = np.where(rivers_mf)
    r_lons = LON_MIN + (rc + 0.5) * DLON
    r_lats = LAT_MAX - (rr + 0.5) * DLAT
    ax_c.scatter(r_lons, r_lats, s=2.0, c="#1565C0", alpha=0.85,
                 marker="s", linewidths=0, zorder=3)

    # Monitoring wells
    w_lats = [w[0] for w in wells]
    w_lons = [w[1] for w in wells]
    ax_c.scatter(w_lons, w_lats, s=28, c="white", edgecolors="black",
                 linewidths=0.7, marker="^", zorder=5)

    # USGS gauge
    ax_c.scatter([GAUGE_LON], [GAUGE_LAT], s=80, c="red",
                 edgecolors="black", linewidths=0.8, marker="*", zorder=6)

    # Watershed boundary
    plot_polygon_boundary(ax_c, ws_poly, color="black", linewidth=2.0,
                          zorder=4)

    # River name labels
    river_labels = {
        "Great Miami R.":  (-84.25, 40.22),
        "Stillwater R.":   (-84.55, 40.08),
        "Mad R.":          (-83.85, 39.98),
        "Twin Creek":      (-84.60, 39.55),
    }
    for name, (rx, ry) in river_labels.items():
        ax_c.text(rx, ry, name, fontsize=6, color="#0D47A1",
                  fontstyle="italic", fontweight="bold",
                  path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                  zorder=7)

    # Label Dayton
    ax_c.text(-84.19, 39.73, "Dayton", fontsize=6, color="#333",
              fontstyle="italic",
              path_effects=[pe.withStroke(linewidth=2, foreground="white")],
              zorder=7)

    # Elevation colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_elev, cmap=cmap_elev),
        ax=ax_c, orientation="vertical", fraction=0.025, pad=0.015,
        shrink=0.75)
    cbar.set_label("Elevation (m asl)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    # Scale bar & north arrow
    add_scale_bar(ax_c, LON_MIN + 0.1, LAT_MIN + 0.1, 25, "25 km")
    add_north_arrow(ax_c, x=0.92, y=0.96, size=11)

    ax_c.set_xlim(LON_MIN - 0.03, LON_MAX + 0.03)
    ax_c.set_ylim(LAT_MIN - 0.03, LAT_MAX + 0.03)
    ax_c.set_xlabel("Longitude (°W)", fontsize=8)
    ax_c.set_ylabel("Latitude (°N)", fontsize=8)
    ax_c.tick_params(labelsize=6.5)
    ax_c.set_title("(c)", fontsize=12, fontweight="bold", loc="left")
    for spine in ax_c.spines.values():
        spine.set_linewidth(0.6)

    # Legend
    leg_c = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="Monitoring wells (85)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=10, label="USGS gauge 03274000"),
        Line2D([0], [0], color="#1565C0", lw=2, label="River network"),
        Line2D([0], [0], color="black", lw=2, label="Watershed boundary"),
    ]
    ax_c.legend(handles=leg_c, loc="lower left", fontsize=6,
                frameon=True, fancybox=False, edgecolor="#CCCCCC",
                framealpha=0.92, handletextpad=0.4,
                borderpad=0.5)

    # ── Suptitle (optional) ───────────────────────────────────────────────
    # fig.suptitle("Study Area: Great Miami River Watershed, Ohio, USA",
    #              fontsize=13, fontweight="bold", y=0.98)

    # ── SAVE ──────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT, "fig01_study_area_pro.png")
    out_pdf = os.path.join(OUT, "fig01_study_area_pro.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
