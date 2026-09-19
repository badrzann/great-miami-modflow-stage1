"""
make_fig01_study_area_final.py
==============================
Publication-quality 2-panel study area figure matching reference style.

  LEFT :  CONUS overview — sage-green land, rivers, state boundaries,
          GMRW red polygon with dashed box, connecting lines to detail panel
  RIGHT:  Detailed DEM — jet colormap, red watershed boundary,
          black triangle observation wells, 50 km scale bar

Font: Times New Roman throughout.
Output: 300+ dpi PNG and vector PDF.
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid

warnings.filterwarnings("ignore")

# ── Global font: Times New Roman ──────────────────────────────────────────────
matplotlib.rcParams["font.family"]     = "serif"
matplotlib.rcParams["font.serif"]      = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"          # Times-like math
matplotlib.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  &  GRID CONSTANTS
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
    merged = merged.buffer(0.008).buffer(-0.004)   # smooth jagged cell edges
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DOWNLOAD NATURAL EARTH DATA
# ─────────────────────────────────────────────────────────────────────────────
def download_data():
    print("  Downloading NE 10m NA rivers ...")
    rivers10 = gpd.read_file(
        "https://naciscdn.org/naturalearth/10m/physical/"
        "ne_10m_rivers_north_america.zip")

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

    return rivers10, rivers50, lakes50, states


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SCALE BAR   (segmented publication style)
# ─────────────────────────────────────────────────────────────────────────────
def add_segmented_scalebar(ax, x0, y0, total_km, n_segments, lat_ref,
                           bar_height_deg=0.25, fontsize=8):
    """Draw an alternating black/white segmented scale bar (km)."""
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat_ref)))
    seg_km = total_km / n_segments
    seg_deg = seg_km * deg_per_km

    for i in range(n_segments):
        colour = "black" if i % 2 == 0 else "white"
        rect = mpatches.Rectangle(
            (x0 + i * seg_deg, y0), seg_deg, bar_height_deg,
            facecolor=colour, edgecolor="black", linewidth=0.8, zorder=10)
        ax.add_patch(rect)

    # Tick labels
    for i in range(n_segments + 1):
        xi = x0 + i * seg_deg
        km_val = int(i * seg_km)
        ax.text(xi, y0 - 0.15, str(km_val), ha="center", va="top",
                fontsize=fontsize - 1, zorder=10)

    # "km" label centred below
    ax.text(x0 + total_km * deg_per_km / 2, y0 - 0.9, "km",
            ha="center", va="top", fontsize=fontsize, fontweight="bold",
            zorder=10)


def add_detail_scalebar(ax, lon0, lat0, length_km, fontsize=9):
    """Simple thick black scale bar with end ticks for the detail panel."""
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat0)))
    x1 = lon0 + length_km * deg_per_km
    h = 0.02  # tick half-height in degrees

    ax.plot([lon0, x1], [lat0, lat0], "k-", lw=4, solid_capstyle="butt",
            zorder=10)
    ax.plot([lon0, lon0], [lat0 - h, lat0 + h], "k-", lw=2, zorder=10)
    ax.plot([x1, x1],     [lat0 - h, lat0 + h], "k-", lw=2, zorder=10)
    ax.text((lon0 + x1) / 2, lat0 - 0.06, f"{length_km} km",
            ha="center", va="top", fontsize=fontsize, fontweight="bold",
            zorder=10)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load local data ──────────────────────────────────────────────────
    print("Loading MODFLOW data ...")
    ibound  = load_ibound()
    top     = load_top_elevation()
    wells   = load_well_positions()
    ws_poly = ibound_to_polygon(ibound)
    ws_gdf  = gpd.GeoDataFrame(geometry=[ws_poly], crs="EPSG:4326")
    ws_bbox = ws_poly.bounds  # (minx, miny, maxx, maxy)

    # ── Download external data ───────────────────────────────────────────
    print("Downloading Natural Earth datasets ...")
    rivers10, rivers50, lakes50, states_all = download_data()

    # Filter to CONUS
    us_states = states_all[states_all["admin"] == "United States of America"].copy()
    us_states = us_states.to_crs(epsg=4326)
    exclude = ["Alaska", "Hawaii", "Puerto Rico",
               "United States Virgin Islands",
               "American Samoa", "Guam", "Northern Mariana Islands"]
    conus = us_states[~us_states["name"].isin(exclude)].copy()

    # Clip rivers & lakes
    conus_box = box(-130, 23, -64, 51)

    rivers10 = rivers10.to_crs(epsg=4326)
    rivers10["geometry"] = rivers10.geometry.apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g)
    rivers50 = rivers50.to_crs(epsg=4326)
    lakes50 = lakes50.to_crs(epsg=4326)
    lakes50["geometry"] = lakes50.geometry.apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g)

    rivers_conus   = rivers10.clip(conus_box)
    rivers50_conus = rivers50.clip(conus_box)
    lakes50_conus  = lakes50.clip(conus_box)
    great_lakes    = lakes50_conus[lakes50_conus.geometry.area > 0.5]

    # ══════════════════════════════════════════════════════════════════════
    # COLOUR PALETTE  (matching reference image)
    # ══════════════════════════════════════════════════════════════════════
    C_LAND       = "#C8DCC8"     # sage green land fill
    C_STATE      = "#6B8E6B"     # olive-green state borders
    C_RIVER      = "#6B8E6B"     # dark green rivers (same as state lines)
    C_RIVER_MAJ  = "#4A7A4A"     # slightly darker for 50m major rivers
    C_LAKE       = "#B3D9F2"     # light blue lake fill
    C_LAKE_EDGE  = "#8FBED4"     # lake edge
    C_GMRW       = "#CC0000"     # red for GMRW
    C_BOX        = "#CC666680"   # semi-transparent red box fill

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE LAYOUT — 2 panels side by side
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 7.5), facecolor="white", dpi=150)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0],
                           left=0.02, right=0.91, bottom=0.08, top=0.97,
                           wspace=0.12)
    ax_usa = fig.add_subplot(gs[0])
    ax_dem = fig.add_subplot(gs[1])

    # ══════════════════════════════════════════════════════════════════════
    # LEFT PANEL:  CONUS overview
    # ══════════════════════════════════════════════════════════════════════
    print("Panel LEFT: USA overview ...")

    # Land fill
    conus.plot(ax=ax_usa, color=C_LAND, edgecolor=C_STATE, linewidth=0.4)

    # Great Lakes
    great_lakes.plot(ax=ax_usa, color=C_LAKE, edgecolor=C_LAKE_EDGE,
                     linewidth=0.3)

    # Rivers — green lines matching reference style
    rivers_conus.plot(ax=ax_usa, color=C_RIVER, linewidth=0.25, alpha=0.65)
    rivers50_conus.plot(ax=ax_usa, color=C_RIVER_MAJ, linewidth=0.45, alpha=0.60)

    # GMRW: dashed pink/red bounding box with semi-transparent fill
    pad = 0.8
    box_x0 = ws_bbox[0] - pad
    box_y0 = ws_bbox[1] - pad
    box_w  = (ws_bbox[2] - ws_bbox[0]) + 2 * pad
    box_h  = (ws_bbox[3] - ws_bbox[1]) + 2 * pad

    rect_bg = mpatches.Rectangle(
        (box_x0, box_y0), box_w, box_h,
        facecolor="#CC666620", edgecolor="#CC6666",
        linewidth=1.2, linestyle="--", zorder=4)
    ax_usa.add_patch(rect_bg)

    # GMRW polygon — bold red
    ws_gdf.plot(ax=ax_usa, color=C_GMRW, alpha=0.9,
                edgecolor=C_GMRW, linewidth=1.5, zorder=5)

    # Country outline (dashed dark)
    conus.dissolve().boundary.plot(ax=ax_usa, color="#333333",
                                  linewidth=0.9, linestyle="--")

    ax_usa.set_xlim(-127, -65)
    ax_usa.set_ylim(24, 50)
    ax_usa.set_aspect("equal")
    ax_usa.set_facecolor("white")
    ax_usa.tick_params(labelsize=8)
    ax_usa.set_xlabel("")
    ax_usa.set_ylabel("")
    # Remove axis tick labels for cleaner look (like reference)
    ax_usa.set_xticklabels([])
    ax_usa.set_yticklabels([])
    ax_usa.tick_params(length=0)

    for spine in ax_usa.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#444444")

    # Segmented scale bar at bottom-left (0 — 345 — 690 — 1380 km)
    add_segmented_scalebar(ax_usa, x0=-124, y0=25.5,
                           total_km=1380, n_segments=4,
                           lat_ref=33, bar_height_deg=0.35, fontsize=8)

    # ── Connecting lines removed per request ─────────────────────────────

    # ══════════════════════════════════════════════════════════════════════
    # RIGHT PANEL:  Detailed DEM map
    # ══════════════════════════════════════════════════════════════════════
    print("Panel RIGHT: Detailed DEM ...")

    active = ibound != 0
    elev = top.copy()
    elev[~active] = np.nan
    elev[elev <= 0] = np.nan

    # Fill boundary-adjacent gaps: expand DEM 3 cells outward by nearest-
    # neighbour so colour fills right up to (and slightly past) the boundary
    mask_nan = np.isnan(elev)
    for _ in range(5):                       # expand enough to fill past boundary
        filled = elev.copy()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                shifted = np.roll(np.roll(elev, dr, axis=0), dc, axis=1)
                update = mask_nan & np.isfinite(shifted)
                filled[update] = shifted[update]
        elev = filled
        mask_nan = np.isnan(elev)

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    # DEM range — from originally active cells only
    valid_elev = top[active & (top > 0)]
    vmin = np.nanpercentile(valid_elev, 1)
    vmax = np.nanpercentile(valid_elev, 99)
    print(f"  DEM range: {vmin:.0f} – {vmax:.0f} m")

    cmap_dem = plt.cm.jet.copy()
    cmap_dem.set_bad(color="white", alpha=0)
    norm_dem = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot DEM and clip it to the watershed polygon
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    if hasattr(ws_poly, 'geoms'):
        ws_coords = np.array(ws_poly.geoms[0].exterior.coords)
    else:
        ws_coords = np.array(ws_poly.exterior.coords)

    ws_path = MplPath(ws_coords)
    ws_clip_patch = PathPatch(ws_path, transform=ax_dem.transData,
                              facecolor='none', edgecolor='none')
    ax_dem.add_patch(ws_clip_patch)

    im = ax_dem.imshow(elev, extent=extent, origin="upper",
                       cmap=cmap_dem, norm=norm_dem,
                       interpolation="bilinear", aspect="auto")
    im.set_clip_path(ws_clip_patch)

    # Red watershed boundary (bold)
    if hasattr(ws_poly, "geoms"):
        for geom in ws_poly.geoms:
            xs, ys = geom.exterior.xy
            ax_dem.plot(xs, ys, color=C_GMRW, linewidth=1.2, zorder=5)
    else:
        xs, ys = ws_poly.exterior.xy
        ax_dem.plot(xs, ys, color=C_GMRW, linewidth=1.2, zorder=5)

    # Observation wells — black triangles (matching reference)
    w_lats = [w[0] for w in wells]
    w_lons = [w[1] for w in wells]
    ax_dem.scatter(w_lons, w_lats, s=28, c="black", edgecolors="black",
                   linewidths=0.5, marker="^", zorder=6)

    # DEM colourbar (vertical, right side — matching reference)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.72])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_dem, cmap=cmap_dem),
        cax=cbar_ax, orientation="vertical")
    cbar.set_label("DEM (m)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=8.5)

    # Scale bar: 50 km
    add_detail_scalebar(ax_dem,
                        lon0=ws_bbox[0] + 0.15,
                        lat0=ws_bbox[1] + 0.12,
                        length_km=50, fontsize=9)

    # Axis labels
    ax_dem.set_xlabel("Longitude", fontsize=10)
    ax_dem.set_ylabel("Latitude", fontsize=10)
    ax_dem.tick_params(labelsize=8.5)

    # Format tick labels with degree symbol (show negative longitude)
    from matplotlib.ticker import FuncFormatter
    ax_dem.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{x:.1f}\u00b0"))
    ax_dem.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: f"{y:.1f}\u00b0"))

    # Tight fit to watershed boundary (no white border)
    ax_dem.set_xlim(ws_bbox[0] - 0.015, ws_bbox[2] + 0.015)
    ax_dem.set_ylim(ws_bbox[1] - 0.015, ws_bbox[3] + 0.015)
    for spine in ax_dem.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#333333")

    # Legend (upper-right inside panel — positioned to avoid colorbar)
    leg = [
        Line2D([0], [0], color=C_GMRW, lw=1.2, label="Watershed boundary"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
               markeredgecolor="black", markersize=8,
               label="Observation well"),
    ]
    ax_dem.legend(handles=leg, loc="upper right", fontsize=8.5,
                  frameon=True, fancybox=False, edgecolor="#999999",
                  framealpha=0.95, handletextpad=0.6, borderpad=0.6,
                  bbox_to_anchor=(0.98, 0.98))

    # ── SAVE ─────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT, "fig01_study_area_final.png")
    out_pdf = os.path.join(OUT, "fig01_study_area_final.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
