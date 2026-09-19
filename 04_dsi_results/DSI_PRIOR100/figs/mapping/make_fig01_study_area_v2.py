"""
make_fig01_study_area_v2.py
===========================
Publication-quality study area figure — Zhang et al. (2024) style.

  LEFT :  Two stacked panels (no tick labels, clean style):
          (a) USA overview — Ohio highlighted green, red dot for GMRW
          (b) Ohio / regional — GMRW watershed in red
  RIGHT:  (c) Detailed DEM — jet colormap clipped to watershed,
          red boundary, black-triangle wells, scale bar, north arrow

Key changes from previous version:
  - No connecting lines between panels
  - DEM axes tightly fitted to watershed bounding box
  - Legend below the DEM panel (no overlap)
  - Left column has two stacked inset panels
  - Times New Roman throughout, 300 dpi output
"""

import os, re, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid

warnings.filterwarnings("ignore")

# ── Global font: Times New Roman ─────────────────────────────────────────────
matplotlib.rcParams["font.family"]       = "serif"
matplotlib.rcParams["font.serif"]        = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"]  = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Paths & grid constants ───────────────────────────────────────────────────
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


def load_river_cells():
    """Read swatmf_river2grid.txt — 4-line blocks: (reach cell layer, dist, cond, ...)
    Extract MODFLOW cell-id (sequential 1-based) ➜ (lat, lon).
    """
    cells = set()
    path = os.path.join(BASE, "swatmf_river2grid.txt")
    with open(path) as f:
        n = int(f.readline().strip())
        for _ in range(n):
            parts = f.readline().split()        # reach_id  cell_id  layer
            _ = f.readline()                     # distance
            _ = f.readline()                     # conductance
            cell_id = int(parts[1])              # 1-based sequential
            row0 = (cell_id - 1) // NCOL
            col0 = (cell_id - 1) % NCOL
            cells.add((row0, col0))
    lats_lons = []
    for r, c in cells:
        lat = LAT_MAX - (r + 0.5) * DLAT
        lon = LON_MIN + (c + 0.5) * DLON
        lats_lons.append((lat, lon))
    print(f"  River cells: {len(lats_lons)}")
    return lats_lons


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
# 2.  DOWNLOAD NATURAL EARTH STATES
# ─────────────────────────────────────────────────────────────────────────────
def download_states():
    print("  Downloading NE 50m states ...")
    return gpd.read_file(
        "https://naciscdn.org/naturalearth/50m/cultural/"
        "ne_50m_admin_1_states_provinces.zip")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPER: segmented scale bar  &  north arrow
# ─────────────────────────────────────────────────────────────────────────────
def add_scalebar(ax, lon0, lat0, length_km, n_seg=4, fontsize=9):
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat0)))
    total_deg = length_km * deg_per_km
    seg_deg   = total_deg / n_seg
    h = 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for i in range(n_seg):
        colour = "black" if i % 2 == 0 else "white"
        rect = mpatches.Rectangle(
            (lon0 + i * seg_deg, lat0), seg_deg, h,
            facecolor=colour, edgecolor="black", linewidth=0.6, zorder=10)
        ax.add_patch(rect)

    ax.text(lon0, lat0 - h * 1.0, "0", ha="center", va="top",
            fontsize=fontsize - 1, zorder=10)
    ax.text(lon0 + total_deg, lat0 - h * 1.0,
            f"{length_km} km", ha="center", va="top",
            fontsize=fontsize - 1, zorder=10)


def add_north_arrow(ax, x=0.95, y=0.97, size=12):
    ax.annotate("N", xy=(x, y), xycoords="axes fraction",
                fontsize=size, fontweight="bold", ha="center", va="top",
                zorder=10)
    ax.annotate("", xy=(x, y - 0.015), xycoords="axes fraction",
                xytext=(x, y - 0.085), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="fancy", fc="black", ec="black",
                                lw=1.2),
                zorder=10)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load local data ──────────────────────────────────────────────────
    print("Loading MODFLOW data ...")
    ibound   = load_ibound()
    top      = load_top_elevation()
    rivers   = load_river_cells()
    wells    = load_well_positions()
    ws_poly  = ibound_to_polygon(ibound)
    ws_gdf   = gpd.GeoDataFrame(geometry=[ws_poly], crs="EPSG:4326")
    ws_bbox  = ws_poly.bounds          # (minx, miny, maxx, maxy)
    cx, cy   = ws_poly.centroid.x, ws_poly.centroid.y

    # ── Download external data ───────────────────────────────────────────
    print("Downloading Natural Earth datasets ...")
    states_all = download_states()

    us_states = states_all[states_all["admin"] == "United States of America"].copy()
    us_states = us_states.to_crs(epsg=4326)
    exclude = ["Alaska", "Hawaii", "Puerto Rico",
               "United States Virgin Islands",
               "American Samoa", "Guam", "Northern Mariana Islands"]
    conus = us_states[~us_states["name"].isin(exclude)].copy()
    ohio  = conus[conus["name"] == "Ohio"].copy()

    # Neighbouring states for regional panel
    neighbours = ["Ohio", "Indiana", "Kentucky", "West Virginia",
                  "Pennsylvania", "Michigan"]
    regional = conus[conus["name"].isin(neighbours)].copy()

    # ══════════════════════════════════════════════════════════════════════
    #  FIGURE LAYOUT (Zhang et al. style)
    #    ┌──────┬──────────────┐
    #    │ (a)  │              │
    #    │ USA  │     (c)      │
    #    ├──────┤   DEM detail │
    #    │ (b)  │              │
    #    │ Ohio │              │
    #    └──────┴──────────────┘
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(14, 8), facecolor="white", dpi=150)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[0.38, 0.62],
        height_ratios=[1, 1],
        left=0.02, right=0.88, bottom=0.08, top=0.97,
        wspace=0.05, hspace=0.06)

    ax_usa    = fig.add_subplot(gs[0, 0])
    ax_region = fig.add_subplot(gs[1, 0])
    ax_dem    = fig.add_subplot(gs[:, 1])

    # ══════════════════════════════════════════════════════════════════════
    #  (a) USA overview — Ohio in green, red dot for GMRW
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (a): USA overview ...")
    conus.plot(ax=ax_usa, color="#E8E8E8", edgecolor="#AAAAAA", linewidth=0.3)
    ohio.plot(ax=ax_usa, color="#66BB6A", edgecolor="#333333", linewidth=0.6)

    ax_usa.plot(cx, cy, "o", color="red", markersize=5,
                markeredgecolor="darkred", markeredgewidth=0.5, zorder=10)

    # Label "Ohio" with a thin leader line
    oh_cx = ohio.dissolve().centroid.x.values[0]
    oh_cy = ohio.dissolve().centroid.y.values[0]
    ax_usa.annotate("Ohio", xy=(oh_cx, oh_cy),
                    xytext=(oh_cx + 5, oh_cy + 2),
                    fontsize=8, fontstyle="italic", color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.5),
                    ha="left", va="center")

    ax_usa.set_xlim(-126, -66)
    ax_usa.set_ylim(24.5, 50)
    ax_usa.set_aspect("equal")
    ax_usa.set_facecolor("white")
    ax_usa.set_xticklabels([])
    ax_usa.set_yticklabels([])
    ax_usa.tick_params(length=0)
    ax_usa.text(0.03, 0.95, "(a)", transform=ax_usa.transAxes,
                fontsize=13, fontweight="bold", va="top")
    for sp in ax_usa.spines.values():
        sp.set_linewidth(0.7)

    # ══════════════════════════════════════════════════════════════════════
    #  (b) Ohio / regional — GMRW in red
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (b): Ohio / regional ...")
    regional.plot(ax=ax_region, color="#F0F0F0", edgecolor="#999999",
                  linewidth=0.4)
    ohio.plot(ax=ax_region, color="#C8E6C9", edgecolor="#333333",
              linewidth=0.6)
    ws_gdf.plot(ax=ax_region, color="#CC0000", edgecolor="#990000",
                linewidth=1.2, alpha=0.85, zorder=5)

    ax_region.text(cx + 0.35, cy + 0.15, "GMRW", fontsize=8,
                   fontweight="bold", color="#CC0000", zorder=10)

    # Label neighbouring state names
    for _, row in regional.iterrows():
        c = row.geometry.centroid
        oh_bds = ohio.total_bounds
        if oh_bds[0] - 0.5 < c.x < oh_bds[2] + 0.5 and \
           oh_bds[1] - 0.5 < c.y < oh_bds[3] + 0.5:
            ax_region.text(c.x, c.y, row["name"], fontsize=6,
                           ha="center", va="center", color="#777777",
                           fontstyle="italic")

    oh_bds = ohio.total_bounds
    pad_r = 0.5
    ax_region.set_xlim(oh_bds[0] - pad_r, oh_bds[2] + pad_r)
    ax_region.set_ylim(oh_bds[1] - pad_r, oh_bds[3] + pad_r)
    ax_region.set_aspect("equal")
    ax_region.set_facecolor("white")
    ax_region.set_xticklabels([])
    ax_region.set_yticklabels([])
    ax_region.tick_params(length=0)
    ax_region.text(0.03, 0.95, "(b)", transform=ax_region.transAxes,
                   fontsize=13, fontweight="bold", va="top")
    for sp in ax_region.spines.values():
        sp.set_linewidth(0.7)

    # ══════════════════════════════════════════════════════════════════════
    #  (c) Detailed DEM — tightly fitted, no white border
    # ══════════════════════════════════════════════════════════════════════
    print("Panel (c): Detailed DEM ...")

    active = ibound != 0
    elev = top.copy()
    elev[~active] = np.nan
    elev[elev <= 0] = np.nan

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    valid_elev = elev[np.isfinite(elev)]
    vmin = np.nanpercentile(valid_elev, 1)
    vmax = np.nanpercentile(valid_elev, 99)
    print(f"  DEM range: {vmin:.0f} – {vmax:.0f} m")

    cmap_dem = plt.cm.jet.copy()
    cmap_dem.set_bad(color="white", alpha=1.0)
    norm_dem = mcolors.Normalize(vmin=vmin, vmax=vmax)

    ax_dem.imshow(elev, extent=extent, origin="upper",
                  cmap=cmap_dem, norm=norm_dem,
                  interpolation="bilinear", aspect="auto")

    # River cells — blue
    r_lats = [r[0] for r in rivers]
    r_lons = [r[1] for r in rivers]
    ax_dem.scatter(r_lons, r_lats, s=2, c="#1565C0", alpha=0.8,
                   marker="s", zorder=4, linewidths=0)

    # Red watershed boundary
    if hasattr(ws_poly, "geoms"):
        for geom in ws_poly.geoms:
            xs, ys = geom.exterior.xy
            ax_dem.plot(xs, ys, color="#CC0000", linewidth=2.0, zorder=5)
    else:
        xs, ys = ws_poly.exterior.xy
        ax_dem.plot(xs, ys, color="#CC0000", linewidth=2.0, zorder=5)

    # Observation wells — black triangles
    w_lats = [w[0] for w in wells]
    w_lons = [w[1] for w in wells]
    ax_dem.scatter(w_lons, w_lats, s=28, c="black", edgecolors="black",
                   linewidths=0.5, marker="^", zorder=6)

    # ── TIGHT axis limits to watershed boundary ──────────────────────────
    pad_dem = 0.015
    ax_dem.set_xlim(ws_bbox[0] - pad_dem, ws_bbox[2] + pad_dem)
    ax_dem.set_ylim(ws_bbox[1] - pad_dem, ws_bbox[3] + pad_dem)

    # Tick formatting with degree symbols
    from matplotlib.ticker import FuncFormatter
    ax_dem.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{abs(x):.1f}\u00b0W"))
    ax_dem.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y:.1f}\u00b0N"))
    ax_dem.tick_params(labelsize=9)
    ax_dem.set_xlabel("Longitude", fontsize=10)
    ax_dem.set_ylabel("Latitude", fontsize=10)
    ax_dem.text(0.02, 0.98, "(c)", transform=ax_dem.transAxes,
                fontsize=13, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))
    for sp in ax_dem.spines.values():
        sp.set_linewidth(0.8)

    # ── Scale bar (bottom-left inside panel) ─────────────────────────────
    add_scalebar(ax_dem,
                 lon0=ws_bbox[0] + 0.05,
                 lat0=ws_bbox[1] + 0.04,
                 length_km=20, fontsize=9)

    # ── North arrow (top-right inside panel) ─────────────────────────────
    add_north_arrow(ax_dem, x=0.96, y=0.97)

    # ── DEM colourbar (vertical, right of panel) ─────────────────────────
    cbar_ax = fig.add_axes([0.89, 0.15, 0.015, 0.72])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_dem, cmap=cmap_dem),
        cax=cbar_ax, orientation="vertical")
    cbar.set_label("Elevation (m)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # ── Legend — horizontal, BELOW the DEM axes (no overlap with map) ────
    leg_handles = [
        Line2D([0], [0], color="#CC0000", lw=2.0,
               label="Watershed boundary"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
               markeredgecolor="black", markersize=8,
               label="Observation well"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1565C0",
               markeredgecolor="#1565C0", markersize=7,
               label="River cell"),
    ]
    ax_dem.legend(handles=leg_handles,
                  loc="upper center",
                  bbox_to_anchor=(0.5, -0.06),
                  ncol=3, fontsize=9,
                  frameon=True, fancybox=False,
                  edgecolor="#888888", framealpha=1.0,
                  handletextpad=0.5, columnspacing=1.5,
                  borderpad=0.4)

    # ── NO connecting lines (removed per request) ────────────────────────

    # ── SAVE ─────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT, "fig01_study_area_final.png")
    out_pdf = os.path.join(OUT, "fig01_study_area_final.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
