"""
make_luse_map.py
================
Publication-quality SWAT land-use map for the GMRW paper.

Layout (same 2-panel format as make_hk_map.py):
  Left  – USA + Ohio locator map with GMRW bounding box
  Right – SWAT land-use grid mapped onto MODFLOW IBOUND mask with:
            • White inactive cells
            • Categorical colour legend outside (below) the map
            • Red watershed boundary
            • 85 monitoring wells (white triangle, black outline)
            • Hamilton gauge (gold star)
            • Scale bar (50 km) in free space (bottom-right)
            • Proper lon/lat axes

Data sources:
  finalresult/swatmf_run/modflow_GMRW.bas       -> IBOUND active-cell mask
  finalresult/swatmf_run/modflow.obs            -> 85 observation wells
  finalresult/swatmf_run/swatmf_grid2dhru.txt   -> cell -> distributed-HRU
  finalresult/swatmf_run/swatmf_dhru2hru.txt    -> dHRU -> dominant HRU
  finalresult/swatmf_run/*.hru                  -> land-use code per HRU

Output:
  D:/GMRW/finalresult/fig/fig_luse_map.png  (300 DPI)
  D:/GMRW/finalresult/fig/fig_luse_map.pdf
"""

import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# PATHS AND GRID CONSTANTS  (same as make_hk_map.py)
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
# SOURCE TIF
# ─────────────────────────────────────────────────────────────────────────────
LUSE_TIF = r"D:/GMRW_intraction/linkage/swat model/GMRW/Source/crop/landuse (2).tif"

# ─────────────────────────────────────────────────────────────────────────────
# NLCD COLOUR TABLE  (official USGS NLCD 2019 colours, keyed by raster value)
# ─────────────────────────────────────────────────────────────────────────────
NLCD_INFO = {
    # value  label                              colour (USGS official)
    11: ("Open Water",                        "#476BA0"),
    21: ("Developed – Open Space",            "#DDC9C9"),
    22: ("Developed – Low Intensity",         "#D89382"),
    23: ("Developed – Medium Intensity",      "#ED0000"),
    24: ("Developed – High Intensity",        "#AA0000"),
    31: ("Barren Land",                       "#B2ADA3"),
    41: ("Deciduous Forest",                  "#68AB5F"),
    42: ("Evergreen Forest",                  "#1C6330"),
    43: ("Mixed Forest",                      "#B5C9A1"),
    52: ("Shrub / Scrub",                     "#CCBA7C"),
    71: ("Grassland / Herbaceous",            "#E2E2C1"),
    81: ("Pasture / Hay",                     "#DBD83D"),
    82: ("Cultivated Crops",                  "#AA7028"),
    90: ("Woody Wetlands",                    "#BAD8EA"),
    95: ("Emergent Herbaceous Wetlands",      "#70A3BA"),
}

NLCD_CODES  = list(NLCD_INFO.keys())          # ordered list of int values
NLCD_TO_IDX = {v: i for i, v in enumerate(NLCD_CODES)}
NLCD_COLORS = [NLCD_INFO[k][1] for k in NLCD_CODES]
NLCD_LABELS = [NLCD_INFO[k][0] for k in NLCD_CODES]

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
# 2. LOAD WELLS
# ─────────────────────────────────────────────────────────────────────────────
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
# 3. WARP TIF TO WGS84 AND CLIP TO IBOUND WATERSHED
# ─────────────────────────────────────────────────────────────────────────────
def load_tif_rgba(ibound, out_width=1200):
    """Warp the NLCD TIF to WGS84, build RGBA image, blank inactive cells white.
    Returns (rgba_img, extent, counts) where
      rgba_img : float32 array (out_height × out_width × 4)
      extent   : [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]  for imshow
      counts   : dict {nlcd_value: pixel_count} for present classes
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    # Output grid covering model bbox in WGS84
    lat_span = LAT_MAX - LAT_MIN
    lon_span = LON_MAX - LON_MIN
    cos_mid  = np.cos(np.radians(0.5 * (LAT_MIN + LAT_MAX)))
    out_height = int(out_width * (lat_span / (lon_span * cos_mid)))

    import rasterio.transform as rtransform
    dst_transform = rtransform.from_bounds(
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, out_width, out_height
    )
    dst_crs = "EPSG:4326"

    warped = np.zeros((out_height, out_width), dtype=np.uint8)
    with rasterio.open(LUSE_TIF) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=warped,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=255,
        )

    # Build colour LUT (256 entries, white for unknown/nodata)
    lut = np.ones((256, 4), dtype=np.float32)
    for v, (_, hex_col) in NLCD_INFO.items():
        if v < 256:
            lut[v] = [int(hex_col[1:3],16)/255,
                      int(hex_col[3:5],16)/255,
                      int(hex_col[5:7],16)/255, 1.0]

    rgba = lut[warped]   # (out_height, out_width, 4) — vectorised lookup

    # Mask pixels outside IBOUND watershed → white
    pix_dlon = lon_span / out_width
    pix_dlat = lat_span / out_height
    pix_cols = np.arange(out_width,  dtype=float)
    pix_rows = np.arange(out_height, dtype=float)
    mf_cols = np.clip(((pix_cols * pix_dlon) / DLON).astype(int), 0, NCOL - 1)
    mf_rows = np.clip(((pix_rows * pix_dlat) / DLAT).astype(int), 0, NROW - 1)
    # ibound row 0 = north; image row 0 = north → direct mapping
    active_mask = ibound[mf_rows[:, np.newaxis], mf_cols[np.newaxis, :]] != 0
    rgba[~active_mask] = [1.0, 1.0, 1.0, 1.0]

    # Count NLCD values inside watershed only
    counts = {}
    inside_vals = warped[active_mask]
    for v in inside_vals:
        iv = int(v)
        if iv in NLCD_INFO:
            counts[iv] = counts.get(iv, 0) + 1

    total = sum(counts.values())
    print(f"  Pixels inside watershed: {total}")
    for v, n in sorted(counts.items()):
        print(f"    {v:3d}  {NLCD_INFO[v][0]:35s}  {n:6d}  ({100*n/total:.1f}%)")

    return rgba, [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], counts


# ─────────────────────────────────────────────────────────────────────────────
# 5. WATERSHED BOUNDARY SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
def watershed_boundary_segments(ibound):
    active = ibound != 0
    segs = []
    dh = np.diff(active.astype(int), axis=0)
    for r, c in zip(*np.where(dh != 0)):
        lat_y = LAT_MAX - (r + 1) * DLAT
        segs.append([(LON_MIN + c*DLON, lat_y), (LON_MIN + (c+1)*DLON, lat_y)])
    dv = np.diff(active.astype(int), axis=1)
    for r, c in zip(*np.where(dv != 0)):
        lon_x = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x, LAT_MAX - r*DLAT), (lon_x, LAT_MAX - (r+1)*DLAT)])
    for c in range(NCOL):
        if active[0, c]:
            segs.append([(LON_MIN+c*DLON, LAT_MAX), (LON_MIN+(c+1)*DLON, LAT_MAX)])
        if active[NROW-1, c]:
            segs.append([(LON_MIN+c*DLON, LAT_MIN), (LON_MIN+(c+1)*DLON, LAT_MIN)])
    for r in range(NROW):
        if active[r, 0]:
            segs.append([(LON_MIN, LAT_MAX-r*DLAT), (LON_MIN, LAT_MAX-(r+1)*DLAT)])
        if active[r, NCOL-1]:
            segs.append([(LON_MAX, LAT_MAX-r*DLAT), (LON_MAX, LAT_MAX-(r+1)*DLAT)])
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# 6. LOCATOR PANEL  (identical to make_hk_map.py)
# ─────────────────────────────────────────────────────────────────────────────
def draw_locator(ax):
    us_lon = [-124.7, -124.7, -67.0, -67.0, -124.7]
    us_lat = [  24.5,   49.0,  49.0,  24.5,   24.5]
    ax.fill(us_lon, us_lat, color="#c8efb8", alpha=0.9, zorder=1)
    ax.plot(us_lon, us_lat, "k-", lw=0.4, zorder=2)
    states = [
        ([-80.52,-80.52,-81.6,-82.5,-83.1,-83.47,-84.82,-84.82,-83.75,-82.3,-81.5,-80.52],
         [41.98, 40.0,  38.6, 38.4, 38.6,  39.1,  39.1,  41.7,  41.98, 42.32,42.32,41.98],
         "#b7dab3"),
    ]
    for lons, lats, col in states:
        ax.fill(lons, lats, color=col, alpha=1.0, zorder=2)
        ax.plot(lons + [lons[0]], lats + [lats[0]], "k-", lw=0.5, zorder=3)
    rect = mpatches.Rectangle((LON_MIN, LAT_MIN),
                               LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
                               lw=1.5, edgecolor="red",
                               facecolor="#ff8888", alpha=0.55, zorder=5)
    ax.add_patch(rect)
    ax.set_xlim(-125, -65)
    ax.set_ylim(24, 50)
    ax.set_facecolor("#dbeeff")
    ax.set_xlabel("Longitude (°)", fontsize=8)
    ax.set_ylabel("Latitude (°)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(a)", fontsize=10, fontweight="bold", loc="left")
    sb_lon0, sb_lat0 = -116, 25.5
    ax.plot([sb_lon0, sb_lon0 + 3], [sb_lat0, sb_lat0], "k-", lw=2, zorder=6)
    for x in [sb_lon0, sb_lon0+1, sb_lon0+2, sb_lon0+3]:
        ax.plot([x], [sb_lat0], "k|", lw=1.5, ms=4, zorder=6)
    ax.text(sb_lon0 + 1.5, sb_lat0 - 0.8, "0  345  690  1380 km",
            ha="center", fontsize=5.5, zorder=6)
    ax.annotate("N", xy=(-67.5, 47.2), fontsize=8, ha="center", fontweight="bold", zorder=6)
    ax.annotate("", xy=(-67.5, 48.5), xytext=(-67.5, 47.5),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2), zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SCALE BAR
# ─────────────────────────────────────────────────────────────────────────────
def add_scale_bar(ax, length_km=50):
    km_per_deg_lon = 111.32 * np.cos(np.radians(39.5))
    bar_deg = length_km / km_per_deg_lon
    x0 = LON_MAX - 0.06 * (LON_MAX - LON_MIN) - bar_deg
    y0 = LAT_MIN + 0.04 * (LAT_MAX - LAT_MIN)
    ax.annotate("", xy=(x0 + bar_deg, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-", color="k", lw=2.0),
                annotation_clip=False)
    for x in [x0, x0 + bar_deg]:
        ax.plot(x, y0, "k|", lw=2, ms=5)
    ax.text(x0 + bar_deg / 2, y0 - 0.025, f"{length_km} km",
            ha="center", va="top", fontsize=8, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_luse_figure(ibound, rgba_img, extent, nlcd_counts):
    active = ibound != 0

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58],
                          left=0.04, right=0.98, top=0.93, bottom=0.18,
                          wspace=0.06)
    ax_loc  = fig.add_subplot(gs[0, 0])
    ax_luse = fig.add_subplot(gs[0, 1])

    # ── Left: Locator ────────────────────────────────────────────────────────
    draw_locator(ax_loc)

    # ── Right: Land-use map ───────────────────────────────────────────────────
    ax_luse.set_facecolor("white")

    # Display full-resolution warped TIF (no cell-grid artefacts)
    ax_luse.imshow(rgba_img,
                   extent=extent,          # [lon_min, lon_max, lat_min, lat_max]
                   origin="upper",
                   aspect="auto",
                   interpolation="nearest",
                   zorder=2)

    # Watershed boundary
    bdy = watershed_boundary_segments(ibound)
    lc  = LineCollection(bdy, colors="black", linewidths=1.2, zorder=6)
    ax_luse.add_collection(lc)

    # Scale bar (bottom-right free space)
    add_scale_bar(ax_luse, length_km=50)

    # North arrow (top-right)
    ax_luse.annotate("N", xy=(LON_MAX - 0.07, LAT_MAX - 0.10),
                     fontsize=10, ha="center", fontweight="bold")
    ax_luse.annotate("", xy=(LON_MAX - 0.07, LAT_MAX - 0.04),
                     xytext=(LON_MAX - 0.07, LAT_MAX - 0.13),
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))

    # Axes formatting
    ax_luse.set_xlim(LON_MIN, LON_MAX)
    ax_luse.set_ylim(LAT_MIN, LAT_MAX)
    ax_luse.set_aspect("equal")
    lon_ticks = np.arange(-84.8, -83.5, 0.2)
    lat_ticks = np.arange(39.2,  40.8,  0.2)
    ax_luse.set_xticks(lon_ticks)
    ax_luse.set_xticklabels([f"{v:.1f}°" for v in lon_ticks], fontsize=8)
    ax_luse.set_yticks(lat_ticks)
    ax_luse.set_yticklabels([f"{v:.1f}°" for v in lat_ticks], fontsize=8)
    ax_luse.tick_params(direction="in", top=True, right=True)
    ax_luse.set_xlabel("Longitude (°)", fontsize=9)
    ax_luse.set_ylabel("Latitude (°)",  fontsize=9)
    ax_luse.set_title("(b)", fontsize=10, fontweight="bold", loc="left")

    # ── Legend: outside below, 4 columns ─────────────────────────────────────
    legend_patches = []
    for v in sorted(nlcd_counts.keys()):
        label, hex_col = NLCD_INFO[v]
        legend_patches.append(
            mpatches.Patch(facecolor=hex_col,
                           edgecolor="#555555", linewidth=0.5,
                           label=label)
        )
    legend_patches += [
        Line2D([0], [0], color="black", lw=1.5, label="Watershed boundary"),
    ]

    ax_luse.legend(handles=legend_patches,
                   loc="upper center",
                   bbox_to_anchor=(0.5, -0.10),
                   ncol=4,
                   fontsize=8, framealpha=0.95,
                   borderpad=0.6, edgecolor="#888888",
                   handlelength=1.2, handleheight=1.0)

    fig.suptitle(
        "SWAT land-use classification of the Great Miami River Watershed",
        fontsize=11, fontweight="bold", y=0.99
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading IBOUND …")
    ibound = load_ibound()
    print(f"  Active cells: {(ibound != 0).sum()}")

    print("Warping land-use TIF to WGS84 …")
    rgba_img, extent, nlcd_counts = load_tif_rgba(ibound)

    print("\nGenerating figure …")
    fig = make_luse_figure(ibound, rgba_img, extent, nlcd_counts)

    out_png = os.path.join(OUT, "fig_luse_map.png")
    out_pdf = os.path.join(OUT, "fig_luse_map.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,          bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved:\n   {out_png}\n   {out_pdf}")
