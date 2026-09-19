"""
make_soil_map.py
================
Publication-quality SWAT soil map for the GMRW paper.

Single-panel map showing SSURGO soil texture distribution:
  * Sandy Clay Loam / Group C  (Af17-1-2a-2)  -- 25.9 % of watershed
  * Sandy Clay Loam / Group D  (Af32-2ab-3)   -- 74.0 % of watershed
  * Loam / Group C             (Ao39-2b-4)    -- < 0.1 % of watershed

Nodata gaps inside the watershed are filled by nearest-valid-neighbour.
Inactive IBOUND cells are shown as white.

Data sources
------------
  finalresult/swatmf_run/modflow_GMRW.bas          -> IBOUND active-cell mask
  GMRW_intraction/.../Source/soil/soil_new.tif     -> SSURGO soil raster (EPSG:26916)
    raster values: 2 = Af17-1-2a-2 | 3 = Af32-2ab-3 | 4 = Ao39-2b-4

Output
------
  D:/GMRW/finalresult/fig/fig_soil_map.png   (300 DPI)
  D:/GMRW/finalresult/fig/fig_soil_map.pdf
"""

import os
import numpy as np
import rasterio
import rasterio.transform as rtransform
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# PATHS AND GRID CONSTANTS
# ---------------------------------------------------------------------------
BASE = "D:/GMRW/finalresult/swatmf_run"
OUT  = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

SOIL_TIF = (
    r"D:/GMRW_intraction/linkage/swat model/GMRW/Source/soil/soil_new.tif"
)

NROW, NCOL = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW

DEG = chr(176)

# ---------------------------------------------------------------------------
# SOIL CLASSIFICATION TABLE
# value : (texture_label, hydrologic_group, legend_label, hex_color)
# Texture from SWAT .sol files:
#   2 = Af17-1-2a-2 -> SANDY_CLAY_LOAM, Group C
#   3 = Af32-2ab-3  -> SANDY_CLAY_LOAM, Group D
#   4 = Ao39-2b-4   -> LOAM,            Group C
# ---------------------------------------------------------------------------
SOIL_INFO = {
    2: ("Sandy Clay Loam", "C", "Sandy Clay Loam  (Hydrologic Group C)", "#E8C87E"),
    3: ("Sandy Clay Loam", "D", "Sandy Clay Loam  (Hydrologic Group D)", "#8B4513"),
    4: ("Loam",            "C", "Loam             (Hydrologic Group C)", "#5D8A4E"),
}
NODATA_VAL = -9999
WHITE = [1.0, 1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# 1. LOAD IBOUND
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. WARP SOIL TIF TO WGS84 AND BUILD RGBA IMAGE
# ---------------------------------------------------------------------------
def load_soil_rgba(ibound, out_width=1200):
    """Warp soil_new.tif to WGS84, fill nodata gaps, build RGBA image."""

    lat_span = LAT_MAX - LAT_MIN
    lon_span = LON_MAX - LON_MIN
    cos_mid  = np.cos(np.radians(0.5 * (LAT_MIN + LAT_MAX)))
    out_height = int(out_width * (lat_span / (lon_span * cos_mid)))

    dst_transform = rtransform.from_bounds(
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, out_width, out_height
    )
    dst_crs = "EPSG:4326"

    warped = np.full((out_height, out_width), NODATA_VAL, dtype=np.int16)
    with rasterio.open(SOIL_TIF) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=warped,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=NODATA_VAL,
        )

    # Build active-pixel mask (same pixel grid as warped)
    pix_dlon = lon_span / out_width
    pix_dlat = lat_span / out_height
    pix_cols = np.arange(out_width,  dtype=float)
    pix_rows = np.arange(out_height, dtype=float)
    mf_cols = np.clip((pix_cols * pix_dlon / DLON).astype(int), 0, NCOL - 1)
    mf_rows = np.clip((pix_rows * pix_dlat / DLAT).astype(int), 0, NROW - 1)
    active_mask = ibound[mf_rows[:, np.newaxis], mf_cols[np.newaxis, :]] != 0

    # ---- Nearest-neighbour fill for nodata pixels inside watershed ----------
    nodata_inside = (warped == NODATA_VAL) & active_mask
    if nodata_inside.any():
        print("  Filling %d nodata pixels inside watershed..." % nodata_inside.sum())
        # valid = known soil value, inside OR outside watershed
        valid_mask = warped != NODATA_VAL
        # distance_transform_edt returns indices of nearest valid pixel
        _, nearest_idx = distance_transform_edt(
            ~valid_mask, return_indices=True
        )
        filled = warped.copy()
        fill_rows, fill_cols = np.where(nodata_inside)
        filled[fill_rows, fill_cols] = warped[
            nearest_idx[0][fill_rows, fill_cols],
            nearest_idx[1][fill_rows, fill_cols],
        ]
        warped = filled

    # Build colour LUT  (values 0-255; soil values are 2/3/4)
    lut = np.ones((256, 4), dtype=np.float32)   # default white
    for v, (_sn, _sg, _lbl, hex_col) in SOIL_INFO.items():
        if 0 <= v < 256:
            r = int(hex_col[1:3], 16) / 255.0
            g = int(hex_col[3:5], 16) / 255.0
            b = int(hex_col[5:7], 16) / 255.0
            lut[v] = [r, g, b, 1.0]

    # Clip raster values to [0, 255] for LUT indexing (nodata -> 0 -> white)
    idx = np.clip(warped.astype(np.int32), 0, 255)
    rgba = lut[idx]   # (out_height, out_width, 4)

    # Mask pixels outside active IBOUND -> white
    rgba[~active_mask] = WHITE

    # Print coverage statistics
    print("Soil type distribution inside watershed:")
    inside_vals = warped[active_mask]
    total = len(inside_vals)
    from collections import Counter
    counts = Counter(inside_vals.tolist())
    for v in sorted(SOIL_INFO.keys()):
        sn, sg, lbl, _ = SOIL_INFO[v]
        n = counts.get(v, 0)
        pct = 100.0 * n / total if total > 0 else 0.0
        print("  %4d  %-22s  Group %s  %7d px  (%5.1f%%)" % (v, sn, sg, n, pct))
    n_nd = counts.get(NODATA_VAL, 0) + counts.get(0, 0)
    if n_nd:
        print("  Unfilled nodata pixels inside watershed: %d" % n_nd)

    return rgba, [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]


# ---------------------------------------------------------------------------
# 3. WATERSHED BOUNDARY SEGMENTS
# ---------------------------------------------------------------------------
def watershed_boundary_segments(ibound):
    active = ibound != 0
    segs = []
    dh = np.diff(active.astype(int), axis=0)
    for r, c in zip(*np.where(dh != 0)):
        lat_y = LAT_MAX - (r + 1) * DLAT
        segs.append([(LON_MIN + c * DLON, lat_y), (LON_MIN + (c + 1) * DLON, lat_y)])
    dv = np.diff(active.astype(int), axis=1)
    for r, c in zip(*np.where(dv != 0)):
        lon_x = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x, LAT_MAX - r * DLAT), (lon_x, LAT_MAX - (r + 1) * DLAT)])
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


# ---------------------------------------------------------------------------
# 4. DRAW FIGURE
# ---------------------------------------------------------------------------
def make_soil_figure(ibound):
    print("Warping soil raster to WGS84...")
    rgba, extent = load_soil_rgba(ibound)

    print("Building figure...")
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.14)
    ax.set_facecolor("white")

    ax.imshow(rgba, origin="upper", extent=extent, aspect="auto",
              interpolation="nearest", zorder=2)

    # Watershed boundary
    bdy = watershed_boundary_segments(ibound)
    ax.add_collection(LineCollection(bdy, colors="black", linewidths=1.4, zorder=5))

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")

    lon_ticks = np.arange(-84.8, -83.5, 0.2)
    lat_ticks = np.arange(39.2,  40.8,  0.2)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels(["%.1f%s" % (v, DEG) for v in lon_ticks], fontsize=9)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels(["%.1f%s" % (v, DEG) for v in lat_ticks], fontsize=9)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("Longitude (%s)" % DEG, fontsize=10)
    ax.set_ylabel("Latitude (%s)"  % DEG, fontsize=10)

    # Legend
    legend_patches = []
    for v in sorted(SOIL_INFO.keys()):
        sn, sg, lbl, hex_col = SOIL_INFO[v]
        legend_patches.append(
            mpatches.Patch(facecolor=hex_col, edgecolor="#555555",
                           linewidth=0.6, label=lbl)
        )
    # Add watershed boundary entry
    from matplotlib.lines import Line2D
    legend_patches.append(
        Line2D([0], [0], color="black", lw=1.5, label="Watershed boundary")
    )
    ax.legend(handles=legend_patches,
              loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=2, fontsize=9, framealpha=0.95,
              borderpad=0.7, edgecolor="#888888",
              handlelength=1.4, handleheight=1.0)

    fig.suptitle("SSURGO Soil Classification -- Great Miami River Watershed",
                 fontsize=11, fontweight="bold", y=0.97)
    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading IBOUND...")
    ibound = load_ibound()
    n_act = int((ibound != 0).sum())
    print("  Active cells: %d" % n_act)

    fig = make_soil_figure(ibound)

    out_png = os.path.join(OUT, "fig_soil_map.png")
    out_pdf = os.path.join(OUT, "fig_soil_map.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,          bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: %s" % out_png)
