"""
Figure 1 – Publication-quality study area map for the GMRW manuscript.

Layout: main panel with inset location map.
  - Watershed boundary (from IBOUND contour)
  - River network (from MODFLOW .riv cells)
  - 85 groundwater observation wells (from well_name_mapping.csv)
  - USGS streamflow gauge 03274000 at Miamisburg
  - North arrow, scale bar, lat/lon grid

All geometry derived from the MODFLOW grid (geographic coordinates).
No external shapefiles are modified or reprojected.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
NROW, NCOL = 197, 135
LON_MIN, LAT_MIN = -84.855, 39.185
DX, DY = 0.01, 0.01

SWATMF_DIR = Path(r"D:\GMRW\finalresult\swatmf_run")
OUT_DIR = Path(r"D:\GMRW\finalresult\fig")

# USGS gauge 03274000 – Great Miami River at Miamisburg, OH
GAUGE_LON, GAUGE_LAT = -84.2851, 39.6368

# ── Helper: row/col → geographic centre ──────────────────────────────────────
def rc2lonlat(row, col):
    """Convert 0-based row, col to geographic centre of cell."""
    lon = LON_MIN + (col + 0.5) * DX
    lat = LAT_MIN + (NROW - row - 0.5) * DY   # row 0 = top (north)
    return lon, lat

# ── Load IBOUND ──────────────────────────────────────────────────────────────
def load_ibound(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        if "IBOUND" in line.upper():
            start = i + 1
            break
    vals = []
    for line in lines[start:]:
        parts = [x for x in line.split() if x.lstrip('-').isdigit()]
        if not parts:
            continue
        vals.extend(int(x) for x in parts)
        if len(vals) >= NROW * NCOL:
            break
    return np.array(vals[: NROW * NCOL]).reshape((NROW, NCOL))

# ── Load river cells ─────────────────────────────────────────────────────────
def load_river_cells(filepath):
    cells = []
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                r, c = int(parts[1]) - 1, int(parts[2]) - 1
                cells.append((r, c))
    return cells

# ── Load observation wells ───────────────────────────────────────────────────
def load_wells(filepath):
    return pd.read_csv(filepath)

# ── Extract watershed boundary contour ───────────────────────────────────────
def ibound_boundary_coords(ibound):
    """Return lon/lat arrays tracing the outer boundary of active cells."""
    from matplotlib import _contour
    # Create geographic coordinate arrays for contour
    lon_edges = np.linspace(LON_MIN, LON_MIN + NCOL * DX, NCOL + 1)
    lat_edges = np.linspace(LAT_MIN + NROW * DY, LAT_MIN, NROW + 1)  # top→bottom
    return lon_edges, lat_edges

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
ibound = load_ibound(SWATMF_DIR / "modflow_GMRW.bas")
riv_cells = load_river_cells(SWATMF_DIR / "modflow_GMRW.riv")
wells = load_wells(SWATMF_DIR / "well_name_mapping.csv")

# Build geographic grids
lon_centres = LON_MIN + (np.arange(NCOL) + 0.5) * DX
lat_centres = LAT_MIN + (NROW - np.arange(NROW) - 0.5) * DY  # row 0 = north
lon_grid, lat_grid = np.meshgrid(lon_centres, lat_centres)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 10), dpi=300)

# 1. Plot active MODFLOW cells as light fill
active_mask = np.where(ibound != 0, 1.0, np.nan)
ax.pcolormesh(
    np.linspace(LON_MIN, LON_MIN + NCOL * DX, NCOL + 1),
    np.linspace(LAT_MIN, LAT_MIN + NROW * DY, NROW + 1),
    active_mask[::-1],          # flip so row-0 is south in pcolormesh
    cmap=ListedColormap(["#f0f0f0"]),
    edgecolors="none",
    zorder=1,
)

# 2. Watershed boundary (contour at 0.5 on IBOUND → thick black line)
ax.contour(
    lon_grid, lat_grid, ibound.astype(float),
    levels=[0.5], colors="black", linewidths=1.4, zorder=5,
)

# 3. River cells – plot as blue squares (cell size)
riv_lons = np.array([rc2lonlat(r, c)[0] for r, c in riv_cells])
riv_lats = np.array([rc2lonlat(r, c)[1] for r, c in riv_cells])
ax.scatter(riv_lons, riv_lats, s=3, c="#1f77b4", marker="s",
           linewidths=0, zorder=3, label="River cells")

# 4. Observation wells (red triangles)
ax.scatter(wells["lon"], wells["lat"], s=28, c="none",
           edgecolors="#d62728", marker="^", linewidths=0.8,
           zorder=6, label="Observation wells (n = 85)")

# 5. USGS streamflow gauge (large star)
ax.scatter(GAUGE_LON, GAUGE_LAT, s=120, c="#ff7f0e", marker="*",
           edgecolors="black", linewidths=0.6, zorder=7,
           label="USGS gauge 03274000")

# ── Axes & cartographic elements ─────────────────────────────────────────────
ax.set_xlim(LON_MIN - 0.05, LON_MIN + NCOL * DX + 0.05)
ax.set_ylim(LAT_MIN - 0.05, LAT_MIN + NROW * DY + 0.05)
ax.set_xlabel("Longitude (°W)", fontsize=11)
ax.set_ylabel("Latitude (°N)", fontsize=11)
ax.set_aspect("auto")
ax.tick_params(labelsize=9)

# Grid lines
ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, zorder=0)

# Legend
leg = ax.legend(loc="lower left", fontsize=8, framealpha=0.9,
                edgecolor="gray", fancybox=False)

# Scale bar  (approximate: 1° lat ≈ 111 km → at 40°N, 1° lon ≈ 85 km)
# Use ScaleBar with dx in degrees; we supply metres_per_unit
ax.add_artist(ScaleBar(
    dx=111_320,         # metres per degree latitude (approximate)
    units="m",
    location="lower right",
    length_fraction=0.18,
    font_properties={"size": 8},
    box_alpha=0.8,
    sep=3,
))

# North arrow
arrow_x, arrow_y = 0.95, 0.95  # axes fraction
ax.annotate(
    "N",
    xy=(arrow_x, arrow_y), xycoords="axes fraction",
    fontsize=12, fontweight="bold", ha="center", va="top",
)
ax.annotate(
    "",
    xy=(arrow_x, arrow_y - 0.005), xycoords="axes fraction",
    xytext=(arrow_x, arrow_y - 0.06), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
)

# ── Inset map (US location) ──────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax_inset = fig.add_axes([0.12, 0.72, 0.28, 0.22],
                            projection=ccrs.LambertConformal())
    ax_inset.set_extent([-100, -75, 30, 48], crs=ccrs.PlateCarree())
    ax_inset.add_feature(cfeature.LAND, facecolor="#e8e8e8")
    ax_inset.add_feature(cfeature.OCEAN, facecolor="#d4eaf7")
    ax_inset.add_feature(cfeature.STATES, edgecolor="gray", linewidth=0.3)
    ax_inset.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)

    # Watershed location rectangle
    from matplotlib.patches import Rectangle
    import cartopy.crs as ccrs
    rect = Rectangle(
        (LON_MIN, LAT_MIN),
        NCOL * DX, NROW * DY,
        linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.4,
        transform=ccrs.PlateCarree(),
    )
    ax_inset.add_patch(rect)
    ax_inset.set_title("Study area location", fontsize=7, pad=2)
except Exception as e:
    print(f"Inset map skipped: {e}")

# ── Save ──────────────────────────────────────────────────────────────────────
fig.tight_layout()
out_png = OUT_DIR / "fig01_study_area_pub.png"
out_pdf = OUT_DIR / "fig01_study_area_pub.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved:\n  {out_png}\n  {out_pdf}")
