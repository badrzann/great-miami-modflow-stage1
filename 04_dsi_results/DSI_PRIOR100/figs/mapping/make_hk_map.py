"""
make_hk_map.py
==============
Generate a publication-quality HK map for the GMRW paper.

Layout (2 panels, like the DEM reference figure):
  Left  – USA + Ohio locator map with GMRW bounding box
  Right – MODFLOW base HK map with:
            • Proper longitude / latitude axes
            • Discrete jet colormap  (0–200 m/day, step 20)
            • Individual grid cells visible
            • Watershed (active-cell) boundary as red line
            • 85 monitoring wells as filled black triangles
            • Hamilton stream gauge (Sub 5) as a gold star
            • Scale bar (50 km)

Output:
  D:/GMRW/finalresult/fig/fig_hk_map.png   (300 DPI)
  D:/GMRW/finalresult/fig/fig_hk_map.pdf
"""

import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# PATHS AND GRID CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE = "D:/GMRW/finalresult/swatmf_run"
OUT  = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

NROW, NCOL = 197, 135
DELR_M = 913.25926   # column spacing [m]
DELC_M = 918.07107   # row spacing    [m]

# Geographic extent
LON_MIN = -84.855
LON_MAX = -83.585
LAT_MIN =  39.185
LAT_MAX =  40.710

# Derived per-cell step in degrees
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
# 2. LOAD BASE HK FROM UPW
# ─────────────────────────────────────────────────────────────────────────────
def load_hk():
    vals = []
    capturing = False
    with open(os.path.join(BASE, "modflow_GMRW.upw")) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "INTERNAL" in s.upper() and "HK" in s.upper().replace("HDRY", ""):
                capturing = True
                continue
            if capturing:
                if re.match(r'(INTERNAL|EXTERNAL|CONSTANT)\b', s, re.I) and len(vals) > 0:
                    break
                for tok in s.split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
                if len(vals) >= NROW * NCOL:
                    break
    return np.array(vals[:NROW * NCOL], dtype=float).reshape(NROW, NCOL)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD OBSERVATION WELL POSITIONS  (modflow.obs)
# ─────────────────────────────────────────────────────────────────────────────
def load_wells():
    """Returns arrays of lon/lat for all 85 monitoring wells."""
    wells = []
    with open(os.path.join(BASE, "modflow.obs")) as f:
        f.readline()       # header comment
        n = int(f.readline().strip())
        for _ in range(n):
            parts = f.readline().split()
            row0 = int(parts[0]) - 1     # 0-based
            col0 = int(parts[1]) - 1
            lon = LON_MIN + (col0 + 0.5) * DLON
            lat = LAT_MAX - (row0 + 0.5) * DLAT
            wells.append((lon, lat))
    return wells


# ─────────────────────────────────────────────────────────────────────────────
# 4. COORDINATE GRIDS FOR pcolormesh
# ─────────────────────────────────────────────────────────────────────────────
def make_coord_grids():
    """Return 2-D corner-point arrays for pcolormesh (shape NROW+1, NCOL+1)."""
    cols = np.arange(NCOL + 1)
    rows = np.arange(NROW + 1)
    lon_corners = LON_MIN + cols * DLON       # (NCOL+1,)
    lat_corners = LAT_MAX - rows * DLAT       # (NROW+1,)  row0=north
    LON, LAT = np.meshgrid(lon_corners, lat_corners)
    return LON, LAT


# ─────────────────────────────────────────────────────────────────────────────
# 5. WATERSHED BOUNDARY OUTLINE
# ─────────────────────────────────────────────────────────────────────────────
def watershed_boundary_coords(ibound):
    """Extract edge-segments of the active domain and return as list of lon/lat
    polyline segments suitable for LineCollection plotting."""
    active = ibound != 0
    segs = []

    # Horizontal edges (between row r and r+1)
    diff_h = np.diff(active.astype(int), axis=0)          # (196,135)
    for r, c in zip(*np.where(diff_h != 0)):
        lat_y = LAT_MAX - (r + 1) * DLAT
        lon_x0 = LON_MIN + c * DLON
        lon_x1 = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x0, lat_y), (lon_x1, lat_y)])

    # Vertical edges (between col c and c+1)
    diff_v = np.diff(active.astype(int), axis=1)          # (197,134)
    for r, c in zip(*np.where(diff_v != 0)):
        lat_y0 = LAT_MAX - r * DLAT
        lat_y1 = LAT_MAX - (r + 1) * DLAT
        lon_x = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x, lat_y0), (lon_x, lat_y1)])

    # Border edges of active cells on the domain edge
    for c in range(NCOL):
        if active[0, c]:
            lat_y = LAT_MAX
            segs.append([(LON_MIN + c * DLON, lat_y),
                         (LON_MIN + (c+1) * DLON, lat_y)])
        if active[NROW-1, c]:
            lat_y = LAT_MIN
            segs.append([(LON_MIN + c * DLON, lat_y),
                         (LON_MIN + (c+1) * DLON, lat_y)])
    for r in range(NROW):
        if active[r, 0]:
            lon_x = LON_MIN
            segs.append([(lon_x, LAT_MAX - r * DLAT),
                         (lon_x, LAT_MAX - (r+1) * DLAT)])
        if active[r, NCOL-1]:
            lon_x = LON_MAX
            segs.append([(lon_x, LAT_MAX - r * DLAT),
                         (lon_x, LAT_MAX - (r+1) * DLAT)])

    return segs


# ─────────────────────────────────────────────────────────────────────────────
# 6. DRAW USA + OHIO LOCATOR (left panel)
# ─────────────────────────────────────────────────────────────────────────────
def draw_locator(ax):
    """Approximate USA / Ohio locator with GMRW bounding box."""
    # Rough contiguous US bounding filled background
    us_lon = [-124.7, -124.7, -67.0, -67.0, -124.7]
    us_lat = [  24.5,   49.0,  49.0,  24.5,   24.5]
    ax.fill(us_lon, us_lat, color="#c8efb8", alpha=0.9, zorder=1)
    ax.plot(us_lon, us_lat, "k-", lw=0.4, zorder=2)

    # Approximate state outlines (Ohio highlighted)
    states = [
        # Ohio rough polygon
        ([-80.52,-80.52,-81.6,-82.5,-83.1,-83.47,-84.82,-84.82,-83.75,-82.3,-81.5,-80.52],
         [41.98, 40.0,  38.6, 38.4, 38.6,  39.1,  39.1,  41.7,  41.98, 42.32,42.32,41.98],
         "#b7dab3"),
    ]
    for lons, lats, col in states:
        ax.fill(lons, lats, color=col, alpha=1.0, zorder=2)
        ax.plot(lons + [lons[0]], lats + [lats[0]], "k-", lw=0.5, zorder=3)

    # GMRW bounding box
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

    # Scale bar  (approx 345 km = 3° lon at ~38° lat)
    sb_lon0, sb_lat0 = -116, 25.5
    ax.plot([sb_lon0, sb_lon0 + 3], [sb_lat0, sb_lat0], "k-", lw=2, zorder=6)
    for x in [sb_lon0, sb_lon0+1, sb_lon0+2, sb_lon0+3]:
        ax.plot([x], [sb_lat0], "k|", lw=1.5, ms=4, zorder=6)
    ax.text(sb_lon0 + 1.5, sb_lat0 - 0.8, "0  345  690  1380 km",
            ha="center", fontsize=5.5, zorder=6)

    # North arrow
    ax.annotate("N", xy=(-67.5, 47.2), fontsize=8, ha="center",
                fontweight="bold", zorder=6)
    ax.annotate("", xy=(-67.5, 48.5), xytext=(-67.5, 47.5),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2), zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SCALE BAR HELPER  (adds a 50 km bar on the HK panel)
# ─────────────────────────────────────────────────────────────────────────────
def add_scale_bar(ax, length_km=50):
    """Place a 50 km scale bar near the bottom-right of the HK axes."""
    km_per_deg_lon = 111.32 * np.cos(np.radians(39.5))
    bar_deg = length_km / km_per_deg_lon

    # Use current axes limits so it stays inside the tightened view
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    x0 = xhi - 0.06 * (xhi - xlo) - bar_deg
    y0 = ylo + 0.04 * (yhi - ylo)

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
def make_hk_figure(ibound, hk, wells):
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    # Two-panel layout: locator (30%) | HK map (70%)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58],
                          left=0.04, right=0.98, top=0.90, bottom=0.14,
                          wspace=0.06)
    ax_loc = fig.add_subplot(gs[0, 0])
    ax_hk  = fig.add_subplot(gs[0, 1])

    active = ibound != 0

    # ── Left: Locator ────────────────────────────────────────────────────────
    draw_locator(ax_loc)

    # ── Right: HK map ────────────────────────────────────────────────────────
    # Build coordinate grids for pcolormesh
    LON_G, LAT_G = make_coord_grids()

    # HK display array: mask inactive cells so they are not drawn at all
    hk_disp = np.ma.masked_where(~active, hk.astype(float))

    # Discrete jet colormap: 0, 20, 40, …, 200  (11 boundaries → 10 bands)
    hk_levels = np.arange(0, 201, 20)           # [0,20,40,...,200]
    hk_cmap   = plt.cm.jet.copy()
    hk_norm   = BoundaryNorm(hk_levels, ncolors=hk_cmap.N, clip=True)

    ax_hk.set_facecolor("white")

    # HK pcolormesh — masked cells are transparent (deleted)
    pcm = ax_hk.pcolormesh(LON_G, LAT_G, hk_disp,
                            cmap=hk_cmap, norm=hk_norm,
                            shading="flat",
                            linewidth=0, edgecolors="none",
                            zorder=2)

    # Watershed boundary (red outline)
    from matplotlib.collections import LineCollection
    bdy_segs = watershed_boundary_coords(ibound)
    lc = LineCollection(bdy_segs, colors="red", linewidths=1.2,
                        zorder=6, label="Watershed boundary")
    ax_hk.add_collection(lc)

    # Monitoring wells — white fill + black edge for visibility on all HK colours
    wx = [w[0] for w in wells]
    wy = [w[1] for w in wells]
    ax_hk.scatter(wx, wy, s=30, marker="^",
                  facecolors="white", edgecolors="black", linewidths=0.8,
                  zorder=8, label="Observation well")

    # Hamilton gauge (Sub 5) – gold star
    gauge_lon = -84.5719
    gauge_lat =  39.3908
    ax_hk.scatter(gauge_lon, gauge_lat, s=120, marker="*",
                  color="#FFD700", edgecolors="#333333", linewidths=0.6,
                  zorder=9, label="Stream gauge (Hamilton)")

    # Axes cosmetics — tighten to active-cell bounding box
    act_r, act_c = np.where(active)
    pad = 0.02  # degrees padding
    ax_hk.set_xlim(LON_MIN + act_c.min() * DLON - pad,
                   LON_MIN + (act_c.max() + 1) * DLON + pad)
    ax_hk.set_ylim(LAT_MAX - (act_r.max() + 1) * DLAT - pad,
                   LAT_MAX - act_r.min() * DLAT + pad)
    ax_hk.set_aspect("equal")
    ax_hk.set_xlabel("Longitude (°)", fontsize=9)
    ax_hk.set_ylabel("Latitude (°)",  fontsize=9)
    ax_hk.set_title("(b)", fontsize=10, fontweight="bold", loc="left")

    # Longitude ticks every 0.2°
    lon_ticks = np.arange(-84.8, -83.5, 0.2)
    lat_ticks = np.arange(39.2,  40.8,  0.2)
    ax_hk.set_xticks(lon_ticks)
    ax_hk.set_xticklabels([f"{v:.1f}°" for v in lon_ticks], fontsize=8)
    ax_hk.set_yticks(lat_ticks)
    ax_hk.set_yticklabels([f"{v:.1f}°" for v in lat_ticks], fontsize=8)
    ax_hk.tick_params(direction="in", top=True, right=True)

    # Colorbar — discrete, on the right
    cbar = fig.colorbar(pcm, ax=ax_hk, fraction=0.04, pad=0.02, shrink=0.92)
    cbar.set_label("HK (m/day)", fontsize=9)
    cbar.set_ticks(hk_levels)
    cbar.ax.tick_params(labelsize=8)

    # Legend placed OUTSIDE the map axes — below the HK panel, centred
    legend_elements = [
        Line2D([0], [0], color="red", lw=1.5, label="Watershed boundary"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=0.8,
               markersize=8, lw=0, label="Observation well"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
               markeredgecolor="#333333", markersize=10, lw=0,
               label="Stream gauge \u2013 Hamilton (Sub 5)"),
    ]
    ax_hk.legend(handles=legend_elements,
                 loc="upper center",
                 bbox_to_anchor=(0.5, -0.09),
                 ncol=3,
                 fontsize=8.5, framealpha=0.95,
                 borderpad=0.6, edgecolor="#888888")

    # Scale bar
    add_scale_bar(ax_hk, length_km=50)

    # North arrow — use axes limits so it stays in view
    xlo, xhi = ax_hk.get_xlim()
    ylo, yhi = ax_hk.get_ylim()
    na_x = xhi - 0.04 * (xhi - xlo)
    na_y = yhi - 0.06 * (yhi - ylo)
    ax_hk.annotate("N", xy=(na_x, na_y - 0.01),
                   fontsize=10, ha="center", fontweight="bold")
    ax_hk.annotate("", xy=(na_x, na_y + 0.06),
                   xytext=(na_x, na_y - 0.04),
                   arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))

    fig.suptitle(
        "Spatial distribution of hydraulic conductivity (HK) in the GMRW",
        fontsize=11, fontweight="bold", y=0.97
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading IBOUND...")
    ibound = load_ibound()
    print(f"  Active cells: {(ibound != 0).sum()}")

    print("Loading HK...")
    hk = load_hk()
    print(f"  HK: min={hk[ibound!=0].min():.1f}  max={hk[ibound!=0].max():.1f}  m/day")

    print("Loading well positions...")
    wells = load_wells()
    print(f"  Wells: {len(wells)}")

    print("Generating figure...")
    fig = make_hk_figure(ibound, hk, wells)

    out_png = os.path.join(OUT, "fig_hk_map.png")
    out_pdf = os.path.join(OUT, "fig_hk_map.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved:\n   {out_png}\n   {out_pdf}")
