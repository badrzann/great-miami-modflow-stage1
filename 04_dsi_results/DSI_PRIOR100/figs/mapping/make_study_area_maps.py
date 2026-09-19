"""
make_study_area_maps.py
=======================
Generate Figure 1 (study area maps) for the GMRW paper:

  Panel A  – Location map: USA + Ohio inset showing GMRW boundary
  Panel B  – Land-use / vegetation map with subbasin boundaries, wells, gauge
  Panel C  – Soil hydrologic group map on the MODFLOW grid
  Panel D  – MODFLOW base HK map – full 197×135 grid (inactive=grey, active=coloured)

All spatial data are read directly from the SWAT-MODFLOW model files
(no shapefiles required).

Data sources used
-----------------
  finalresult/swatmf_run/modflow_GMRW.bas       -> IBOUND active-cell mask
  finalresult/swatmf_run/modflow_GMRW.upw       -> base HK array (INTERNAL FREE, 2-183 m/d)
  finalresult/swatmf_run/modflow.obs            -> 85 observation well row/col positions
  finalresult/swatmf_run/swatmf_grid2dhru.txt   -> grid-cell → distributed-HRU mapping
  finalresult/swatmf_run/swatmf_dhru2hru.txt    -> distributed-HRU → SWAT HRU mapping
  finalresult/swatmf_run/*.hru                  -> land-use code per HRU
  finalresult/swatmf_run/*.sol                  -> soil hydrologic group per HRU
  finalresult/swatmf_run/swatmf_river2grid.txt  -> river cell locations
"""

import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = "D:/GMRW/finalresult/swatmf_run"
OUT  = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

NROW, NCOL = 197, 135

# Approximate geographic extent of GMRW from DEM figure (SW Ohio)
# (used for axis labels only – the grid is plotted in row/col space)
LON_MIN, LON_MAX = -84.85, -83.55
LAT_MIN, LAT_MAX =  39.20,  40.70

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD IBOUND
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
    arr = np.array(vals[:NROW * NCOL], dtype=int).reshape(NROW, NCOL)
    print(f"  IBOUND: {(arr != 0).sum()} active cells")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD HK FROM UPW  (INTERNAL FREE array)
# ─────────────────────────────────────────────────────────────────────────────
def load_hk():
    vals = []
    capturing = False
    with open(os.path.join(BASE, "modflow_GMRW.upw")) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "INTERNAL" in s.upper() and "HK" in s.upper().replace("HDRY",""):
                capturing = True
                continue
            if capturing:
                # Stop after any subsequent INTERNAL/EXTERNAL keyword for next array
                if re.match(r'(INTERNAL|EXTERNAL|CONSTANT)\b', s, re.I) and len(vals) > 0:
                    break
                for tok in s.split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
                if len(vals) >= NROW * NCOL:
                    break
    arr = np.array(vals[:NROW * NCOL], dtype=float).reshape(NROW, NCOL)
    print(f"  HK: min={arr[arr>0].min():.2f}  max={arr.max():.2f}  m/d (non-zero)")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RIVER MASK
# ─────────────────────────────────────────────────────────────────────────────
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
                c0 = (cell_num - 1) % NCOL
                if 0 <= r0 < NROW and 0 <= c0 < NCOL:
                    mask[r0, c0] = True
            except (ValueError, IndexError):
                pass
        i += 3
    print(f"  River cells: {mask.sum()}")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 4b. LOAD OBSERVATION WELL POSITIONS  (modflow.obs)
# ─────────────────────────────────────────────────────────────────────────────
def load_well_positions():
    """Returns list of (row0, col0) zero-indexed positions for all 85 wells."""
    wells = []
    path = os.path.join(BASE, "modflow.obs")
    with open(path) as f:
        f.readline()          # skip header comment
        n = int(f.readline().strip())
        for _ in range(n):
            parts = f.readline().split()
            row0 = int(parts[0]) - 1   # 1-based → 0-based
            col0 = int(parts[1]) - 1
            wells.append((row0, col0))
    print(f"  Wells loaded: {len(wells)}")
    return wells


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PARSE HRU FILES  → {(subbasin, hru_in_sub): land_use, soil_hyd_group}
# ─────────────────────────────────────────────────────────────────────────────
def parse_hru_attributes():
    """Returns two dicts keyed by (subbasin, hru_in_sub):
       hru_luse  -> SWAT land-use code (str)
       hru_hydgrp-> hydrologic group letter (A/B/C/D)
    """
    hru_luse   = {}
    hru_hydgrp = {}
    hru_files  = glob.glob(os.path.join(BASE, "*.hru"))
    for fp in hru_files:
        fname = os.path.basename(fp)
        # filename = SSSSSHHHHH.hru  (5-digit sub, 4-digit hru-in-sub)
        base = fname.replace(".hru", "")
        if len(base) != 9:
            continue
        try:
            sub = int(base[:5])
            hrn = int(base[5:])
        except ValueError:
            continue
        with open(fp) as f:
            first_line = f.readline()
        m = re.search(r'Luse:(\w+)', first_line)
        if m:
            hru_luse[(sub, hrn)] = m.group(1)

    # Read hydrologic group from .sol files
    sol_files = glob.glob(os.path.join(BASE, "*.sol"))
    for fp in sol_files:
        fname = os.path.basename(fp)
        base  = fname.replace(".sol", "")
        if len(base) != 9:
            continue
        try:
            sub = int(base[:5])
            hrn = int(base[5:])
        except ValueError:
            continue
        hydgrp = "C"  # default
        try:
            with open(fp) as f:
                for line in f:
                    if "Hydrologic Group" in line:
                        m = re.search(r':\s*([ABCD]+)', line)
                        if m:
                            hydgrp = m.group(1)[0]  # take first letter
                        break
        except Exception:
            pass
        hru_hydgrp[(sub, hrn)] = hydgrp

    print(f"  Parsed {len(hru_luse)} HRU land-use entries")
    print(f"  Parsed {len(hru_hydgrp)} HRU soil-hydgrp entries")
    return hru_luse, hru_hydgrp


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PARSE swatmf_dhru2hru.txt → dHRU → dominant (subbasin, hru_in_sub)
# ─────────────────────────────────────────────────────────────────────────────
def parse_dhru2hru():
    """Returns dict: dhru_id (1-based) -> (subbasin, hru_in_sub)
       Uses the HRU with the largest fractional weight.
    """
    dhru2hru = {}
    path = os.path.join(BASE, "swatmf_dhru2hru.txt")
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    header = lines[i].split()
    n_dhru = int(header[0])
    # n_hru_total = int(header[1])  # not needed
    i += 1

    while i < len(lines):
        # Line: dhru_id  n_hrus  subbasin_id
        parts = lines[i].split()
        if len(parts) < 3:
            i += 1
            continue
        try:
            dhru_id = int(parts[0])
            n_hrus  = int(parts[1])
            subbasin = int(parts[2])
        except ValueError:
            i += 1
            continue
        i += 1

        # Next: hru id(s) in subbasin - may span multiple lines
        hru_ids = []
        while len(hru_ids) < n_hrus and i < len(lines):
            for tok in lines[i].split():
                try:
                    hru_ids.append(int(tok))
                except ValueError:
                    pass
            i += 1

        # Next: weight(s)
        weights = []
        while len(weights) < n_hrus and i < len(lines):
            for tok in lines[i].split():
                try:
                    weights.append(float(tok))
                except ValueError:
                    pass
            i += 1

        if hru_ids:
            best_idx = int(np.argmax(weights)) if weights else 0
            dominant_hru = hru_ids[best_idx] if best_idx < len(hru_ids) else hru_ids[0]
            dhru2hru[dhru_id] = (subbasin, dominant_hru)

    print(f"  dhru2hru: {len(dhru2hru)} distributed HRUs mapped")
    return dhru2hru


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PARSE swatmf_grid2dhru.txt → cell_num → dominant dHRU_id
# ─────────────────────────────────────────────────────────────────────────────
def parse_grid2dhru():
    """Returns (cell2dhru, cell2sub):
       cell2dhru[r,c] = dominant dHRU index (0 = inactive)
       cell2sub [r,c] = dominant subbasin number (0 = inactive)
    """
    cell2dhru = np.zeros((NROW, NCOL), dtype=int)
    cell2sub  = np.zeros((NROW, NCOL), dtype=int)
    path = os.path.join(BASE, "swatmf_grid2dhru.txt")
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    header = lines[i].split()
    n_cells = int(header[0])
    i += 1

    for _ in range(n_cells):
        if i >= len(lines):
            break
        parts = lines[i].split()
        if len(parts) < 2:
            i += 1
            continue
        try:
            cell_num = int(parts[0])
            n_d      = int(parts[1])
        except ValueError:
            i += 1
            continue
        i += 1

        r0 = (cell_num - 1) // NCOL
        c0 = (cell_num - 1) % NCOL

        # dhru ids
        dhru_ids = []
        while len(dhru_ids) < n_d and i < len(lines):
            for tok in lines[i].split():
                try:
                    dhru_ids.append(int(tok))
                except ValueError:
                    pass
            i += 1

        # subbasin ids (read and keep)
        sub_ids = []
        while len(sub_ids) < n_d and i < len(lines):
            for tok in lines[i].split():
                try:
                    sub_ids.append(int(tok))
                except ValueError:
                    pass
            i += 1

        # weights
        weights = []
        while len(weights) < n_d and i < len(lines):
            for tok in lines[i].split():
                try:
                    weights.append(float(tok))
                except ValueError:
                    pass
            i += 1

        if dhru_ids and 0 <= r0 < NROW and 0 <= c0 < NCOL:
            best_idx          = int(np.argmax(weights)) if weights else 0
            best_dhru         = dhru_ids[best_idx] if best_idx < len(dhru_ids) else dhru_ids[0]
            best_sub          = sub_ids[best_idx]  if sub_ids and best_idx < len(sub_ids)  else 0
            cell2dhru[r0, c0] = best_dhru
            cell2sub [r0, c0] = best_sub

    print(f"  grid2dhru: {(cell2dhru > 0).sum()} cells mapped")
    return cell2dhru, cell2sub


# ─────────────────────────────────────────────────────────────────────────────
# 7.  BUILD LAND-USE AND SOIL GRIDS
# ─────────────────────────────────────────────────────────────────────────────
# SWAT land-use codes and their display labels / colours
LUSE_INFO = {
    # code        label                       colour (matching SWAT reference palette)
    "AGRL":  ("Agricultural Dist.",           "#5050C8"),   # deep blue-purple like AGGR ref
    "PAST":  ("Pasture / Hay",                "#AACCAA"),   # light sage green
    "FRST":  ("Forest – Mixed",               "#339999"),   # teal-cyan  (like FRST ref)
    "FRSD":  ("Forest – Deciduous",           "#55BB33"),   # bright lime-green
    "FRSE":  ("Forest – Evergreen",           "#CC7722"),   # warm orange-brown
    "RNGE":  ("Range-Grasses",                "#226633"),   # dark forest green
    "RNGB":  ("Range-Brush",                  "#BBCC33"),   # yellow-green
    "WETN":  ("Wetlands – Mixed",             "#CCB077"),   # tan-khaki
    "WETL":  ("Wetlands",                     "#BBAA66"),   # tan variant
    "WATR":  ("Water",                        "#99BBFF"),   # light blue
    "URHD":  ("Urban – High Density",         "#AA22CC"),   # strong purple
    "URMD":  ("Urban – Med. Density",         "#44CCEE"),   # bright cyan
    "URLD":  ("Urban – Low Density",          "#1133AA"),   # dark navy
    "UIDU":  ("Industrial / Comm.",           "#FF4488"),   # hot pink
    "BARR":  ("Barren",                       "#882222"),   # dark red
}
LUSE_CODES = list(LUSE_INFO.keys())
LUSE_COLORS = [LUSE_INFO[k][1] for k in LUSE_CODES]
LUSE_LABELS = [LUSE_INFO[k][0] for k in LUSE_CODES]

HYDGRP_INFO = {
    "A": ("Hydrol. Group A (High Infilt.)",  "#2171B5"),
    "B": ("Hydrol. Group B (Mod. Infilt.)",  "#6BAED6"),
    "C": ("Hydrol. Group C (Slow Infilt.)",  "#FD8D3C"),
    "D": ("Hydrol. Group D (Very Slow)",     "#D7301F"),
}
HYDGRP_CODES  = ["A", "B", "C", "D"]
HYDGRP_COLORS = [HYDGRP_INFO[k][1] for k in HYDGRP_CODES]
HYDGRP_LABELS = [HYDGRP_INFO[k][0] for k in HYDGRP_CODES]


def build_luse_grid(cell2dhru, dhru2hru, hru_luse, ibound):
    grid = np.full((NROW, NCOL), -1, dtype=int)
    for r in range(NROW):
        for c in range(NCOL):
            if ibound[r, c] == 0:
                continue
            dhru = cell2dhru[r, c]
            if dhru == 0:
                continue
            sub_hru = dhru2hru.get(dhru)
            if sub_hru is None:
                continue
            luse = hru_luse.get(sub_hru, "AGRL")
            if luse in LUSE_CODES:
                grid[r, c] = LUSE_CODES.index(luse)
            else:
                grid[r, c] = LUSE_CODES.index("AGRL")
    return grid


def build_soil_grid(cell2dhru, dhru2hru, hru_hydgrp, ibound):
    grid = np.full((NROW, NCOL), -1, dtype=int)
    for r in range(NROW):
        for c in range(NCOL):
            if ibound[r, c] == 0:
                continue
            dhru = cell2dhru[r, c]
            if dhru == 0:
                continue
            sub_hru = dhru2hru.get(dhru)
            if sub_hru is None:
                continue
            grp = hru_hydgrp.get(sub_hru, "C")
            if grp in HYDGRP_CODES:
                grid[r, c] = HYDGRP_CODES.index(grp)
            else:
                grid[r, c] = 2  # default C
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# 7b. BUILD SUBBASIN GRID  (subbasin number per active cell)
# ─────────────────────────────────────────────────────────────────────────────
def build_subbasin_grid(cell2sub, ibound):
    """Returns NROW×NCOL int array: dominant subbasin ID per active cell (0=inactive)."""
    grid = cell2sub.copy()
    grid[ibound == 0] = 0
    return grid


def subbasin_boundary_segments(sub_grid, ibound):
    """Return two LineCollections (horizontal + vertical edges) for inter-subbasin borders."""
    active = ibound != 0
    segs = []

    # Horizontal edges: between row r and r+1
    b = np.diff(sub_grid, axis=0) != 0           # shape (196, 135)
    b &= active[:196, :] & active[1:, :]
    rows, cols = np.where(b)
    for r, c in zip(rows + 1, cols):
        segs.append([(c, r), (c + 1, r)])

    # Vertical edges: between col c and c+1
    b2 = np.diff(sub_grid, axis=1) != 0          # shape (197, 134)
    b2 &= active[:, :134] & active[:, 1:]
    rows2, cols2 = np.where(b2)
    for r, c in zip(rows2, cols2 + 1):
        segs.append([(c, r), (c, r + 1)])

    return segs


def find_gauge_cell():
    """Return (row0, col0) for the HAMILTON gauge (Sub 5) from its known coordinates.
    
    Hamilton, OH stream gauge: Lat 39.3908, Lon -84.5719  (subbasin 5 outlet)
    
    Grid mapping:
      col0 = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * NCOL)
      row0 = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * NROW)
    """
    gauge_lat =  39.3908
    gauge_lon = -84.5719
    col0 = int((gauge_lon - LON_MIN) / (LON_MAX - LON_MIN) * NCOL)
    row0 = int((LAT_MAX - gauge_lat) / (LAT_MAX - LAT_MIN) * NROW)
    # clamp to valid grid range
    col0 = max(0, min(NCOL - 1, col0))
    row0 = max(0, min(NROW - 1, row0))
    print(f"  Hamilton gauge (Sub 5): lat={gauge_lat}, lon={gauge_lon} → "
          f"grid row={row0+1}, col={col0+1}")
    return row0, col0


# ─────────────────────────────────────────────────────────────────────────────
# 8.  COORDINATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def col_to_lon(c):
    return LON_MIN + (c / NCOL) * (LON_MAX - LON_MIN)

def row_to_lat(r):
    # row 0 = north, row NROW-1 = south
    return LAT_MAX - (r / NROW) * (LAT_MAX - LAT_MIN)


def set_geo_axes(ax, title="", fontsize=9):
    """Apply approximate lon/lat tick labels to a grid-indexed axes."""
    col_ticks = np.linspace(0, NCOL, 6)
    row_ticks = np.linspace(0, NROW, 6)
    ax.set_xticks(col_ticks)
    ax.set_xticklabels([f"{col_to_lon(c):.1f}°" for c in col_ticks], fontsize=6)
    ax.set_yticks(row_ticks)
    ax.set_yticklabels([f"{row_to_lat(r):.1f}°" for r in row_ticks], fontsize=6)
    ax.set_xlabel("Longitude", fontsize=fontsize-1)
    ax.set_ylabel("Latitude",  fontsize=fontsize-1)
    ax.set_title(title, fontsize=fontsize, fontweight="bold", loc="left")
    ax.set_xlim(0, NCOL)
    ax.set_ylim(NROW, 0)  # row 0 at top


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DRAW USA + OHIO LOCATOR MAP (no cartopy – uses simple polygons)
# ─────────────────────────────────────────────────────────────────────────────
def draw_locator(ax):
    """Simple Ohio outline using approximate polygon coordinates."""
    # Rough Ohio state boundary
    ohio_lon = [-80.52, -80.52, -81.0, -82.0, -82.7, -83.3,
                -83.45,-83.65,-84.8, -84.82,-84.8, -84.0,
                -83.77,-82.5, -81.5, -80.52]
    ohio_lat = [ 41.98,  40.0,  39.1,  38.4,  38.5,  38.7,
                 39.15,  39.05, 39.1,  39.55, 41.7,  41.96,
                 41.98,  42.33, 42.32, 41.98]

    ax.fill(ohio_lon, ohio_lat, color="#c8e6c9", alpha=0.8, zorder=2)
    ax.plot(ohio_lon, ohio_lat, "k-", lw=0.8, zorder=3)

    # Approximate GMRW bounding box
    rect = mpatches.Rectangle(
        (LON_MIN, LAT_MIN),
        LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
        linewidth=1.5, edgecolor="red", facecolor="#ef9a9a", alpha=0.5, zorder=4
    )
    ax.add_patch(rect)

    # Background (approximate USA extent)
    ax.set_xlim(-95, -74)
    ax.set_ylim(36.5, 45.5)
    ax.set_facecolor("#e3f2fd")

    # Very rough neighbouring state borders (simplified bounding boxes)
    for state_box in [
        (-84.8, 38.0, -82.0, 39.0, "#dcedc8"),   # Kentucky strip
        (-80.5, 39.7, -77.0, 41.5, "#dcedc8"),   # Pennsylvania  strip
        (-84.8, 41.7, -83.0, 42.5, "#dcedc8"),   # Michigan strip
        (-87.5, 37.8, -84.8, 42.5, "#dcedc8"),   # Indiana/Illinois strip
    ]:
        x0, y0, x1, y1, col = state_box
        ar = mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                                 linewidth=0.5, edgecolor="#777",
                                 facecolor=col, alpha=0.5, zorder=1)
        ax.add_patch(ar)

    ax.text(-84.0, 39.8, "GMRW", fontsize=7, color="red",
            fontweight="bold", ha="center", zorder=5)
    ax.text(-82.3, 40.3, "Ohio", fontsize=8, color="#1b5e20",
            ha="center", zorder=4)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude",  fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("(A) Study Area Location", fontsize=9, fontweight="bold", loc="left")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    # North arrow
    ax.annotate("N", xy=(-74.5, 44.8), fontsize=8, ha="center", fontweight="bold")
    ax.annotate("", xy=(-74.5, 45.2), xytext=(-74.5, 44.5),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(ibound, hk, river_mask, luse_grid, soil_grid,
                sub_grid, wells, gauge_cell):
    fig = plt.figure(figsize=(18, 15))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28,
                          left=0.06, right=0.97, top=0.95, bottom=0.04)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    active = ibound != 0

    # Pre-build subbasin boundary segments (shared across panels)
    sub_segs = subbasin_boundary_segments(sub_grid, ibound)

    # river row/col for overlay
    rx, ry = np.where(river_mask & active)

    # ── Panel A: Locator ────────────────────────────────────────────────────
    draw_locator(ax_a)

    # ── Panel B: Land-use map ───────────────────────────────────────────────
    n_luse = len(LUSE_CODES)
    luse_cmap = ListedColormap(LUSE_COLORS)
    luse_norm = BoundaryNorm(boundaries=np.arange(-0.5, n_luse), ncolors=n_luse)

    luse_plot = luse_grid.astype(float)
    luse_plot[~active] = np.nan

    # Grey background for entire domain extent
    bg = np.where(active, 0.0, 1.0)
    ax_b.imshow(bg, cmap="Greys", vmin=0, vmax=1,
                interpolation="nearest", aspect="equal",
                extent=[0, NCOL, NROW, 0], alpha=0.12)

    ax_b.imshow(
        luse_plot, cmap=luse_cmap, norm=luse_norm,
        interpolation="nearest", aspect="equal",
        extent=[0, NCOL, NROW, 0]
    )

    # Subbasin boundaries
    lc_b = LineCollection(sub_segs, colors="#444444", linewidths=0.35,
                           alpha=0.55, zorder=4)
    ax_b.add_collection(lc_b)

    # River overlay
    ax_b.scatter(ry + 0.5, rx + 0.5, s=0.8, c="royalblue",
                 linewidths=0, zorder=5, alpha=0.75)

    # Monitoring wells  (open blue triangles)
    if wells:
        wr = np.array([w[0] for w in wells]) + 0.5
        wc = np.array([w[1] for w in wells]) + 0.5
        ax_b.scatter(wc, wr, s=14, marker="^", edgecolors="#003399",
                     facecolors="none", linewidths=0.7, zorder=8,
                     label="GW Monitoring Well")

    # Stream gauge in subbasin 5 – Hamilton, OH  (gold star)
    if gauge_cell is not None:
        gr, gc = gauge_cell
        ax_b.scatter(gc + 0.5, gr + 0.5, s=90, marker="*",
                     color="#FFD700", edgecolors="#333333", linewidths=0.6,
                     zorder=10, label="Stream Gauge – Hamilton (Sub 5)")

    set_geo_axes(ax_b, title="(B) Land Use / Land Cover")

    # Legend
    present_codes = sorted({LUSE_CODES[int(v)]
                             for v in luse_plot[~np.isnan(luse_plot)]
                             if 0 <= int(v) < n_luse})
    leg_patches = [mpatches.Patch(facecolor=LUSE_INFO[k][1],
                                   label=LUSE_INFO[k][0],
                                   linewidth=0.5, edgecolor="k")
                   for k in present_codes]
    leg_patches += [
        Line2D([0], [0], color="royalblue", lw=1.2, label="Stream Network"),
        Line2D([0], [0], color="#444444", lw=0.8, label="Subbasin Boundary"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="none",
               markeredgecolor="#003399", markersize=5, lw=0,
               label="GW Monitoring Well"),
    ]
    if gauge_cell is not None:
        leg_patches.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
                   markeredgecolor="#333333", markersize=8, lw=0,
                   label="Stream Gauge – Hamilton (Sub 5)")
        )
    ax_b.legend(handles=leg_patches, loc="lower left",
                fontsize=5.0, framealpha=0.88, ncol=1,
                handlelength=1.2, handleheight=0.9, borderpad=0.5)

    # ── Panel C: Soil Hydrologic Group ──────────────────────────────────────
    hydgrp_cmap = ListedColormap(HYDGRP_COLORS)
    hydgrp_norm = BoundaryNorm(boundaries=np.arange(-0.5, 4), ncolors=4)

    soil_plot = soil_grid.astype(float)
    soil_plot[~active] = np.nan

    ax_c.imshow(bg, cmap="Greys", vmin=0, vmax=1,
                interpolation="nearest", aspect="equal",
                extent=[0, NCOL, NROW, 0], alpha=0.12)

    ax_c.imshow(
        soil_plot, cmap=hydgrp_cmap, norm=hydgrp_norm,
        interpolation="nearest", aspect="equal",
        extent=[0, NCOL, NROW, 0]
    )

    # Subbasin boundaries
    lc_c = LineCollection(sub_segs, colors="#444444", linewidths=0.35,
                           alpha=0.55, zorder=4)
    ax_c.add_collection(lc_c)

    ax_c.scatter(ry + 0.5, rx + 0.5, s=0.8, c="royalblue",
                 linewidths=0, zorder=5, alpha=0.75)

    set_geo_axes(ax_c, title="(C) Soil Hydrologic Group")

    hg_patches = [mpatches.Patch(facecolor=HYDGRP_INFO[k][1],
                                  label=HYDGRP_INFO[k][0],
                                  linewidth=0.5, edgecolor="k")
                  for k in HYDGRP_CODES]
    hg_patches += [
        Line2D([0], [0], color="royalblue", lw=1.2, label="Stream Network"),
        Line2D([0], [0], color="#444444", lw=0.8, label="Subbasin Boundary"),
    ]
    ax_c.legend(handles=hg_patches, loc="lower left",
                fontsize=6, framealpha=0.88,
                handlelength=1.2, handleheight=0.9, borderpad=0.5)

    # ── Panel D: MODFLOW Base HK – full 197×135 grid ────────────────────────
    # Inactive cells → light grey; active cells → HK (log-scale colourmap)
    hk_full = hk.copy()
    # Set inactive cells to a sentinel value below vmin so they render as grey
    hk_active = hk_full.copy()
    hk_active[~active] = np.nan
    hk_active[hk_active <= 0] = np.nan

    hk_valid = hk_active[~np.isnan(hk_active)]
    vmin = max(hk_valid.min(), 0.5)
    vmax = hk_valid.max()

    # Grey background for the full 197×135 extent
    full_bg = np.ones((NROW, NCOL))          # 1 = inactive (grey)
    full_bg[active] = np.nan                 # NaN = active (will be overlaid)
    ax_d.imshow(full_bg, cmap=mcolors.ListedColormap(["#DDDDDD"]),
                vmin=0, vmax=2,
                interpolation="nearest", aspect="equal",
                extent=[0, NCOL, NROW, 0], zorder=1)

    im_d = ax_d.imshow(
        hk_active,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="RdYlGn",
        interpolation="nearest", aspect="equal",
        extent=[0, NCOL, NROW, 0], zorder=2
    )

    # Subbasin boundaries on HK panel
    lc_d = LineCollection(sub_segs, colors="#333333", linewidths=0.30,
                           alpha=0.45, zorder=5)
    ax_d.add_collection(lc_d)

    # Grid lines every 15 cells to show cell structure
    step = 15
    for r in range(0, NROW + 1, step):
        ax_d.axhline(r, color="k", lw=0.20, alpha=0.30, zorder=3)
    for c in range(0, NCOL + 1, step):
        ax_d.axvline(c, color="k", lw=0.20, alpha=0.30, zorder=3)

    # River overlay
    ax_d.scatter(ry + 0.5, rx + 0.5, s=0.6, c="royalblue",
                 linewidths=0, zorder=6, alpha=0.70)

    set_geo_axes(ax_d, title="(D) MODFLOW Grid – Base Hydraulic Conductivity HK")

    cb = fig.colorbar(im_d, ax=ax_d, fraction=0.04, pad=0.03, shrink=0.85)
    cb.set_label("HK (m/day, log scale)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax_d.legend(
        [mpatches.Patch(facecolor="#DDDDDD", edgecolor="k", label="Inactive cells"),
         Line2D([0], [0], color="royalblue", lw=1.2, label="Stream Network"),
         Line2D([0], [0], color="#333333", lw=0.8, label="Subbasin Boundary")],
        ["Inactive cells", "Stream Network", "Subbasin Boundary"],
        loc="lower left", fontsize=6, framealpha=0.88
    )

    # ── North arrows on B, C, D ──────────────────────────────────────────────
    for ax in (ax_b, ax_c, ax_d):
        ax.annotate("N", xy=(NCOL - 5, 7), fontsize=8, ha="center",
                    fontweight="bold", color="k", zorder=15)
        ax.annotate("", xy=(NCOL - 5, 3), xytext=(NCOL - 5, 9),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2),
                    zorder=15)

    # ── Super-title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Great Miami River Watershed (GMRW) – Study Area Overview\n"
        "SW Ohio, USA  |  9,911 km²  |  MODFLOW Grid: 197 × 135 cells (~913 m)  |  "
        "HK range: 2–183 m/day  |  85 monitoring wells",
        fontsize=10.5, fontweight="bold", y=0.995
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11.  RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading MODFLOW data...")
    ibound     = load_ibound()
    hk         = load_hk()
    river_mask = load_river_mask()

    print("\nLoading well positions...")
    wells = load_well_positions()

    print("\nParsing HRU attributes...")
    hru_luse, hru_hydgrp = parse_hru_attributes()

    print("\nParsing distributed-HRU mappings...")
    dhru2hru = parse_dhru2hru()

    print("\nParsing grid → distributed-HRU mapping...")
    cell2dhru, cell2sub = parse_grid2dhru()

    print("\nBuilding land-use grid...")
    luse_grid = build_luse_grid(cell2dhru, dhru2hru, hru_luse, ibound)

    print("Building soil hydrologic-group grid...")
    soil_grid = build_soil_grid(cell2dhru, dhru2hru, hru_hydgrp, ibound)

    print("Building subbasin grid...")
    sub_grid = build_subbasin_grid(cell2sub, ibound)
    n_sub = len(np.unique(sub_grid[sub_grid > 0]))
    print(f"  Unique subbasins mapped: {n_sub}")

    gauge_cell = find_gauge_cell()

    print("\nGenerating figure...")
    fig = make_figure(ibound, hk, river_mask, luse_grid, soil_grid,
                      sub_grid, wells, gauge_cell)

    out_png = os.path.join(OUT, "fig01_study_area_maps.png")
    out_pdf = os.path.join(OUT, "fig01_study_area_maps.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n✓ Saved:\n   {out_png}\n   {out_pdf}")
