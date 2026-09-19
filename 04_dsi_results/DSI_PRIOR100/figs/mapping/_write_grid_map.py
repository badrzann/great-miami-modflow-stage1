"""Helper: writes a clean ASCII-safe make_grid_map.py"""
script = """\
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

BASE = "D:/GMRW/finalresult/swatmf_run"
OUT  = "D:/GMRW/finalresult/fig"
os.makedirs(OUT, exist_ok=True)

NROW, NCOL = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW

COL_ACTIVE   = "#555555"
COL_INACTIVE = "#D6D6D6"
COL_EDGE     = "#888888"
COL_BDY      = "black"

WELL_FILE = "D:/GMRW/finalresult/obs_data/head_mean_104.csv"
RIV_FILE  = os.path.join(BASE, "modflow_GMRW.riv")

DEG = chr(176)   # degree symbol


def rc_to_lonlat(row1, col1):
    lon = LON_MIN + (col1 - 0.5) * DLON
    lat = LAT_MAX - (row1 - 0.5) * DLAT
    return lon, lat


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


def load_river_lonlat(ibound):
    lons, lats = [], []
    with open(RIV_FILE) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    layer, row1, col1 = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                r, c = row1 - 1, col1 - 1
                if 0 <= r < NROW and 0 <= c < NCOL and ibound[r, c] != 0:
                    lon, lat = rc_to_lonlat(row1, col1)
                    lons.append(lon)
                    lats.append(lat)
    return np.array(lons), np.array(lats)


def load_wells_lonlat(ibound):
    df = pd.read_csv(WELL_FILE)
    lons, lats = [], []
    for _, row in df.iterrows():
        r, c = int(row["row"]) - 1, int(row["clo"]) - 1
        if 0 <= r < NROW and 0 <= c < NCOL and ibound[r, c] != 0:
            lon, lat = rc_to_lonlat(int(row["row"]), int(row["clo"]))
            lons.append(lon)
            lats.append(lat)
    return np.array(lons), np.array(lats)


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


def add_scale_bar(ax, length_km=50):
    km_per_deg_lon = 111.32 * np.cos(np.radians(39.5))
    bar_deg = length_km / km_per_deg_lon
    x0 = LON_MIN + 0.04 * (LON_MAX - LON_MIN)
    y0 = LAT_MIN + 0.04 * (LAT_MAX - LAT_MIN)
    ax.plot([x0, x0 + bar_deg], [y0, y0], "k-", lw=2.5)
    for x in [x0, x0 + bar_deg]:
        ax.plot(x, y0, "k|", lw=2, ms=6)
    ax.text(x0 + bar_deg / 2, y0 - 0.022, str(length_km) + " km",
            ha="center", va="top", fontsize=8.5, fontweight="bold")


def add_north_arrow(ax):
    x = LON_MIN + 0.055 * (LON_MAX - LON_MIN)
    y_base = LAT_MIN + 0.12 * (LAT_MAX - LAT_MIN)
    y_tip  = LAT_MIN + 0.165 * (LAT_MAX - LAT_MIN)
    ax.annotate("", xy=(x, y_tip), xytext=(x, y_base),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    ax.text(x, y_base - 0.015, "N", ha="center", va="top",
            fontsize=9, fontweight="bold")


def make_grid_figure(ibound, riv_lons, riv_lats, well_lons, well_lats):
    active   = ibound != 0
    inactive = ibound == 0
    disp = active.astype(float)

    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.14)
    ax.set_facecolor(COL_INACTIVE)

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([COL_INACTIVE, COL_ACTIVE])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], ncolors=2)

    lon_edges = np.linspace(LON_MIN, LON_MAX, NCOL + 1)
    lat_edges = np.linspace(LAT_MAX, LAT_MIN, NROW + 1)
    LON_G, LAT_G = np.meshgrid(lon_edges, lat_edges)

    ax.pcolormesh(LON_G, LAT_G, disp,
                  cmap=cmap, norm=norm,
                  shading="flat", linewidth=0, edgecolors="none", zorder=2)

    h_segs = [[(LON_MIN, LAT_MAX - r*DLAT), (LON_MAX, LAT_MAX - r*DLAT)]
              for r in range(NROW + 1)]
    v_segs = [[(LON_MIN + c*DLON, LAT_MIN), (LON_MIN + c*DLON, LAT_MAX)]
              for c in range(NCOL + 1)]
    ax.add_collection(LineCollection(
        h_segs, colors=COL_EDGE, linewidths=0.25, alpha=0.6, zorder=3))
    ax.add_collection(LineCollection(
        v_segs, colors=COL_EDGE, linewidths=0.25, alpha=0.6, zorder=3))

    bdy = watershed_boundary_segments(ibound)
    ax.add_collection(LineCollection(bdy, colors=COL_BDY, linewidths=1.4, zorder=6))

    ax.scatter(riv_lons, riv_lats,
               s=2.5, c="#2166AC", marker="s", linewidths=0, zorder=7)
    ax.scatter(well_lons, well_lats,
               s=20, c="#FF6600", edgecolors="#8B3A00", linewidths=0.5,
               marker="o", zorder=8)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")
    lon_ticks = np.arange(-84.8, -83.5, 0.2)
    lat_ticks = np.arange(39.2,  40.8,  0.2)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels(["%.1f" % v + DEG for v in lon_ticks], fontsize=9)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels(["%.1f" % v + DEG for v in lat_ticks], fontsize=9)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("Longitude (" + DEG + ")", fontsize=10)
    ax.set_ylabel("Latitude (" + DEG + ")",  fontsize=10)

    add_scale_bar(ax, length_km=50)
    add_north_arrow(ax)

    legend_handles = [
        mpatches.Patch(facecolor=COL_ACTIVE, edgecolor="#555", lw=0.5,
                       label="Active IBOUND cells  (n = " + format(active.sum(), ",") + ")"),
        mpatches.Patch(facecolor=COL_INACTIVE, edgecolor="#555", lw=0.5,
                       label="Inactive IBOUND cells (n = " + format(inactive.sum(), ",") + ")"),
        Line2D([0], [0], color=COL_BDY, lw=1.5, label="Watershed boundary"),
        Line2D([0], [0], color="#2166AC", lw=0, marker="s", markersize=5,
               label="River network (n = " + format(len(riv_lons), ",") + " cells)"),
        Line2D([0], [0], color="#FF6600", lw=0, marker="o", markersize=7,
               markeredgecolor="#8B3A00", markeredgewidth=0.5,
               label="Obs. wells (n = " + str(len(well_lons)) + ")"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, fontsize=9, framealpha=0.95,
              borderpad=0.6, edgecolor="#888888",
              handlelength=1.4, handleheight=1.0)

    fig.suptitle(
        "MODFLOW finite-difference grid - Great Miami River Watershed",
        fontsize=11, fontweight="bold", y=0.97)
    return fig


if __name__ == "__main__":
    print("Loading IBOUND ...")
    ibound = load_ibound()
    n_act = (ibound != 0).sum()
    n_ina = (ibound == 0).sum()
    print("  Active cells:", n_act)
    print("  Inactive cells:", n_ina)
    print("Loading river cells (active only) ...")
    riv_lons, riv_lats = load_river_lonlat(ibound)
    print("  River cells kept:", len(riv_lons))
    print("Loading wells (active only) ...")
    well_lons, well_lats = load_wells_lonlat(ibound)
    print("  Wells kept:", len(well_lons))
    print("Generating figure ...")
    fig = make_grid_figure(ibound, riv_lons, riv_lats, well_lons, well_lats)
    out_png = os.path.join(OUT, "fig_grid_map.png")
    out_pdf = os.path.join(OUT, "fig_grid_map.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,          bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out_png)
"""

with open(r"D:\GMRW\mapping\make_grid_map.py", "w", encoding="ascii") as f:
    f.write(script)
print("Written OK, lines:", script.count("\n"))
