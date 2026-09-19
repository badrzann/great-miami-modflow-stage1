"""
map_mcd_wells.py
================
Plot all MCD "Depth to Water" wells against the GMRW watershed boundary.
Reports how many wells fall inside vs outside the boundary.

Outputs:
  D:/GMRW/finalresult/fig/fig_mcd_wells_boundary.png  (300 DPI)
  D:/GMRW/finalresult/fig/fig_mcd_wells_boundary.pdf
"""

import os, re, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NROW, NCOL   = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW

BASE = "D:/GMRW/finalresult/swatmf_run"
OUT  = "D:/GMRW/finalresult/fig"
OBS_CSV = "D:/GMRW/finalresult/obs_data/mcd_well_name_location_matches.csv"
MCD_CACHE = "D:/GMRW/_tmp/mcd_all_wells_coords.json"
MCD_PORTAL = "https://waterdata.mcdwater.org"

os.makedirs(OUT, exist_ok=True)

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
    arr = np.array(vals[:NROW * NCOL], dtype=int).reshape(NROW, NCOL)
    print(f"  IBOUND: {(arr != 0).sum()} active cells")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2. WATERSHED BOUNDARY SEGMENTS (for plotting)
# ─────────────────────────────────────────────────────────────────────────────
def watershed_boundary_segments(ibound):
    active = ibound != 0
    segs = []
    diff_h = np.diff(active.astype(int), axis=0)
    for r, c in zip(*np.where(diff_h != 0)):
        lat_y  = LAT_MAX - (r + 1) * DLAT
        lon_x0 = LON_MIN + c * DLON
        lon_x1 = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x0, lat_y), (lon_x1, lat_y)])
    diff_v = np.diff(active.astype(int), axis=1)
    for r, c in zip(*np.where(diff_v != 0)):
        lat_y0 = LAT_MAX - r * DLAT
        lat_y1 = LAT_MAX - (r + 1) * DLAT
        lon_x  = LON_MIN + (c + 1) * DLON
        segs.append([(lon_x, lat_y0), (lon_x, lat_y1)])
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
# 3. INSIDE-BOUNDARY TEST (via IBOUND grid cell lookup)
# ─────────────────────────────────────────────────────────────────────────────
def is_inside_boundary(lon, lat, ibound):
    row = int((LAT_MAX - lat) / DLAT)
    col = int((lon - LON_MIN) / DLON)
    if 0 <= row < NROW and 0 <= col < NCOL:
        return ibound[row, col] != 0
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. FETCH ALL MCD DEPTH-TO-WATER WELL COORDINATES FROM PORTAL
# ─────────────────────────────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return s


def accept_disclaimer(session, return_url="/Data/List"):
    r = session.get(f"{MCD_PORTAL}/Data/List", timeout=30)
    m = re.search(r'__RequestVerificationToken.*?value="([^"]+)"', r.text)
    if not m:
        raise RuntimeError("Could not find anti-forgery token")
    tok = m.group(1)
    session.post(
        f"{MCD_PORTAL}/AcceptDisclaimer",
        data={"__RequestVerificationToken": tok, "returnUrl": return_url},
        timeout=30
    )
    print("  Disclaimer accepted.")


def try_fetch_geojson(session):
    """
    Try several candidate endpoint patterns that AQUARIUS WebPortal uses
    for GeoJSON well locations on the map view.
    """
    candidates = [
        # Pattern 1: Map GeoJSON with filter parameters
        "/Data/GetLocations?parameter=Depth+to+Water&computedstatistic=LATEST&interval=Latest&wkid=4326",
        "/Data/FilteredLocations?parameter=Depth+to+Water&computedstatistic=LATEST&interval=Latest&wkid=4326",
        "/Data/GetMapLocations?parameter=Depth+to+Water&interval=Latest&wkid=4326",
        # Pattern 2: Generic location list
        "/Data/GetAllLocations?wkid=4326",
        "/Data/LocationGeoJSON?parameter=Depth+to+Water&wkid=4326",
    ]
    for path in candidates:
        try:
            r = session.get(f"{MCD_PORTAL}{path}", timeout=30)
            if r.status_code == 200 and "features" in r.text.lower():
                print(f"  GeoJSON found at: {path}")
                return r.json()
            print(f"  {path} → {r.status_code}")
        except Exception as e:
            print(f"  {path} → error: {e}")
    return None


def try_fetch_map_page_layers(session):
    """
    Load the Map page and look for the data source URL embedded in the HTML.
    """
    url = f"{MCD_PORTAL}/Data/Map/Parameter/Depth%20to%20Water/Statistic/LATEST/Interval/Latest"
    r = session.get(url, timeout=30)
    # Look for data-url, dataSource transport url, or any /Data/ path with wkid
    patterns = [
        r'data-url=["\']([^"\']+wkid[^"\']*)["\']',
        r'"url"\s*:\s*"([^"]+/Data/[^"]+)"',
        r"url\s*:\s*['\"]([^'\"]+/Data/Map[^'\"]*)['\"]",
        r'(\/Data\/[A-Za-z]+\?[^"\'<>\s]+wkid[^"\'<>\s]*)',
        r'(\/Data\/[A-Za-z]+\?[^"\'<>\s]+arameter[^"\'<>\s]*)',
    ]
    for pat in patterns:
        m = re.search(pat, r.text, re.IGNORECASE)
        if m:
            candidate = m.group(1)
            print(f"  Found candidate URL in map page: {candidate}")
            return candidate
    # Save a snippet for debug
    with open("D:/GMRW/_tmp/mcd_map_page_snippet.html", "w", encoding="utf-8") as f:
        f.write(r.text[:5000])
    print("  No data URL found in map page HTML (snippet saved to _tmp/).")
    return None


def extract_features_from_geojson(geojson):
    """Parse GeoJSON FeatureCollection and return list of (id, name, lon, lat)."""
    wells = []
    features = geojson.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", [None, None])
        lon, lat = float(coords[0]), float(coords[1])
        loc_id   = props.get("Identifier") or props.get("Id") or props.get("LocationIdentifier") or ""
        name     = props.get("Name") or props.get("LocationName") or props.get("label") or loc_id
        wells.append({"mcd_location_id": loc_id, "name": name, "lon": lon, "lat": lat})
    return wells


def fetch_mcd_wells():
    """
    Try to get all MCD Depth-to-Water well coordinates. Uses cache if available.
    Falls back to the 84 matched wells from the CSV if portal is unreachable.
    """
    # ── Use cache ──────────────────────────────────────────────────────────
    if os.path.exists(MCD_CACHE):
        print(f"  Loading MCD well coords from cache: {MCD_CACHE}")
        with open(MCD_CACHE) as f:
            return json.load(f)

    print("  Fetching MCD well coordinates from portal …")
    session = make_session()
    wells = []
    try:
        accept_disclaimer(session)
        # Try direct GeoJSON endpoints
        geojson = try_fetch_geojson(session)
        if geojson:
            wells = extract_features_from_geojson(geojson)
            print(f"  Got {len(wells)} wells from GeoJSON endpoint.")

        # If that failed, try finding the URL from the map page
        if not wells:
            data_url = try_fetch_map_page_layers(session)
            if data_url:
                full_url = data_url if data_url.startswith("http") else f"{MCD_PORTAL}{data_url}"
                r = session.get(full_url, timeout=30)
                if r.status_code == 200:
                    geojson = r.json()
                    wells = extract_features_from_geojson(geojson)
                    print(f"  Got {len(wells)} wells from map page data URL.")

    except Exception as e:
        print(f"  Portal fetch failed: {e}")

    if wells:
        with open(MCD_CACHE, "w") as f:
            json.dump(wells, f, indent=2)
        print(f"  Cached {len(wells)} wells → {MCD_CACHE}")
    else:
        print("  Could not fetch from portal; will use CSV coordinates only.")

    return wells


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOAD MATCHED WELLS CSV (always available – 84 wells with lat/lon)
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_wells():
    df = pd.read_csv(OBS_CSV)
    wells = []
    for _, row in df.iterrows():
        loc_id = str(row["mcd_location_id"]).strip()
        name   = str(row.get("local_well_name", loc_id)).strip()
        lon    = float(row["lon"])
        lat    = float(row["lat"])
        wells.append({"mcd_location_id": loc_id, "name": name, "lon": lon, "lat": lat})
    print(f"  CSV: {len(wells)} matched wells with coordinates.")
    return wells


# ─────────────────────────────────────────────────────────────────────────────
# 6. MERGE WELLS (portal + csv, deduplicate by mcd_location_id)
# ─────────────────────────────────────────────────────────────────────────────
def merge_wells(portal_wells, csv_wells):
    seen   = {}
    all_w  = []
    for w in portal_wells:
        k = w["mcd_location_id"]
        seen[k] = w
        all_w.append(w)
    # Add CSV wells not already present
    for w in csv_wells:
        k = w["mcd_location_id"]
        if k not in seen:
            seen[k] = w
            all_w.append(w)
    print(f"  Total wells after merge: {len(all_w)}")
    return all_w


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLASSIFY WELLS
# ─────────────────────────────────────────────────────────────────────────────
def classify_wells(wells, ibound):
    inside  = [w for w in wells if is_inside_boundary(w["lon"], w["lat"], ibound)]
    outside = [w for w in wells if not is_inside_boundary(w["lon"], w["lat"], ibound)]
    return inside, outside


# ─────────────────────────────────────────────────────────────────────────────
# 8. DOWNLOAD / LOAD US STATES (for locator map – optional)
# ─────────────────────────────────────────────────────────────────────────────
_STATES_CACHE = "D:/GMRW/_tmp_us_states.json"
_STATES_URL   = ("https://raw.githubusercontent.com/"
                 "PublicaMundi/MappingAPI/master/data/geojson/us-states.json")

def load_us_states():
    if os.path.exists(_STATES_CACHE):
        with open(_STATES_CACHE) as f:
            data = json.load(f)
    else:
        import urllib.request
        print("  Downloading US states GeoJSON …")
        with urllib.request.urlopen(_STATES_URL, timeout=15) as r:
            data = json.loads(r.read())
        with open(_STATES_CACHE, "w") as f:
            json.dump(data, f)
    polys = []
    for feat in data["features"]:
        geom = feat["geometry"]
        rings = (geom["coordinates"] if geom["type"] == "Polygon"
                 else [ring for part in geom["coordinates"] for ring in part])
        for ring in rings:
            polys.append(([p[0] for p in ring], [p[1] for p in ring]))
    return polys


# ─────────────────────────────────────────────────────────────────────────────
# 9. DRAW THE MAP
# ─────────────────────────────────────────────────────────────────────────────
def draw_map(ibound, inside_wells, outside_wells, total_wells):
    segs = watershed_boundary_segments(ibound)
    active = ibound != 0

    fig = plt.figure(figsize=(14, 10))

    # ── Left panel: USA locator ───────────────────────────────────────────
    ax_loc = fig.add_axes([0.02, 0.30, 0.22, 0.40])
    ax_loc.set_aspect("equal")
    ax_loc.set_xlim(-130, -65)
    ax_loc.set_ylim(24, 50)
    ax_loc.set_facecolor("#D6EAF8")
    ax_loc.axis("off")
    try:
        states = load_us_states()
        for (lons, lats) in states:
            ax_loc.fill(lons, lats, fc="#F0F0F0", ec="gray", lw=0.4, zorder=1)
    except Exception:
        pass
    # GMRW box on locator
    bx = [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN]
    by = [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN]
    ax_loc.fill(bx, by, fc="red", alpha=0.5, zorder=3)
    ax_loc.plot(bx, by, "r-", lw=1.5, zorder=4)
    ax_loc.set_title("Location", fontsize=8, pad=2)

    # ── Right panel: main map ─────────────────────────────────────────────
    ax = fig.add_axes([0.28, 0.08, 0.68, 0.86])
    ax.set_facecolor("#EBF5FB")

    # Active-cell mask (light fill = watershed area)
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    ax.imshow(active, origin="upper", extent=extent,
              cmap=plt.cm.colors.ListedColormap(["none", "#D5E8D4"]),
              aspect="auto", zorder=1, alpha=0.8)

    # Watershed boundary
    lc = LineCollection(segs, linewidths=1.2, colors="black", zorder=3)
    ax.add_collection(lc)

    # Outside wells (red circles)
    if outside_wells:
        ox = [w["lon"] for w in outside_wells]
        oy = [w["lat"] for w in outside_wells]
        ax.scatter(ox, oy, s=55, c="#E74C3C", edgecolors="darkred",
                   linewidths=0.6, zorder=6, label=f"Outside boundary ({len(outside_wells)})")
        # Label them
        for w in outside_wells:
            ax.annotate(w["mcd_location_id"], (w["lon"], w["lat"]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=5.5, color="darkred", zorder=7)

    # Inside wells (blue triangles)
    if inside_wells:
        ix = [w["lon"] for w in inside_wells]
        iy = [w["lat"] for w in inside_wells]
        ax.scatter(ix, iy, s=60, c="#2874A6", marker="^",
                   edgecolors="#1A5276", linewidths=0.6, zorder=6,
                   label=f"Inside boundary ({len(inside_wells)})")

    # Axes formatting
    ax.set_xlim(LON_MIN - 0.05, LON_MAX + 0.05)
    ax.set_ylim(LAT_MIN - 0.05, LAT_MAX + 0.05)
    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # Title
    n_total = len(inside_wells) + len(outside_wells)
    ax.set_title(
        f"MCD Groundwater Wells — Depth to Water\n"
        f"Total: {n_total}   Inside watershed: {len(inside_wells)}   "
        f"Outside watershed: {len(outside_wells)}",
        fontsize=12, fontweight="bold"
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#D5E8D4", edgecolor="black", label="Watershed (active MODFLOW cells)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2874A6",
               markeredgecolor="#1A5276", markersize=9,
               label=f"Inside boundary: {len(inside_wells)}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
               markeredgecolor="darkred", markersize=8,
               label=f"Outside boundary: {len(outside_wells)}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="gray")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("─" * 60)
    print("Loading MODFLOW IBOUND …")
    ibound = load_ibound()

    print("Loading CSV wells (84 matched) …")
    csv_wells = load_csv_wells()

    print("Fetching MCD well coordinates from portal …")
    portal_wells = fetch_mcd_wells()

    print("Merging well lists …")
    all_wells = merge_wells(portal_wells, csv_wells)

    print("Classifying wells (inside / outside watershed) …")
    inside, outside = classify_wells(all_wells, ibound)

    print()
    print("=" * 60)
    print(f"  Total MCD wells with coordinates : {len(all_wells)}")
    print(f"  Inside  watershed boundary       : {len(inside)}")
    print(f"  Outside watershed boundary       : {len(outside)}")
    print()
    if outside:
        print("  Wells OUTSIDE boundary:")
        for w in sorted(outside, key=lambda x: x["mcd_location_id"]):
            print(f"    {w['mcd_location_id']:<18}  lon={w['lon']:.4f}  lat={w['lat']:.4f}")
    print("=" * 60)

    print("\nDrawing map …")
    fig = draw_map(ibound, inside, outside, len(all_wells))

    png_path = os.path.join(OUT, "fig_mcd_wells_boundary.png")
    pdf_path = os.path.join(OUT, "fig_mcd_wells_boundary.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
