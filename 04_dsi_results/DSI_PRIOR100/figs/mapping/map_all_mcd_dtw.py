"""
1. Categorize all 210 MCD locations → groundwater wells only
2. Fetch lat/lon for each groundwater well from portal location page
3. Classify inside/outside watershed boundary
4. Plot the map
"""
import json, re, time, os
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# ── Constants ─────────────────────────────────────────────────────────────
NROW, NCOL   = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW
BASE_MODEL = "D:/GMRW/finalresult/swatmf_run"
OUT   = "D:/GMRW/finalresult/fig"
OBS_CSV = "D:/GMRW/finalresult/obs_data/mcd_well_name_location_matches.csv"
MCD_LOC_JSON = "D:/GMRW/_tmp/mcd_all_location_ids.json"
COORDS_CACHE = "D:/GMRW/_tmp/mcd_gw_coords.json"
MCD_PORTAL   = "https://waterdata.mcdwater.org"
os.makedirs(OUT, exist_ok=True)

# ── Load IBOUND ────────────────────────────────────────────────────────────
def load_ibound():
    vals = []
    with open(os.path.join(BASE_MODEL, "modflow_GMRW.bas")) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.upper() == "FREE":
                continue
            for tok in s.split():
                try: vals.append(int(float(tok)))
                except: pass
            if len(vals) >= NROW * NCOL:
                break
    return np.array(vals[:NROW*NCOL], dtype=int).reshape(NROW, NCOL)

# ── Watershed boundary segments ────────────────────────────────────────────
def watershed_segments(ibound):
    active = ibound != 0
    segs = []
    dh = np.diff(active.astype(int), axis=0)
    for r, c in zip(*np.where(dh != 0)):
        y  = LAT_MAX - (r+1)*DLAT
        segs.append([(LON_MIN+c*DLON, y), (LON_MIN+(c+1)*DLON, y)])
    dv = np.diff(active.astype(int), axis=1)
    for r, c in zip(*np.where(dv != 0)):
        y0, y1 = LAT_MAX - r*DLAT, LAT_MAX - (r+1)*DLAT
        x = LON_MIN + (c+1)*DLON
        segs.append([(x, y0), (x, y1)])
    for c in range(NCOL):
        if active[0, c]:     segs.append([(LON_MIN+c*DLON, LAT_MAX), (LON_MIN+(c+1)*DLON, LAT_MAX)])
        if active[NROW-1,c]: segs.append([(LON_MIN+c*DLON, LAT_MIN), (LON_MIN+(c+1)*DLON, LAT_MIN)])
    for r in range(NROW):
        if active[r, 0]:     segs.append([(LON_MIN, LAT_MAX-r*DLAT), (LON_MIN, LAT_MAX-(r+1)*DLAT)])
        if active[r,NCOL-1]: segs.append([(LON_MAX, LAT_MAX-r*DLAT), (LON_MAX, LAT_MAX-(r+1)*DLAT)])
    return segs

def is_inside(lon, lat, ibound):
    row = int((LAT_MAX - lat) / DLAT)
    col = int((lon - LON_MIN) / DLON)
    if 0 <= row < NROW and 0 <= col < NCOL:
        return ibound[row, col] != 0
    return False

# ── Categorise: identify groundwater wells ─────────────────────────────────
def is_groundwater_well(loc_id):
    """
    Return True if the location ID looks like a groundwater monitoring well.
    Exclude: 8-digit USGS stream gauges (03xxxxxx), numbered/lettered precip
    stations (0-33, AB, etc.), and named river quality stations.
    """
    exc_names = {"BOLTON","ENGLEWOOD","HUFFMAN","MIAMISBURG","MIAMIVILLA",
                 "MIAMIVILLA"}
    if loc_id in exc_names:
        return False
    # USGS stream gauges start with "03"
    if loc_id.startswith("03") and len(loc_id) >= 7:
        return False
    # Numbered/lettered precipitation stations
    # (single digit, two digit, e.g. 0-33; also AB, AB1, AB2, 1A, 1B, 6G, 23A, 23I, 3B, 16B, 16B1, 24A)
    stripped = re.sub(r'[A-Z]','', loc_id)
    if stripped.isdigit() and int(stripped) <= 60 and len(loc_id) <= 4:
        return False
    return True

# ── Load all MCD locations and filter to groundwater wells ─────────────────
with open(MCD_LOC_JSON) as f:
    all_locs = json.load(f)

gw_locs = [x for x in all_locs if is_groundwater_well(x["Id"])]
print(f"Total MCD locations       : {len(all_locs)}")
print(f"Groundwater (DTW) wells   : {len(gw_locs)}")
print()

# ── Load CSV coordinates (85 matched model wells) ─────────────────────────
import csv
csv_coords = {}
with open(OBS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        wid = row["mcd_location_id"].strip()
        csv_coords[wid] = (float(row["lon"]), float(row["lat"]))
print(f"CSV coordinates available : {len(csv_coords)}")

# ── Fetch coordinates from portal for wells not in CSV ────────────────────
def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r = session.get(f"{MCD_PORTAL}/Data/List", timeout=30)
    m = re.search(r'__RequestVerificationToken.*?value="([^"]+)"', r.text)
    tok = m.group(1)
    session.post(f"{MCD_PORTAL}/AcceptDisclaimer",
        data={"__RequestVerificationToken": tok, "returnUrl": "/Data/List"},
        timeout=30)
    return session

def try_get_coords_from_page(session, id_number, loc_id):
    """Try to extract lat/lon from portal location page HTML."""
    url = f"{MCD_PORTAL}/Data/Location?id={id_number}&interval=Latest"
    try:
        r = session.get(url, timeout=25)
        html = r.text
        # Search for various lat/lon patterns in the HTML
        patterns = [
            r'"latitude"\s*:\s*(-?\d+\.?\d*)',
            r'"longitude"\s*:\s*(-?\d+\.?\d*)',
            r'latitude["\s:]+(-?\d+\.\d+)',
            r'longitude["\s:]+(-?\d+\.\d+)',
            r'lat["\s:]+(-?\d+\.\d{4,})',
            r'lon["\s:]+(-8[0-9]\.\d{4,})',
            r'lng["\s:]+(-8[0-9]\.\d{4,})',
            r'data-lat=["\'](-?\d+\.\d+)["\']',
            r'data-lon=["\'](-?\d+\.\d+)["\']',
            r'data-lng=["\'](-?\d+\.\d+)["\']',
            r'center\s*:\s*\[(-8[0-9]\.\d+),\s*([34]\d\.\d+)\]',
            r'(-8[3-5]\.\d{4,})',   # rough lon for SW Ohio
        ]
        lat = lon = None
        # Try to find lat/lon JSON pair
        lat_m = re.search(r'"[Ll]at(?:itude)?"\s*:\s*(-?\d+\.\d+)', html)
        lon_m = re.search(r'"[Ll]on(?:gitude)?"\s*:\s*(-?\d+\.\d+)', html)
        if lat_m and lon_m:
            lat = float(lat_m.group(1))
            lon = float(lon_m.group(1))
        if not lat:
            # Try combined pattern [lon, lat] or {lon, lat}
            m = re.search(r'\[(-8[34]\.\d{4,})\s*,\s*(3[9-9]\.\d{4,})\]', html)
            if m:
                lon, lat = float(m.group(1)), float(m.group(2))
        if not lat:
            # Look for center: or center= pattern used by Kendo map
            m = re.search(r'center["\s:=\[]+([34][0-9]\.\d{4,})\s*,\s*(-8[34]\.\d{4,})', html)
            if m:
                lat, lon = float(m.group(1)), float(m.group(2))
        return lon, lat
    except Exception as e:
        return None, None

# ── Build full coordinates list ────────────────────────────────────────────
if os.path.exists(COORDS_CACHE):
    print(f"Loading cached coordinates: {COORDS_CACHE}")
    with open(COORDS_CACHE) as f:
        extra_coords = json.load(f)
else:
    print("Fetching missing well coordinates from portal...")
    session = make_session()
    extra_coords = {}
    missing = [w for w in gw_locs if w["Id"] not in csv_coords]
    print(f"  Wells needing coords: {len(missing)}")
    for i, w in enumerate(missing):
        id_num = w.get("IDNumber", "")
        loc_id = w["Id"]
        lon, lat = try_get_coords_from_page(session, id_num, loc_id)
        if lon and lat:
            extra_coords[loc_id] = (lon, lat)
            print(f"  [{i+1}/{len(missing)}] {loc_id}: lon={lon:.4f} lat={lat:.4f}")
        else:
            print(f"  [{i+1}/{len(missing)}] {loc_id}: coords not found in page")
        time.sleep(0.2)
    with open(COORDS_CACHE, "w") as f:
        json.dump(extra_coords, f, indent=2)
    print(f"  Saved {len(extra_coords)} extra coords → {COORDS_CACHE}")

# ── Assemble all well records with coordinates ─────────────────────────────
all_coords = {**{k: v for k,v in csv_coords.items()}, **extra_coords}

wells_with_coords = []
wells_no_coords   = []
for w in gw_locs:
    wid = w["Id"]
    if wid in all_coords:
        lon, lat = all_coords[wid]
        wells_with_coords.append({"id": wid, "display": w.get("DisplayText",""), "lon": lon, "lat": lat})
    else:
        wells_no_coords.append(wid)

print(f"\nGroundwater wells total   : {len(gw_locs)}")
print(f"  With coordinates        : {len(wells_with_coords)}")
print(f"  Without coordinates     : {len(wells_no_coords)}")
if wells_no_coords:
    print(f"  Missing: {wells_no_coords}")

# ── Load IBOUND & classify ─────────────────────────────────────────────────
ibound = load_ibound()
inside  = [w for w in wells_with_coords if is_inside(w["lon"], w["lat"], ibound)]
outside = [w for w in wells_with_coords if not is_inside(w["lon"], w["lat"], ibound)]

print(f"\n{'='*55}")
print(f"MCD Depth to Water wells (groundwater only): {len(gw_locs)}")
print(f"  Plotted (have coords)   : {len(wells_with_coords)}")
print(f"  Inside boundary         : {len(inside)}")
print(f"  Outside boundary        : {len(outside)}")
print(f"  No coords (not plotted) : {len(wells_no_coords)}")
print()
print("Wells OUTSIDE boundary:")
for w in sorted(outside, key=lambda x: x["id"]):
    print(f"  {w['id']:<25}  lon={w['lon']:.4f}  lat={w['lat']:.4f}")

# ── Load US states ─────────────────────────────────────────────────────────
_STATES_CACHE = "D:/GMRW/_tmp_us_states.json"

def load_us_states():
    with open(_STATES_CACHE) as f:
        data = json.load(f)
    polys = []
    for feat in data["features"]:
        geom = feat["geometry"]
        rings = (geom["coordinates"] if geom["type"] == "Polygon"
                 else [r for p in geom["coordinates"] for r in p])
        for ring in rings:
            polys.append(([p[0] for p in ring], [p[1] for p in ring]))
    return polys

# ── Draw map ──────────────────────────────────────────────────────────────
segs   = watershed_segments(ibound)
active = ibound != 0

fig = plt.figure(figsize=(14, 10))

# Locator
ax_loc = fig.add_axes([0.02, 0.30, 0.22, 0.40])
ax_loc.set_aspect("equal"); ax_loc.set_xlim(-130,-65); ax_loc.set_ylim(24,50)
ax_loc.set_facecolor("#D6EAF8"); ax_loc.axis("off")
try:
    for (lons,lats) in load_us_states():
        ax_loc.fill(lons,lats,fc="#F0F0F0",ec="gray",lw=0.4,zorder=1)
except: pass
bx=[LON_MIN,LON_MAX,LON_MAX,LON_MIN,LON_MIN]
by=[LAT_MIN,LAT_MIN,LAT_MAX,LAT_MAX,LAT_MIN]
ax_loc.fill(bx,by,fc="red",alpha=0.5,zorder=3)
ax_loc.plot(bx,by,"r-",lw=1.5,zorder=4)
ax_loc.set_title("Location",fontsize=8,pad=2)

# Main map
ax = fig.add_axes([0.28, 0.08, 0.68, 0.86])
ax.set_facecolor("#EBF5FB")
extent=[LON_MIN,LON_MAX,LAT_MIN,LAT_MAX]
cmap2 = matplotlib.colors.ListedColormap(["none","#D5E8D4"])
ax.imshow(active, origin="upper", extent=extent, cmap=cmap2, aspect="auto", zorder=1, alpha=0.85)
lc = LineCollection(segs, linewidths=1.3, colors="black", zorder=3)
ax.add_collection(lc)

# Outside (red circles)
if outside:
    ox=[w["lon"] for w in outside]; oy=[w["lat"] for w in outside]
    ax.scatter(ox,oy,s=55,c="#E74C3C",edgecolors="darkred",lw=0.6,zorder=6)
    for w in outside:
        ax.annotate(w["id"],(w["lon"],w["lat"]),textcoords="offset points",
                    xytext=(4,3),fontsize=5,color="darkred",zorder=7)

# Inside (blue triangles)
if inside:
    ix=[w["lon"] for w in inside]; iy=[w["lat"] for w in inside]
    ax.scatter(ix,iy,s=60,c="#2874A6",marker="^",edgecolors="#1A5276",lw=0.6,zorder=6)

ax.set_xlim(LON_MIN-0.05, LON_MAX+0.05)
ax.set_ylim(LAT_MIN-0.05, LAT_MAX+0.05)
ax.set_xlabel("Longitude (°)", fontsize=11)
ax.set_ylabel("Latitude (°)",  fontsize=11)
ax.tick_params(labelsize=9)
ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

n_total = len(wells_with_coords)
n_all   = len(gw_locs)
ax.set_title(
    f"MCD Groundwater Monitoring Network — Depth to Water\n"
    f"Total DTW wells: {n_all}   Plotted (with coords): {n_total}"
    f"   Inside: {len(inside)}   Outside: {len(outside)}",
    fontsize=11, fontweight="bold"
)

leg = [
    mpatches.Patch(fc="#D5E8D4", ec="black", label="Watershed (active MODFLOW cells)"),
    Line2D([0],[0], marker="^", color="w", mfc="#2874A6", mec="#1A5276", ms=9,
           label=f"Inside boundary: {len(inside)}"),
    Line2D([0],[0], marker="o", color="w", mfc="#E74C3C", mec="darkred", ms=8,
           label=f"Outside boundary: {len(outside)}"),
]
ax.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.92, edgecolor="gray")

png = os.path.join(OUT, "fig_mcd_dtw_all_wells.png")
pdf = os.path.join(OUT, "fig_mcd_dtw_all_wells.pdf")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {png}")
print(f"Saved: {pdf}")
