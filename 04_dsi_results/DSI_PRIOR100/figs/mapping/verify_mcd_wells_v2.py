"""
Final corrected verification - filter to true groundwater (DTW) wells only.
GW wells identified by county prefix codes (BUT, MON, CLA, MIA, PRE, SHE, WAR, 
HAM, DAR, CHA, GRE, MIA, LO-, BU-, MI-, CL-, PR-, SHE, etc.)

Non-GW locations:
- Stream gauges: start with '032...'
- Precip stations: numeric/short IDs (0-34, single letter combos)
- River quality stations: BOLTON, ENGLEWOOD, HUFFMAN, MIAMISBURG, MIAMIVILLA
"""
import requests, re, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "https://waterdata.mcdwater.org"
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest", "Referer": BASE + "/Data/Map"})
r0 = s.get(f"{BASE}/Data/List", timeout=30)
m = re.search(r'__RequestVerificationToken.*?value="([^"]+)"', r0.text)
tok = m.group(1)
s.post(f"{BASE}/AcceptDisclaimer", data={"__RequestVerificationToken": tok, "returnUrl": "/Data/Map"}, timeout=30)
r = s.post(f"{BASE}/Map/Indicators", data={"interval":"Latest","legend":"1","utcOffset":"240","date":"2026-04-09","wkid":"4326"}, timeout=30)
all_features = r.json()["features"]

# Load IBOUND
bas_file = "D:/GMRW/finalresult/swatmf_run/modflow_GMRW.bas"
with open(bas_file) as f2:
    lines = f2.readlines()
NROW, NCOL = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX = 39.185, 40.710
DLON = (LON_MAX - LON_MIN) / NCOL
DLAT = (LAT_MAX - LAT_MIN) / NROW

ibound = np.zeros((NROW, NCOL), dtype=int)
vals = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        continue
    for part in stripped.split():
        if part.lstrip('-').isdigit():
            vals.append(int(part))
    if len(vals) >= NROW * NCOL:
        break
for i, v in enumerate(vals[:NROW*NCOL]):
    ibound[i//NCOL, i%NCOL] = v

def inside(lon, lat):
    col_i = int((lon - LON_MIN) / DLON)
    row_i = int((LAT_MAX - lat) / DLAT)
    if row_i < 0 or row_i >= NROW or col_i < 0 or col_i >= NCOL:
        return False
    return ibound[row_i, col_i] != 0

# Classify all 208 features into categories
STREAM_GAUGES = {"BOLTON"}  # also 03... prefix
PRECIP_QUALITY_NAMES = {"Stillwater River at Englewood Dam Quality Station",
                         "Mad River at Huffman Dam Quality Station",
                         "Great Miami River at Miamisburg Quality Station",
                         "Great Miami River at Huber Heights Quality Station",
                         "Great Miami River near Fairfield Quality Station"}

# Groundwater well prefixes (county codes)
GW_PREFIXES = ("BUT", "MON", "CLA", "MIA", "PRE", "SHE", "WAR", "HAM", "DAR",
               "CHA", "GRE", "LO-", "BU-", "MI-", "CL-", "PR-", "HUF", "LOG",
               "ENG", "MIA", "16B", "W-9")

def is_gw_well(loc_id, loc_name):
    """True if this is a groundwater/DTW well (not stream gauge, not precip station)."""
    loc_id = str(loc_id).strip()
    # Stream gauges: USGS format 03...
    if loc_id.startswith("03"):
        return False
    # River quality stations by name
    if loc_name.strip() in PRECIP_QUALITY_NAMES:
        return False
    # Named quality stations with partial match
    if any(q in loc_name for q in ["Quality Station", "WWTP", "Increm"]):
        return False
    # Short numeric / alphabetic IDs = precip stations
    # These are 0-100 (no decimal) or 1-2 letter + 1 number like AB, AB1, AB2
    if len(loc_id) <= 4 and loc_id not in GW_PREFIXES:
        # Might be a precip station - check if it looks like a precip name
        precip_names_kw = ["DAM", "RESERVOIR", "RAINFALL", "PRECIP", "GAGE"]
        if any(kw in loc_name.upper() for kw in precip_names_kw):
            return False
        # Numeric short IDs are precip stream gauges
        clean_id = loc_id.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if clean_id.isdigit() and int(clean_id) <= 40:
            # short numeric IDs - these are precipitation stations
            # UNLESS it starts with a county prefix
            if not any(loc_id.startswith(p) for p in GW_PREFIXES):
                return False
    # Otherwise it's a GW well
    return True

all_rows = []
for feat in all_features:
    lon, lat = feat["geometry"]["coordinates"]
    props = feat["properties"]
    loc_id = props["locationIdentifier"]
    loc_name = props["location"]
    in_b = inside(lon, lat)
    gw = is_gw_well(loc_id, loc_name)
    all_rows.append({
        "locationId": props["locationId"],
        "locationIdentifier": loc_id,
        "locationName": loc_name,
        "lon": lon,
        "lat": lat,
        "inside": in_b,
        "is_gw": gw,
    })
df = pd.DataFrame(all_rows)

print("=== Location category breakdown ===")
print(f"Total locations: {len(df)}")
print(f"Groundwater/DTW wells: {df['is_gw'].sum()}")
print(f"Other (stream gauges, precip, quality): {(~df['is_gw']).sum()}")
print(f"\nOther locations (non-GW):")
for _, row in df[~df["is_gw"]].iterrows():
    print(f"  {row['locationIdentifier']:15s}  {row['locationName'][:50]}")

# GW wells analysis
df_gw = df[df["is_gw"]].copy()
print(f"\n=== GW/DTW Well Analysis ===")
print(f"Total GW/DTW wells: {len(df_gw)}")
print(f"Inside boundary: {df_gw['inside'].sum()}")
print(f"Outside boundary: {(~df_gw['inside']).sum()}")

# Load model wells
df_model = pd.read_csv("D:/GMRW/finalresult/obs_data/mcd_well_name_location_matches.csv")
model_ids = set(df_model["mcd_location_id"].astype(str).str.strip())

# Cross-check
gw_inside = df_gw[df_gw["inside"]].copy()
gw_outside = df_gw[~df_gw["inside"]].copy()
gw_inside_ids = set(gw_inside["locationIdentifier"].astype(str).str.strip())

in_model = gw_inside_ids & model_ids
not_in_model_inside = gw_inside_ids - model_ids
model_not_in_gw_inside = model_ids - gw_inside_ids

print(f"\nGW wells inside boundary: {len(gw_inside)}")
print(f"  -> Used in model: {len(in_model)}")
print(f"  -> NOT in model: {len(not_in_model_inside)}")
print(f"\nModel wells NOT in GW-inside set: {len(model_not_in_gw_inside)}")
if model_not_in_gw_inside:
    for wid in sorted(model_not_in_gw_inside):
        m_rows = df_model[df_model["mcd_location_id"].astype(str).str.strip() == wid]
        if len(m_rows) > 0:
            r = m_rows.iloc[0]
            # Check if this well is in df_gw but outside boundary
            match = df_gw[df_gw["locationIdentifier"].astype(str).str.strip() == wid]
            if len(match) > 0:
                loc_inside = match.iloc[0]["inside"]
                lon_v = match.iloc[0]["lon"]
                lat_v = match.iloc[0]["lat"]
                print(f"  {wid:15s} obsnme={r['obsnme']:25s} inside={loc_inside} ({lat_v:.4f}, {lon_v:.4f})")
            else:
                print(f"  {wid:15s} obsnme={r['obsnme']:25s} NOT FOUND in portal data")

# Check model well coordinates vs portal coordinates
print(f"\n\n=== Model wells coordinate cross-check ===")
# Compare model lat/lon (from CSV) with portal lat/lon
mismatches = []
for _, m_row in df_model.iterrows():
    m_id = str(m_row["mcd_location_id"]).strip()
    m_lat = m_row["lat"]
    m_lon = m_row["lon"]
    portal_match = df[df["locationIdentifier"].astype(str).str.strip() == m_id]
    if len(portal_match) > 0:
        p_lat = portal_match.iloc[0]["lat"]
        p_lon = portal_match.iloc[0]["lon"]
        dist = ((m_lat - p_lat)**2 + (m_lon - p_lon)**2)**0.5
        if dist > 0.01:  # > ~1km disagreement
            mismatches.append((m_id, m_lat, m_lon, p_lat, p_lon, dist))

print(f"Wells with significant coord differences (>{0.01} deg ~ 1km): {len(mismatches)}")
for m_id, m_lat, m_lon, p_lat, p_lon, dist in sorted(mismatches, key=lambda x: -x[5]):
    print(f"  {m_id:15s} model=({m_lat:.4f},{m_lon:.4f}) portal=({p_lat:.4f},{p_lon:.4f}) dist={dist:.4f}")

# ============================================================
# Generate clean verification map
# ============================================================
print("\n=== Building boundary and creating map ===")
boundary_segs = []
for i in range(NROW):
    for j in range(NCOL):
        if ibound[i,j] != 0:
            for di, dj, side in [(-1,0,'top'),(1,0,'bot'),(0,-1,'left'),(0,1,'right')]:
                ni_, nj_ = i+di, j+dj
                if ni_<0 or ni_>=NROW or nj_<0 or nj_>=NCOL or ibound[ni_,nj_]==0:
                    if side=='top':
                        boundary_segs.append(([LON_MIN+j*DLON, LON_MIN+(j+1)*DLON],[LAT_MAX-i*DLAT]*2))
                    elif side=='bot':
                        boundary_segs.append(([LON_MIN+j*DLON, LON_MIN+(j+1)*DLON],[LAT_MAX-(i+1)*DLAT]*2))
                    elif side=='left':
                        boundary_segs.append(([LON_MIN+j*DLON]*2,[LAT_MAX-i*DLAT, LAT_MAX-(i+1)*DLAT]))
                    elif side=='right':
                        boundary_segs.append(([LON_MIN+(j+1)*DLON]*2,[LAT_MAX-i*DLAT, LAT_MAX-(i+1)*DLAT]))

fig, ax = plt.subplots(figsize=(14, 12))
for xd, yd in boundary_segs:
    ax.plot(xd, yd, 'k-', lw=0.35, alpha=0.55)

# Category: model wells inside boundary (in model)
gw_in_model = gw_inside[gw_inside["locationIdentifier"].astype(str).str.strip().isin(model_ids)]
gw_in_nomodel = gw_inside[~gw_inside["locationIdentifier"].astype(str).str.strip().isin(model_ids)]

ax.scatter(gw_in_model["lon"], gw_in_model["lat"], marker='^', s=55, color='#2166ac',
           edgecolors='navy', lw=0.5, zorder=5,
           label=f'GW wells in model, inside boundary (n={len(gw_in_model)})')
if len(gw_in_nomodel) > 0:
    ax.scatter(gw_in_nomodel["lon"], gw_in_nomodel["lat"], marker='D', s=45, color='#f4a460',
               edgecolors='#8b4513', lw=0.5, zorder=5,
               label=f'GW wells inside boundary, not in model (n={len(gw_in_nomodel)})')
ax.scatter(gw_outside["lon"], gw_outside["lat"], marker='x', s=60, color='#d62728',
           lw=1.5, zorder=4,
           label=f'GW wells outside boundary (n={len(gw_outside)})')

# Labels for non-model inside wells
for _, row in gw_in_nomodel.iterrows():
    ax.annotate(row["locationIdentifier"], (row["lon"], row["lat"]),
               fontsize=4.5, xytext=(3, 3), textcoords='offset points', color='#8b4513', zorder=7)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'MCD Groundwater (Depth-to-Water) Well Verification\n'
             f'Total GW Wells: {len(df_gw)}  |  '
             f'Inside boundary: {len(gw_inside)} ({len(gw_in_model)} in model, {len(gw_in_nomodel)} not in model)  |  '
             f'Outside: {len(gw_outside)}', fontsize=12)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

out_dir = "D:/GMRW/finalresult/fig"
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{out_dir}/fig_mcd_dtw_verification.png", dpi=150, bbox_inches='tight')
plt.savefig(f"{out_dir}/fig_mcd_dtw_verification.pdf", bbox_inches='tight')
print(f"Figure saved.")

print("\n" + "="*60)
print("FINAL VERIFICATION SUMMARY")
print("="*60)
print(f"1. Total MCD GW/DTW wells in portal:  {len(df_gw)}")
print(f"2. GW wells INSIDE study boundary:    {len(gw_inside)}")
print(f"   - Used in the SWAT-MF model:       {len(gw_in_model)}")
print(f"   - NOT used in model:               {len(gw_in_nomodel)}")
print(f"3. GW wells OUTSIDE study boundary:   {len(gw_outside)}")
print(f"4. Verification: model wells in set:  {len(in_model)}/{len(model_ids)}")
if model_not_in_gw_inside:
    print(f"   Remaining {len(model_not_in_gw_inside)} model wells not in GW-inside set:")
    for wid in sorted(model_not_in_gw_inside):
        print(f"      {wid}")
