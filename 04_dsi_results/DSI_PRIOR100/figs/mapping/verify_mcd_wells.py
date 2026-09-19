"""
Complete 4-point MCD well verification using real coordinates from /Map/Indicators.

1. Get ALL well coords from /Map/Indicators endpoint
2. Filter to Depth-to-Water wells specifically (parameter filter)
3. Count inside/outside IBOUND boundary
4. Verify all inside-boundary wells used in model
5. Generate clean verification map
"""
import requests, re, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ============================================================
# STEP 1: Fetch all well coordinates from /Map/Indicators
# ============================================================
print("=== Step 1: Fetching indicator data from portal ===")

BASE = "https://waterdata.mcdwater.org"
s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json",
    "Referer": BASE + "/Data/Map",
})

r0 = s.get(f"{BASE}/Data/List", timeout=30)
m = re.search(r'__RequestVerificationToken.*?value="([^"]+)"', r0.text)
tok = m.group(1)
s.post(f"{BASE}/AcceptDisclaimer", data={"__RequestVerificationToken": tok, "returnUrl": "/Data/Map"}, timeout=30)

# Get all indicators (no parameter filter = all 208)
r = s.post(f"{BASE}/Map/Indicators", data={
    "interval": "Latest",
    "legend": "1",
    "utcOffset": "240",
    "date": "2026-04-09",
    "wkid": "4326",
}, timeout=30)
all_features = r.json()['features']
print(f"Total features from /Map/Indicators: {len(all_features)}")

# Save full geojson
with open("D:/GMRW/_tmp/all_indicators_geojson.json", "w") as f2:
    json.dump(r.json(), f2, indent=2)

# Get DTW-only features
r_dtw = s.post(f"{BASE}/Map/Indicators", data={
    "interval": "Latest",
    "parameters[0]": "Depth to Water",
    "value": "LATEST",
    "type": "LATEST",
    "legend": "1",
    "utcOffset": "240",
    "date": "2026-04-09",
    "wkid": "4326",
}, timeout=30)
dtw_data = r_dtw.json()
dtw_features = dtw_data['features']
print(f"DTW-only features from /Map/Indicators: {len(dtw_features)}")
with open("D:/GMRW/_tmp/dtw_indicators_geojson.json", "w") as f2:
    json.dump(dtw_data, f2, indent=2)

# ============================================================
# STEP 2: Load model boundary from IBOUND
# ============================================================
print("\n=== Step 2: Loading IBOUND boundary ===")

bas_file = "D:/GMRW/finalresult/swatmf_run/modflow_GMRW.bas"

with open(bas_file, "r") as f2:
    lines = f2.readlines()

NROW, NCOL = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX = 39.185, 40.710
DLON = (LON_MAX - LON_MIN) / NCOL   # ~0.009481
DLAT = (LAT_MAX - LAT_MIN) / NROW   # ~0.007614

# Parse IBOUND
ibound = np.zeros((NROW, NCOL), dtype=int)
ibound_values = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        continue
    parts = stripped.split()
    for part in parts:
        if part.lstrip('-').isdigit():
            ibound_values.append(int(part))
    if len(ibound_values) >= NROW * NCOL:
        break

vals = ibound_values[:NROW * NCOL]
for i, v in enumerate(vals):
    row_i = i // NCOL
    col_i = i % NCOL
    ibound[row_i, col_i] = v
    
print(f"IBOUND loaded: {(ibound != 0).sum()} active cells out of {NROW*NCOL} total")

# Function to check if (lon, lat) is inside study boundary
def is_inside_boundary(lon, lat):
    """Check if point is inside active IBOUND cells."""
    col_f = (lon - LON_MIN) / DLON
    row_f = (LAT_MAX - lat) / DLAT
    col_i = int(col_f)
    row_i = int(row_f)
    if row_i < 0 or row_i >= NROW or col_i < 0 or col_i >= NCOL:
        return False
    return ibound[row_i, col_i] != 0

# Build boundary segments for plotting
print("Building boundary segments...")
boundary_segments = []
for i in range(NROW):
    for j in range(NCOL):
        if ibound[i, j] != 0:
            lon_c = LON_MIN + (j + 0.5) * DLON
            lat_c = LAT_MAX - (i + 0.5) * DLAT
            neighbors = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
            for ni_, nj_ in neighbors:
                if ni_ < 0 or ni_ >= NROW or nj_ < 0 or nj_ >= NCOL or ibound[ni_,nj_] == 0:
                    if ni_ == i-1: # top
                        x1 = LON_MIN + j*DLON; x2 = LON_MIN + (j+1)*DLON
                        y = LAT_MAX - i*DLAT
                        boundary_segments.append(([x1,x2],[y,y]))
                    elif ni_ == i+1: # bottom
                        x1 = LON_MIN + j*DLON; x2 = LON_MIN + (j+1)*DLON
                        y = LAT_MAX - (i+1)*DLAT
                        boundary_segments.append(([x1,x2],[y,y]))
                    elif nj_ == j-1: # left
                        x = LON_MIN + j*DLON
                        y1 = LAT_MAX - i*DLAT; y2 = LAT_MAX - (i+1)*DLAT
                        boundary_segments.append(([x,x],[y1,y2]))
                    elif nj_ == j+1: # right
                        x = LON_MIN + (j+1)*DLON
                        y1 = LAT_MAX - i*DLAT; y2 = LAT_MAX - (i+1)*DLAT
                        boundary_segments.append(([x,x],[y1,y2]))

print(f"Boundary segments: {len(boundary_segments)}")

# ============================================================
# STEP 3: Count inside/outside boundary for ALL features
# ============================================================
print("\n=== Step 3: Classify wells inside/outside boundary ===")

# Build dataframe from all features
all_rows = []
for feat in all_features:
    lon, lat = feat['geometry']['coordinates']
    props = feat['properties']
    inside = is_inside_boundary(lon, lat)
    all_rows.append({
        'locationId': props['locationId'],
        'locationIdentifier': props['locationIdentifier'],
        'locationName': props['location'],
        'lon': lon,
        'lat': lat,
        'inside': inside,
    })
df_all = pd.DataFrame(all_rows)

# Build dataframe from DTW features
dtw_rows = []
for feat in dtw_features:
    lon, lat = feat['geometry']['coordinates']
    props = feat['properties']
    inside = is_inside_boundary(lon, lat)
    dtw_rows.append({
        'locationId': props['locationId'],
        'locationIdentifier': props['locationIdentifier'],
        'locationName': props['location'],
        'lon': lon,
        'lat': lat,
        'inside': inside,
    })
df_dtw = pd.DataFrame(dtw_rows)

print(f"\nAll MCD locations (208 indicator features):")
print(f"  Inside boundary:  {df_all['inside'].sum()}")
print(f"  Outside boundary: {(~df_all['inside']).sum()}")

print(f"\nDTW-specific locations ({len(df_dtw)} features):")
print(f"  Inside boundary:  {df_dtw['inside'].sum()}")
print(f"  Outside boundary: {(~df_dtw['inside']).sum()}")

dtw_inside = df_dtw[df_dtw['inside']].copy()
dtw_outside = df_dtw[~df_dtw['inside']].copy()
print(f"\nDTW wells INSIDE boundary ({len(dtw_inside)}):")
for _, row in dtw_inside.iterrows():
    print(f"  {row['locationIdentifier']:15s}  {row['locationName'][:40]:<40s}  ({row['lat']:.4f}, {row['lon']:.4f})")

print(f"\nDTW wells OUTSIDE boundary ({len(dtw_outside)}):")
for _, row in dtw_outside.iterrows():
    print(f"  {row['locationIdentifier']:15s}  {row['locationName'][:40]:<40s}  ({row['lat']:.4f}, {row['lon']:.4f})")

# ============================================================
# STEP 4: Cross-check with model wells (85 CSV wells)
# ============================================================
print("\n=== Step 4: Cross-check with model wells ===")

obs_csv = "D:/GMRW/finalresult/obs_data/mcd_well_name_location_matches.csv"
df_model = pd.read_csv(obs_csv)
print(f"Model wells in CSV: {len(df_model)}")
print(f"CSV columns: {list(df_model.columns)}")

# Map locationIdentifier to model wells
# The CSV has mcd_location_id which corresponds to locationIdentifier
if 'mcd_location_id' in df_model.columns:
    model_ids = set(df_model['mcd_location_id'].astype(str).str.strip())
else:
    # Try other cols
    print(f"CSV columns: {list(df_model.columns)}")
    id_col = [c for c in df_model.columns if 'id' in c.lower() or 'loc' in c.lower()]
    print(f"ID columns: {id_col}")
    model_ids = set()

print(f"\nModel well IDs (first 10): {sorted(list(model_ids))[:10]}")

# Check which DTW wells inside boundary are in the model
dtw_inside_ids = set(dtw_inside['locationIdentifier'].astype(str).str.strip())
print(f"DTW inside boundary IDs (first 10): {sorted(list(dtw_inside_ids))[:10]}")

in_model = dtw_inside_ids & model_ids
not_in_model = dtw_inside_ids - model_ids
extra_in_model = model_ids - dtw_inside_ids

print(f"\nDTW wells inside boundary that ARE in the model: {len(in_model)}")
print(f"DTW wells inside boundary NOT in the model: {len(not_in_model)}")
print(f"Model wells NOT matched to DTW-inside wells: {len(extra_in_model)}")

if not_in_model:
    print(f"\nDTW wells inside boundary missing from model:")
    for wid in sorted(not_in_model):
        row = dtw_inside[dtw_inside['locationIdentifier'].astype(str).str.strip() == wid].iloc[0]
        print(f"  {wid:15s}  {row['locationName'][:50]}")

if extra_in_model:
    print(f"\nModel wells not found in DTW-inside set:")
    for wid in sorted(extra_in_model):
        mrow = df_model[df_model['mcd_location_id'].astype(str).str.strip() == wid].iloc[0] if 'mcd_location_id' in df_model.columns else None
        if mrow is not None:
            print(f"  {wid:15s}  obsnme={mrow.get('obsnme','?')} local={mrow.get('local_well_name','?')}")
        else:
            print(f"  {wid}")

# ============================================================
# STEP 5: Create verification map
# ============================================================
print("\n=== Step 5: Creating verification map ===")

fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# Plot boundary
for xdata, ydata in boundary_segments:
    ax.plot(xdata, ydata, 'k-', linewidth=0.4, alpha=0.6)

# Separate DTW well categories
dtw_inside_model = df_dtw[
    df_dtw['inside'] & 
    df_dtw['locationIdentifier'].astype(str).str.strip().isin(model_ids)
]
dtw_inside_nonmodel = df_dtw[
    df_dtw['inside'] & 
    ~df_dtw['locationIdentifier'].astype(str).str.strip().isin(model_ids)
]

# Plot DTW wells inside boundary - in model (blue triangles)
ax.scatter(dtw_inside_model['lon'], dtw_inside_model['lat'],
          marker='^', s=60, color='#1f77b4', zorder=5, linewidths=0.5, edgecolors='navy',
          label=f'MCD DTW wells in model, inside boundary (n={len(dtw_inside_model)})')

# Plot DTW wells inside boundary - NOT in model (orange circles)
if len(dtw_inside_nonmodel) > 0:
    ax.scatter(dtw_inside_nonmodel['lon'], dtw_inside_nonmodel['lat'],
              marker='o', s=80, color='orange', zorder=5, linewidths=0.8, edgecolors='darkorange',
              label=f'MCD DTW wells inside boundary, NOT in model (n={len(dtw_inside_nonmodel)})')

# Plot DTW wells outside boundary (red X)
ax.scatter(dtw_outside['lon'], dtw_outside['lat'],
          marker='x', s=60, color='red', zorder=3, linewidths=1.5,
          label=f'MCD DTW wells outside boundary (n={len(dtw_outside)})')

# Labels for model wells (abbreviated)
for _, row in dtw_inside_model.iterrows():
    ax.annotate(row['locationIdentifier'], (row['lon'], row['lat']),
               textcoords='offset points', xytext=(3, 3), fontsize=4, color='#1f77b4', zorder=6)

# Labels for inside-non-model wells
for _, row in dtw_inside_nonmodel.iterrows():
    ax.annotate(row['locationIdentifier'], (row['lon'], row['lat']),
               textcoords='offset points', xytext=(3, 3), fontsize=5, color='darkorange', zorder=6)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'MCD Depth-to-Water Well Verification\n'
             f'Total DTW: {len(df_dtw)}  |  Inside: {len(dtw_inside)} ({len(dtw_inside_model)} in model, {len(dtw_inside_nonmodel)} not in model)  |  Outside: {len(dtw_outside)}', 
             fontsize=13)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

out_dir = "D:/GMRW/finalresult/fig"
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{out_dir}/fig_mcd_dtw_verification.png", dpi=150, bbox_inches='tight')
plt.savefig(f"{out_dir}/fig_mcd_dtw_verification.pdf", bbox_inches='tight')
print(f"Saved to {out_dir}/fig_mcd_dtw_verification.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
print(f"1. Total MCD DTW wells:           {len(df_dtw)}")
print(f"2. DTW wells inside boundary:     {len(dtw_inside)}")
print(f"   - Used in model:               {len(in_model)}")
print(f"   - NOT used in model:           {len(not_in_model)}")
print(f"3. DTW wells outside boundary:    {len(dtw_outside)}")
print(f"4. Model wells verified as in:    {len(in_model)}/{len(model_ids)}")
print(f"")
print(f"Model wells not in DTW-inside:    {len(extra_in_model)}")
if extra_in_model:
    print(f"  (These are model wells that may have classification issues or boundary effects)")
