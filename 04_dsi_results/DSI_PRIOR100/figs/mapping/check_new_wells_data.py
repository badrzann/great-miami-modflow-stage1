"""
Download DTW (Depth to Water) time series for the ~30 inside-boundary GW wells
that are NOT currently in the model, check if they have data in 2003-2023,
and compute their mean head value (elevation - DTW).
"""
import requests, re, json, time, csv
import numpy as np
from datetime import datetime

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
BASE = "https://waterdata.mcdwater.org"

# --- Auth ---
r0 = session.get(f"{BASE}/Data/List", timeout=30)
m = re.search(r'__RequestVerificationToken.*?value="([^"]+)"', r0.text)
tok = m.group(1)
session.post(f"{BASE}/AcceptDisclaimer",
    data={"__RequestVerificationToken": tok, "returnUrl": "/Data/List"}, timeout=30)
print("Session established")

# --- Load GeoJSON and build locationId lookup ---
with open("D:/GMRW/_tmp/all_indicators_geojson.json") as f:
    gj = json.load(f)

loc_id_map = {}  # locationIdentifier -> locationId
loc_name_map = {}
for feat in gj['features']:
    lid = feat['properties']['locationIdentifier']
    loc_id_map[lid] = feat['properties']['locationId']
    loc_name_map[lid] = feat['properties']['location']

# --- Load model well IDs ---
model_ids = set()
with open("D:/GMRW/finalresult/obs_data/mcd_well_name_location_matches.csv") as f:
    for row in csv.DictReader(f):
        model_ids.add(row['mcd_location_id'])

# --- Identify inside-boundary NON-model GW wells ---
with open("D:/GMRW/finalresult/swatmf_run/modflow_GMRW.bas") as f:
    lines = f.readlines()

ibound_lines = []
in_ibound = False
for line in lines:
    if 'IBOUND' in line.upper() and not in_ibound:
        in_ibound = True; continue
    if in_ibound:
        ibound_lines.append(line.strip())
        if len(' '.join(ibound_lines).split()) >= 197 * 135:
            break
data = ' '.join(ibound_lines).split()[:197 * 135]
ibound = np.array([int(x) for x in data]).reshape(197, 135)
LAT_MAX, LON_MIN, LON_MAX = 40.710, -84.855, -83.585
DLAT = (40.710 - 39.185) / 197
DLON = (LON_MAX - LON_MIN) / 135

def inside(lon, lat):
    r = int((LAT_MAX - lat) / DLAT)
    c = int((lon - LON_MIN) / DLON)
    if 0 <= r < 197 and 0 <= c < 135:
        return ibound[r, c] != 0
    return False

# True GW well filter (exclude stream gauges, precip stations, quality sites)
NON_GW = {'BOLTON', 'ENGLEWOOD', 'HUFFMAN', 'MIAMISBURG', 'MIAMIVILLA'}
PRECIP_SHORT = {'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14',
                '15','16','17','18','19','20','21','22','23','24','25','26','27',
                '28','29','30','31','32','33','1A','1B','1B1','3B','6G','16B','16B1',
                '23A','23I','24A','AB','AB1','PR-2A'}

def is_gw(lid):
    if not lid: return False
    if lid.startswith('03') or lid in NON_GW or lid in PRECIP_SHORT: return False
    try:
        int(lid); return False
    except: pass
    if len(lid) <= 3 and lid.isalpha(): return False
    return True

target_wells = []
for feat in gj['features']:
    lid = feat['properties']['locationIdentifier']
    if not is_gw(lid) or lid in model_ids: continue
    lon, lat = feat['geometry']['coordinates']
    if inside(lon, lat):
        target_wells.append({
            'locationIdentifier': lid,
            'locationId': feat['properties']['locationId'],
            'location': feat['properties']['location'],
            'lon': lon, 'lat': lat
        })

print(f"\nTarget wells (inside boundary, not in model): {len(target_wells)}")
for w in target_wells:
    print(f"  {w['locationIdentifier']:15s}  {w['location']:25s}  id={w['locationId']}")

# --- Step 1: Get dataset list for each well ---
print("\n" + "="*65)
print("STEP 1: Finding 'Depth to Water' datasets")
print("="*65)

dtw_datasets = {}  # locationIdentifier -> dataset info

for w in target_wells:
    lid = w['locationIdentifier']
    loc_id = w['locationId']
    try:
        r = session.get(f"{BASE}/Data/GetDatasetStatList/{loc_id}", timeout=20)
        if r.status_code != 200:
            print(f"  {lid}: HTTP {r.status_code}")
            continue
        try:
            data = r.json()
        except:
            print(f"  {lid}: non-JSON response")
            continue
        if not data:
            print(f"  {lid}: empty response")
            continue
        # Find DTW dataset
        dtw = [d for d in data if 'depth' in str(d).lower() and 'water' in str(d).lower()]
        if dtw:
            dtw_datasets[lid] = {'well': w, 'datasets': dtw}
            print(f"  {lid}: found {len(dtw)} DTW dataset(s)")
        else:
            params = [d.get('Parameter', d.get('parameter', '?')) for d in data[:3]]
            print(f"  {lid}: no DTW, has: {params}")
    except Exception as e:
        print(f"  {lid}: ERROR {e}")
    time.sleep(0.15)

print(f"\nWells with DTW datasets: {len(dtw_datasets)}")

# --- Step 2: For wells with DTW data, test Export_Dataset ---
print("\n" + "="*65)
print("STEP 2: Testing Export_Dataset on one well")
print("="*65)

if dtw_datasets:
    first_lid = list(dtw_datasets.keys())[0]
    first_ds = dtw_datasets[first_lid]['datasets'][0]
    print(f"Testing on {first_lid}: {json.dumps(first_ds, indent=2)[:400]}")
    
    # Try to find the dataset ID
    ds_id = first_ds.get('Id') or first_ds.get('DatasetId') or first_ds.get('id')
    if ds_id:
        url = f"{BASE}/Data/Export_Dataset/{ds_id}"
        r = session.get(url, params={
            'startTime': '2003-01-01T00:00:00',
            'endTime': '2023-12-31T23:59:59',
            'timezone': 'UTC'
        }, timeout=30)
        print(f"\nExport_Dataset/{ds_id}: HTTP {r.status_code}")
        print(f"Content-Type: {r.headers.get('content-type','')}")
        print(f"First 500 chars:\n{r.text[:500]}")
else:
    print("No wells with DTW datasets found — checking first well response format")
    if target_wells:
        w = target_wells[0]
        r = session.get(f"{BASE}/Data/GetDatasetStatList/{w['locationId']}", timeout=20)
        print(f"GetDatasetStatList/{w['locationId']}: HTTP {r.status_code}")
        print(f"Response: {r.text[:800]}")
