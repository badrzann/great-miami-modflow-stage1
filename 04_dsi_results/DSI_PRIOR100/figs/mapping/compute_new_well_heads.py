"""
Compare existing 85-well head values vs. computed from Excel,
then compute mean head (2001-2023) for new inside-boundary wells.

Steps:
1. Load MODFLOW DIS TOP array → land surface elevation grid
2. Load existing 85 wells + their head values
3. For matching wells in Excel, compute DTW mean → head = LSE - DTW*0.3048
4. Compare existing vs. computed
5. Compute heads for 24 new wells from Excel
6. Output combined table
"""
import csv, openpyxl, struct, re
import numpy as np
from datetime import datetime

# ─── MODFLOW DIS → land surface elevation ────────────────────────────────────
DIS_FILE = r"D:\GMRW\finalresult\swatmf_run\modflow_GMRW.dis"
NROW, NCOL = 197, 135
LON_MIN, LON_MAX = -84.855, -83.585
LAT_MIN, LAT_MAX =  39.185,  40.710
DLAT = (LAT_MAX - LAT_MIN) / NROW
DLON = (LON_MAX - LON_MIN) / NCOL

def load_top():
    vals = []
    in_top = False
    with open(DIS_FILE) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            if 'TOP' in s.upper() and len(vals) == 0:
                in_top = True; continue
            if in_top:
                parts = s.split()
                for p in parts:
                    try: vals.append(float(p))
                    except: pass
                if len(vals) >= NROW * NCOL:
                    break
    return np.array(vals[:NROW * NCOL]).reshape(NROW, NCOL)

def lon_lat_to_rc(lon, lat):
    r = int((LAT_MAX - lat) / DLAT)
    c = int((lon - LON_MIN) / DLON)
    r = max(0, min(NROW - 1, r))
    c = max(0, min(NCOL - 1, c))
    return r, c

print("Loading MODFLOW TOP array...")
TOP = load_top()
print(f"  TOP range: {TOP.min():.1f} – {TOP.max():.1f} m")

# ─── Load existing 85 wells ───────────────────────────────────────────────────
wells85 = {}  # mcd_id -> {obsnme, local_name, row, col, lat, lon, head_existing}
with open(r"D:\GMRW\finalresult\obs_data\mcd_well_name_location_matches.csv") as f:
    for row in csv.DictReader(f):
        wells85[row['mcd_location_id']] = {
            'obsnme': row['obsnme'],
            'local_name': row['local_well_name'],
            'mcd_id': row['mcd_location_id'],
            'row': int(row['row']), 'col': int(row['col']),
            'lat': float(row['lat']), 'lon': float(row['lon']),
        }

head_rows = list(csv.DictReader(open(r"D:\GMRW\finalresult\obs_data\head_mean_85.csv")))
head_by_wellid = {r['well_id']: float(r['head_mean_m']) for r in head_rows}
mcd_ids_ordered = list(wells85.keys())
for i, mcd_id in enumerate(mcd_ids_ordered):
    well_id = str(i + 1)
    wells85[mcd_id]['head_existing'] = head_by_wellid.get(well_id, None)
    wells85[mcd_id]['well_id'] = well_id
    # Land surface elevation at model grid cell
    r = wells85[mcd_id]['row'] - 1  # 1-indexed -> 0-indexed
    c = wells85[mcd_id]['col'] - 1
    wells85[mcd_id]['lse_model'] = float(TOP[r, c])

# ─── Excel parsing function ───────────────────────────────────────────────────
def parse_dtw_sheet(wb, sheet_name, start_year=2001, end_year=2023):
    """
    Parse a DTW sheet. Data starts at row 7 (0-indexed: row 6).
    Columns: C=Timestamp  D=Value(ft)
    Returns: (mean_dtw_ft, n_readings, date_range, unit) or None
    """
    ws = wb[sheet_name]
    dtw_vals = []
    unit = 'ft'
    # Detect unit from row 6 (0-indexed 5)
    rows_iter = ws.iter_rows(values_only=True)
    for i, row in enumerate(rows_iter):
        if i == 5:  # Row 6: column header
            if row and len(row) >= 4 and row[3]:
                header_str = str(row[3])
                if 'm)' in header_str.lower() or 'meter' in header_str.lower():
                    unit = 'm'
        if i < 6:  # Skip 6 header rows
            continue
        if row is None: continue
        ts = row[2] if len(row) > 2 else None
        val = row[3] if len(row) > 3 else None
        if ts is None or val is None: continue
        # Filter date range
        if isinstance(ts, datetime):
            if ts.year < start_year or ts.year > end_year:
                continue
        else:
            continue
        try:
            val = float(val)
            if val > 0:  # DTW should be positive
                dtw_vals.append(val)
        except:
            continue
    if not dtw_vals:
        return None
    return {
        'mean_dtw_ft': np.mean(dtw_vals) if unit == 'ft' else np.mean(dtw_vals) * 3.28084,
        'mean_dtw_m': (np.mean(dtw_vals) * 0.3048) if unit == 'ft' else np.mean(dtw_vals),
        'n': len(dtw_vals),
        'unit': unit,
        'min_date': min(r[2] for r in ws.iter_rows(values_only=True) if r and r[2] and isinstance(r[2], datetime)).year if False else None,
    }

# ─── Map of new wells to their Excel file + sheet name ───────────────────────
FILE1 = r"D:\GMRW\finalresult\obs_data\Groundwater Well\Groundwater wells - 1.xlsx"
FILE2 = r"D:\GMRW\finalresult\obs_data\Groundwater Well\Groundwater wells - 2.xlsx"

# Also map EXISTING model wells that appear in Excel for comparison
EXISTING_IN_EXCEL = {
    'SHE00066': (FILE1, 'Location - SHE00066 (SH-5)'),
    'MON00014': (FILE1, 'MT-6 MON00014'),
    'MON00022': (FILE1, 'MT-1256 MON00022'),
    'MON00049': (FILE1, 'MT-49 MON00049'),
    'MIA00205': (FILE1, 'MI-205 MIA00205'),
    'CLA00007': (FILE2, 'CL-7 CLA00007'),
    'CLA10011': (FILE2, 'CLAMRTC-D CLA10011'),
    'CLA10012': (FILE2, 'CLAMRTC-S CLA10012'),
    'CHA10010': (FILE2, 'MAD55 CHA10010'),  # NOT in model but in Excel
    'SHE00089': (FILE2, 'SH-76 SHE00089'),  # NOT in model but in Excel
}

NEW_WELLS_EXCEL = {
    # MCD_ID:  (file, sheet_name, local_name, lon, lat)
    'BUT00007': (FILE1, 'Location - BU-7 BUT00007',    'BU-7',       -84.581,   39.338),
    'BUT00008': (FILE1, 'Location - BU-8 BUT00008',    'BU-8',       -84.560,   39.347),
    'BUT00018': (FILE1, 'Location - BU-18 BUT00018',   'BU-18',      -84.578,   39.328),
    'BUT10016': (FILE1, 'BUTNMC-D BUT10016',           'BUTNMC-D',   -84.432,   39.447),
    'CL-10':    (FILE1, 'CL-10 CL-10',                'CL-10',      -83.948,   39.960),
    'CLA00001': (FILE1, 'Foreman CLA00001',            'Foreman',    -83.954,   39.971),
    'CLA00007': (FILE2, 'CL-7 CLA00007',               'CL-7',       -84.023,   39.976),
    'CLA00012': (FILE2, 'Johnson CLA00012',            'Johnson',    -83.988,   40.003),
    'CLA00013': (FILE2, 'Dunn CLA00013',               'Dunn',       -84.010,   39.975),
    'CLA10011': (FILE2, 'CLAMRTC-D CLA10011',          'CLAMRTC-D',  -84.008,   40.008),
    'CLA10012': (FILE2, 'CLAMRTC-S CLA10012',          'CLAMRTC-S',  -84.008,   40.008),
    'CLA10017': (FILE2, 'DAY509D CLA10017',            'DAY509D',    -83.993,   39.916),
    'CHA10010': (FILE2, 'MAD55 CHA10010',              'MAD55',      -83.886,   40.068),
    'MIA00002': (FILE2, 'Holland MIA00002',            'Holland',    -84.232,   40.002),
    'MIA00205': (FILE1, 'MI-205 MIA00205',             'MI-205',     -84.248,   40.029),
    'MON00014': (FILE1, 'MT-6 MON00014',               'MT-6',       -84.186,   39.759),
    'MON00022': (FILE1, 'MT-1256 MON00022',            'MT-1256',    -84.170,   39.637),
    'MON00049': (FILE1, 'MT-49 MON00049',              'MT-49',      -84.139,   39.674),
    'MON00260': (FILE1, 'MT260 MON00260',              'MT260',      -84.187,   39.781),
    'PRE00022': (FILE1, 'Wogoman PRE00022',            'Wogoman',    -84.569,   39.697),
    'SHE00044': (FILE2, 'RITZHAUPT1 SHE00044',        'RITZHAUPT1', -84.154,   40.211),
    'SHE00051': (FILE2, 'Groff SHE00051',              'Groff',      -84.286,   40.261),
    'SHE00066': (FILE1, 'Location - SHE00066 (SH-5)', 'SH-5',       -84.175,   40.285),
    'SHE00089': (FILE2, 'SH-76 SHE00089',             'SH-76',      -84.145,   40.211),
}

# ─── Parse Excel files ────────────────────────────────────────────────────────
print("\nLoading Excel files...")
wb1 = openpyxl.load_workbook(FILE1, read_only=True, data_only=True)
wb2 = openpyxl.load_workbook(FILE2, read_only=True, data_only=True)

def get_wb(fpath):
    return wb1 if fpath == FILE1 else wb2

# ─── SECTION 1: Compare existing model wells vs Excel ────────────────────────
print("\n" + "="*80)
print("SECTION 1: EXISTING 85 MODEL WELLS — Head Comparison")
print("  (showing only wells that also appear in the Excel files)")
print("="*80)
print(f"\n{'#':<4} {'MCD_ID':<15} {'Local Name':<28} {'Existing (m)':<14} {'LSE (m)':<10} {'DTW mean (ft)':<15} {'Computed (m)':<14} {'Diff (m)'}")
print("-"*110)

compared = []
for mcd_id, info in wells85.items():
    if mcd_id not in EXISTING_IN_EXCEL:
        continue
    fpath, sheet = EXISTING_IN_EXCEL[mcd_id]
    wb = get_wb(fpath)
    result = parse_dtw_sheet(wb, sheet)
    lse = info['lse_model']
    head_existing = info['head_existing']
    if result and result['n'] > 0:
        computed = lse - result['mean_dtw_m']
        diff = computed - head_existing if head_existing else None
        compared.append({
            'mcd_id': mcd_id, 'local': info['local_name'],
            'head_existing': head_existing, 'lse': lse,
            'dtw_ft': result['mean_dtw_ft'], 'dtw_m': result['mean_dtw_m'],
            'computed': computed, 'diff': diff, 'n': result['n']
        })
        print(f"{info['well_id']:<4} {mcd_id:<15} {info['local_name']:<28} {head_existing:<14.3f} {lse:<10.1f} {result['mean_dtw_ft']:<15.2f} {computed:<14.3f} {diff:+.3f}  (n={result['n']})")

# ─── SECTION 2: New wells from Excel ─────────────────────────────────────────
print("\n\n" + "="*80)
print("SECTION 2: NEW INSIDE-BOUNDARY WELLS — Mean Head 2001-2023")
print("="*80)
print(f"\n{'MCD_ID':<15} {'Local Name':<28} {'LSE (m)':<10} {'DTW mean (ft)':<15} {'DTW mean (m)':<14} {'Head (m)':<12} {'n readings'}")
print("-"*100)

new_wells_out = []
not_enough_data = []

for mcd_id, (fpath, sheet, local_name, lon, lat) in NEW_WELLS_EXCEL.items():
    if mcd_id in wells85:  # already in model
        continue
    wb = get_wb(fpath)
    r, c = lon_lat_to_rc(lon, lat)
    lse = float(TOP[r, c])
    result = parse_dtw_sheet(wb, sheet)
    if result and result['n'] >= 10:
        head_m = lse - result['mean_dtw_m']
        new_wells_out.append({
            'mcd_id': mcd_id, 'local': local_name,
            'lse': lse, 'dtw_ft': result['mean_dtw_ft'],
            'dtw_m': result['mean_dtw_m'], 'head': head_m,
            'n': result['n'], 'lon': lon, 'lat': lat
        })
        print(f"{mcd_id:<15} {local_name:<28} {lse:<10.1f} {result['mean_dtw_ft']:<15.2f} {result['mean_dtw_m']:<14.3f} {head_m:<12.3f} {result['n']}")
    else:
        n = result['n'] if result else 0
        not_enough_data.append((mcd_id, local_name, n))
        print(f"{mcd_id:<15} {local_name:<28} {lse:<10.1f} {'--':<15} {'--':<14} {'--':<12} INSUFFICIENT DATA (n={n})")

# ─── Close workbooks ──────────────────────────────────────────────────────────
wb1.close(); wb2.close()

# ─── SECTION 3: Combined table ────────────────────────────────────────────────
print("\n\n" + "="*80)
print("SECTION 3: COMBINED OBSERVATION WELLS — ALL 85 + NEW VALID WELLS")
print("="*80)
print(f"\n{'#':<5} {'MCD_ID':<15} {'Local Name':<30} {'Head (m)':<12} {'Source'}")
print("-"*75)

# Print existing 85
for i, (mcd_id, info) in enumerate(wells85.items(), 1):
    flag = ""
    if mcd_id in EXISTING_IN_EXCEL:
        cmp = next((c for c in compared if c['mcd_id'] == mcd_id), None)
        if cmp:
            flag = f"  [Excel computed: {cmp['computed']:.3f}m, diff={cmp['diff']:+.3f}m]"
    print(f"{i:<5} {mcd_id:<15} {info['local_name']:<30} {info['head_existing']:<12.3f} original{flag}")

# Print new wells
for j, w in enumerate(new_wells_out, 1):
    print(f"{85+j:<5} {w['mcd_id']:<15} {w['local']:<30} {w['head']:<12.3f} Excel 2001-2023 (n={w['n']})")

print(f"\n{'='*80}")
print(f"Total existing wells:   85")
print(f"New wells with data:    {len(new_wells_out)}")
print(f"New wells insufficient: {len(not_enough_data)} → {[x[0] for x in not_enough_data]}")
print(f"Total combined:         {85 + len(new_wells_out)}")
