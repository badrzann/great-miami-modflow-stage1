"""
Scan all sheets in both Groundwater Wells Excel files and check if
the 32 not-in-model inside-boundary wells are present.
"""
import openpyxl
import re

TARGET_WELLS = {
    'BUT00007', 'BUT00008', 'BUT00018', 'BUT00066', 'BUT00283', 'BUT10016',
    'CHA10010', 'CL-10', 'CLA00001', 'CLA00007', 'CLA00012', 'CLA00013',
    'CLA10011', 'CLA10012', 'CLA10017',
    'DAR00002', 'DAR00003', 'DAR00006',
    'MIA00002', 'MIA00205',
    'MON00003', 'MON00014', 'MON00022', 'MON00023', 'MON00049', 'MON00260',
    'PRE00022', 'PRE10007',
    'SHE00044', 'SHE00051', 'SHE00066', 'SHE00089',
    # Also alternate names from portal
    'BU-7', 'BU-8', 'BU-18', 'BUT00066', 'BU283', 'BUTNMC-D',
    'MAD55', 'CL-10', 'CLA00001', 'CL-7',
    'CLA00012', 'CLA00013', 'CLAMRTC-D', 'CLAMRTC-S', 'DAY509D',
    'D-2', 'DAR00003', 'Eidson Woods',
    'MIA00002', 'MI-205',
    'NRSF3', 'MT-6', 'MT-1256', 'NRSF4', 'MT-49', 'MT260',
    'PRE00022', 'PRE10007',
    'SHE00044', 'SHE00051', 'SH-5', 'SH-76',
}

files = [
    r"D:\GMRW\finalresult\obs_data\Groundwater Well\Groundwater wells - 1.xlsx",
    r"D:\GMRW\finalresult\obs_data\Groundwater Well\Groundwater wells - 2.xlsx",
]

all_found = {}  # well_id -> [(file, sheet, row, context)]

for fpath in files:
    fname = fpath.split("\\")[-1]
    print(f"\n{'='*65}")
    print(f"FILE: {fname}")
    print(f"{'='*65}")
    
    wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
    print(f"Sheets ({len(wb.sheetnames)}): {wb.sheetnames}\n")
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"  Sheet: {sheet_name!r}  ({ws.max_row} rows x {ws.max_column} cols)")
        
        # Print header row
        header_row = None
        found_in_sheet = {}
        
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            # Grab header
            if row_idx == 1:
                header_row = [str(c) if c is not None else '' for c in row]
                print(f"    Header: {header_row[:8]}")
                continue
            
            # Search each cell string for target well IDs
            row_str = ' '.join(str(c) for c in row if c is not None)
            for well in TARGET_WELLS:
                # Match whole word to avoid partial matches
                if re.search(r'\b' + re.escape(well) + r'\b', row_str, re.IGNORECASE):
                    if well not in found_in_sheet:
                        found_in_sheet[well] = []
                    found_in_sheet[well].append((row_idx, row[:6]))
        
        if found_in_sheet:
            print(f"    *** FOUND {len(found_in_sheet)} target wells in this sheet ***")
            for well, hits in sorted(found_in_sheet.items()):
                print(f"      {well}: {len(hits)} row(s), first at row {hits[0][0]}: {hits[0][1]}")
                key = (well, fname, sheet_name)
                all_found[key] = hits
        else:
            print(f"    (no target wells found)")
        
        # To avoid memory issues with large files, limit scan
        if row_idx > 500000:
            print(f"    (truncated at 500k rows)")
            break
    
    wb.close()

# Summary
print(f"\n{'='*65}")
print("SUMMARY: Target wells found across all files/sheets")
print(f"{'='*65}")

found_well_ids = set(k[0] for k in all_found.keys())
# Normalize: map alternate names to IDs
id_to_alt = {
    'BU-7': 'BUT00007', 'BU-8': 'BUT00008', 'BU-18': 'BUT00018',
    'BU283': 'BUT00283', 'BUTNMC-D': 'BUT10016', 'MAD55': 'CHA10010',
    'CL-7': 'CLA00007', 'CLAMRTC-D': 'CLA10011', 'CLAMRTC-S': 'CLA10012',
    'DAY509D': 'CLA10017', 'D-2': 'DAR00002', 'Eidson Woods': 'DAR00006',
    'MI-205': 'MIA00205', 'NRSF3': 'MON00003', 'MT-6': 'MON00014',
    'MT-1256': 'MON00022', 'NRSF4': 'MON00023', 'MT-49': 'MON00049',
    'MT260': 'MON00260', 'SH-5': 'SHE00066', 'SH-76': 'SHE00089',
}
canonical_ids = {'BUT00007', 'BUT00008', 'BUT00018', 'BUT00066', 'BUT00283', 'BUT10016',
    'CHA10010', 'CL-10', 'CLA00001', 'CLA00007', 'CLA00012', 'CLA00013',
    'CLA10011', 'CLA10012', 'CLA10017', 'DAR00002', 'DAR00003', 'DAR00006',
    'MIA00002', 'MIA00205', 'MON00003', 'MON00014', 'MON00022', 'MON00023',
    'MON00049', 'MON00260', 'PRE00022', 'PRE10007',
    'SHE00044', 'SHE00051', 'SHE00066', 'SHE00089'}

resolved_found = set()
for w in found_well_ids:
    resolved_found.add(id_to_alt.get(w, w))

print(f"\nFound in Excel (canonical IDs): {len(resolved_found & canonical_ids)}")
for w in sorted(resolved_found & canonical_ids):
    print(f"  YES  {w}")

not_found = canonical_ids - resolved_found
print(f"\nNOT found in Excel: {len(not_found)}")
for w in sorted(not_found):
    print(f"  NO   {w}")
