"""
Build grid-level land use and soil arrays using:
1. river2grid.txt for subbasin-to-grid seed mapping
2. Nearest-neighbor to extend subbasins to all active cells
3. HRU data for dominant land use and soil per subbasin
"""
import os, glob, re
import numpy as np
from scipy import ndimage

NROW, NCOL = 197, 135

# ============================================================
# 1. Parse IBOUND
# ============================================================
print("1. Parsing IBOUND...")
ibound = np.zeros((NROW, NCOL), dtype=int)
with open(r'D:\GMRW\swat-modflow files\modflow_GMRW.bas') as f:
    for line in f:
        if 'IBOUND' in line.upper():
            break
    row = 0
    for line in f:
        vals = line.split()
        if len(vals) == NCOL:
            ibound[row, :] = [int(v) for v in vals]
            row += 1
            if row >= NROW:
                break
active_mask = ibound > 0
print(f"   Active cells: {np.sum(active_mask)}")

# ============================================================
# 2. Parse river2grid: river cells with subbasin IDs
# ============================================================
print("2. Parsing river2grid...")
river_cells = []  # list of (row, col, subbasin_id)
with open(r'D:\GMRW\swat-modflow files\swatmf_river2grid.txt') as f:
    n_riv = int(f.readline().strip())
    for _ in range(n_riv):
        parts = f.readline().split()
        riv_idx = int(parts[0])
        grid_cell = int(parts[1])  # 1-indexed linear cell index
        layer = int(parts[2])
        sub_parts = f.readline().split()
        sub_id = int(sub_parts[0])  # take first subbasin if multiple
        cond_parts = f.readline().split()
        conductance = float(cond_parts[0])  # take first value
        r = (grid_cell - 1) // NCOL
        c = (grid_cell - 1) % NCOL
        river_cells.append((r, c, sub_id))

print(f"   River cells: {len(river_cells)}")
unique_subs = set(rc[2] for rc in river_cells)
print(f"   Unique subbasins from river cells: {len(unique_subs)}")

# ============================================================
# 3. Build subbasin grid using nearest-neighbor from river cells
# ============================================================
print("3. Building subbasin grid...")
# Create subbasin seed grid (0 = unknown)
sub_grid = np.zeros((NROW, NCOL), dtype=int)
for r, c, sub_id in river_cells:
    if 0 <= r < NROW and 0 <= c < NCOL:
        sub_grid[r, c] = sub_id

# Use distance transform to fill unknown active cells with nearest subbasin
# Create a mask where subbasin is known
known_mask = sub_grid > 0
# For cells without known subbasin, find nearest cell that has one
# Use scipy's distance_transform_edt with indices
_, nearest_idx = ndimage.distance_transform_edt(~known_mask, return_distances=True, return_indices=True)
# Fill all active cells with their nearest subbasin
for r in range(NROW):
    for c in range(NCOL):
        if active_mask[r, c] and sub_grid[r, c] == 0:
            nr, nc = nearest_idx[0, r, c], nearest_idx[1, r, c]
            sub_grid[r, c] = sub_grid[nr, nc]

# Mask inactive cells
sub_grid[~active_mask] = 0
assigned = np.sum(sub_grid > 0)
print(f"   Cells with subbasin assignment: {assigned}")
print(f"   Unique subbasins in grid: {len(set(sub_grid[sub_grid > 0]))}")

# ============================================================
# 4. Parse HRU headers and .sol files
# ============================================================
print("4. Parsing HRU headers...")
# Build: subbasin -> list of (luse, soil, hru_fr)
sub_hru_props = {}  # subbasin_id -> [(luse, soil, hru_fr), ...]

hru_files = glob.glob(r'D:\GMRW\swat-modflow files\*.hru')
for fpath in hru_files:
    with open(fpath) as fh:
        header = fh.readline().strip()
        # Get HRU_FR from second line
        line2 = fh.readline().strip()
    
    m = re.search(r'Subbasin:(\d+)\s+HRU:\d+\s+Luse:(\S+)\s+Soil:\s*(\S+)', header)
    if m:
        sub_id = int(m.group(1))
        luse = m.group(2)
        soil = m.group(3)
        # Extract HRU_FR from line 2
        hru_fr = float(line2.split('|')[0].strip()) if '|' in line2 else 0.0
        sub_hru_props.setdefault(sub_id, []).append((luse, soil, hru_fr))

print(f"   Parsed {sum(len(v) for v in sub_hru_props.values())} HRUs across {len(sub_hru_props)} subbasins")

# Parse .sol files for soil hydrologic group
print("   Parsing .sol files for hydrologic groups...")
soil_groups = {}
sol_files = glob.glob(r'D:\GMRW\swat-modflow files\*.sol')
for fpath in sol_files:
    with open(fpath) as fh:
        header = fh.readline()
        m_soil = re.search(r'Soil:\s*(\S+)', header)
        soil_name = m_soil.group(1) if m_soil else None
        for line in fh:
            if 'Soil Hydrologic Group' in line:
                grp = line.split(':')[1].strip()
                if soil_name and soil_name not in soil_groups:
                    soil_groups[soil_name] = grp
                break
print(f"   Soil hydrologic groups: {soil_groups}")

# ============================================================
# 5. Compute dominant land use and soil per subbasin
# ============================================================
print("5. Computing dominant luse/soil per subbasin...")
sub_dom_luse = {}
sub_dom_soil = {}
sub_dom_soilgrp = {}

for sub_id, props in sub_hru_props.items():
    # Aggregate land use by total HRU_FR
    luse_area = {}
    soil_area = {}
    soilgrp_area = {}
    for luse, soil, hru_fr in props:
        luse_area[luse] = luse_area.get(luse, 0) + hru_fr
        soil_area[soil] = soil_area.get(soil, 0) + hru_fr
        grp = soil_groups.get(soil, 'Unknown')
        soilgrp_area[grp] = soilgrp_area.get(grp, 0) + hru_fr
    
    sub_dom_luse[sub_id] = max(luse_area, key=luse_area.get)
    sub_dom_soil[sub_id] = max(soil_area, key=soil_area.get)
    sub_dom_soilgrp[sub_id] = max(soilgrp_area, key=soilgrp_area.get)

print(f"   Subbasins with dominant properties: {len(sub_dom_luse)}")

# ============================================================
# 6. Build land use and soil grids
# ============================================================
print("6. Building land use and soil grids...")

# Encode land use as integers for numpy
luse_types = sorted(set(sub_dom_luse.values()))
luse_to_int = {l: i+1 for i, l in enumerate(luse_types)}
int_to_luse = {v: k for k, v in luse_to_int.items()}

soil_types = sorted(set(sub_dom_soil.values()))
soil_to_int = {s: i+1 for i, s in enumerate(soil_types)}
int_to_soil = {v: k for k, v in soil_to_int.items()}

soilgrp_types = sorted(set(sub_dom_soilgrp.values()))
soilgrp_to_int = {s: i+1 for i, s in enumerate(soilgrp_types)}
int_to_soilgrp = {v: k for k, v in soilgrp_to_int.items()}

luse_int_grid = np.zeros((NROW, NCOL), dtype=int)
soil_int_grid = np.zeros((NROW, NCOL), dtype=int)
soilgrp_int_grid = np.zeros((NROW, NCOL), dtype=int)

for r in range(NROW):
    for c in range(NCOL):
        sub = sub_grid[r, c]
        if sub > 0 and sub in sub_dom_luse:
            luse_int_grid[r, c] = luse_to_int[sub_dom_luse[sub]]
            soil_int_grid[r, c] = soil_to_int[sub_dom_soil[sub]]
            soilgrp_int_grid[r, c] = soilgrp_to_int[sub_dom_soilgrp[sub]]

print(f"   Land use types in grid: {luse_types}")
print(f"   Soil types in grid: {soil_types}")
print(f"   Soil groups in grid: {soilgrp_types}")

# Distribution
from collections import Counter
luse_counts = Counter(luse_int_grid[luse_int_grid > 0])
total = sum(luse_counts.values())
print("\n   Land use distribution:")
for code, count in sorted(luse_counts.items(), key=lambda x: -x[1]):
    print(f"     {int_to_luse[code]:6s}: {count:5d} cells ({100*count/total:.1f}%)")

soilgrp_counts = Counter(soilgrp_int_grid[soilgrp_int_grid > 0])
print("\n   Soil group distribution:")
for code, count in sorted(soilgrp_counts.items(), key=lambda x: -x[1]):
    print(f"     {int_to_soilgrp[code]:6s}: {count:5d} cells ({100*count/total:.1f}%)")

# ============================================================
# 7. Save arrays
# ============================================================
np.save(r'D:\GMRW\mapping\luse_int_grid.npy', luse_int_grid)
np.save(r'D:\GMRW\mapping\soil_int_grid.npy', soil_int_grid)
np.save(r'D:\GMRW\mapping\soilgrp_int_grid.npy', soilgrp_int_grid)
np.save(r'D:\GMRW\mapping\sub_grid.npy', sub_grid)
np.save(r'D:\GMRW\mapping\ibound.npy', ibound)

# Save lookup dictionaries
import json
lookup = {
    'luse_to_int': luse_to_int,
    'int_to_luse': {str(k): v for k, v in int_to_luse.items()},
    'soil_to_int': soil_to_int,
    'int_to_soil': {str(k): v for k, v in int_to_soil.items()},
    'soilgrp_to_int': soilgrp_to_int,
    'int_to_soilgrp': {str(k): v for k, v in int_to_soilgrp.items()},
    'soil_groups': soil_groups,
}
with open(r'D:\GMRW\mapping\grid_lookup.json', 'w') as f:
    json.dump(lookup, f, indent=2)

print("\nSaved all grid arrays and lookup tables.")

# Also read TOP elevation for DEM panel
print("\n7. Parsing TOP elevation...")
top = np.zeros((NROW, NCOL), dtype=float)
with open(r'D:\GMRW\swat-modflow files\modflow_GMRW.dis') as f:
    lines = f.readlines()

# Find TOP section (after DELR and DELC)
# Format: NLAY NROW NCOL NPER ITMUNI LENUNI
# Then LAYCBD, then DELR, DELC, then TOP
idx = 0
for i, line in enumerate(lines):
    if 'TOP' in line.upper() or (i > 4 and 'CONSTANT' in line.upper()):
        pass
    # Look for INTERNAL keyword for TOP
    if 'INTERNAL' in line.upper() and 'TOP' in line.upper():
        idx = i + 1
        break
    if 'INTERNAL' in line.upper() and idx == 0 and i > 5:
        idx = i + 1
        break

if idx == 0:
    # Try alternative: after DELR and DELC lines
    const_count = 0
    for i, line in enumerate(lines):
        if 'CONSTANT' in line.upper():
            const_count += 1
        if const_count == 2:  # After DELR and DELC
            # Next should be TOP
            idx = i + 2
            break

print(f"   TOP data starts at line {idx}")
if idx > 0:
    row = 0
    for i in range(idx, len(lines)):
        vals = lines[i].split()
        if len(vals) == NCOL:
            top[row, :] = [float(v) for v in vals]
            row += 1
            if row >= NROW:
                break
        elif len(vals) == 1 and 'INTERNAL' in lines[i].upper():
            continue
    print(f"   TOP rows read: {row}")
    print(f"   TOP range: {top[active_mask].min():.1f} - {top[active_mask].max():.1f} m")
    np.save(r'D:\GMRW\mapping\top_grid.npy', top)
    print("   Saved top_grid.npy")
