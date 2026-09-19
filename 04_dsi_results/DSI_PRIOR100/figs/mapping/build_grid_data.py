"""Build grid-level land use and soil arrays using SWAT-MODFLOW linkage."""
import os, glob, re
import numpy as np

NROW, NCOL = 197, 135

# -- 1. Parse IBOUND --
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
active_cells = np.sum(ibound > 0)
print(f"   Active cells: {active_cells}")

# -- 2. Parse all HRU headers --
print("2. Parsing HRU headers...")
hru_data = {}  # global_hru_id -> (luse, soil)
hru_files = glob.glob(r'D:\GMRW\swat-modflow files\*.hru')
for fpath in hru_files:
    with open(fpath) as fh:
        hdr = fh.readline()
    m = re.search(r'HRU:(\d+)\s+Subbasin:(\d+)\s+HRU:(\d+)\s+Luse:(\S+)\s+Soil:\s*(\S+)', hdr)
    if m:
        hru_data[int(m.group(1))] = (m.group(4), m.group(5))
print(f"   Parsed {len(hru_data)} HRUs")

# -- 3. Parse dhru2hru --
print("3. Parsing dhru2hru...")
dhru2hru = {}  # dhru_id -> [(hru_id, fraction), ...]
with open(r'D:\GMRW\swat-modflow files\swatmf_dhru2hru.txt') as f:
    hdr = f.readline().split()
    n_dhrus = int(hdr[0])
    for _ in range(n_dhrus):
        parts = f.readline().split()
        dhru_id = int(parts[0])
        n_hrus = int(parts[1])
        hru_ids = [int(x) for x in f.readline().split()]
        fracs = [float(x) for x in f.readline().split()]
        dhru2hru[dhru_id] = list(zip(hru_ids, fracs))
print(f"   Parsed {len(dhru2hru)} DHRUs")

# -- 4. Parse dhru2grid: grid_cell -> [(dhru_id, fraction), ...] --
print("4. Parsing dhru2grid...")
grid_dhrus = {}  # cell_linear_index -> [(dhru_id, fraction), ...]
with open(r'D:\GMRW\swat-modflow files\swatmf_dhru2grid.txt') as f:
    hdr = f.readline().split()
    n_entries = int(hdr[0])
    for _ in range(n_entries):
        parts = f.readline().split()
        cell_id = int(parts[0])  # 1-indexed linear cell index
        n = int(parts[1])
        if n > 0:
            dhru_ids = [int(x) for x in f.readline().split()]
            fracs = [float(x) for x in f.readline().split()]
            grid_dhrus[cell_id] = list(zip(dhru_ids, fracs))
print(f"   Active cells with DHRU data: {len(grid_dhrus)}")

# -- 5. For each active cell, determine dominant land use and soil --
print("5. Computing land use and soil per cell...")
luse_grid = np.full((NROW, NCOL), '', dtype='U10')
soil_grid = np.full((NROW, NCOL), '', dtype='U20')

mapped = 0
unmapped = 0
unmapped_dhru_only = 0

for cell_id, dhru_list in grid_dhrus.items():
    row = (cell_id - 1) // NCOL
    col = (cell_id - 1) % NCOL
    
    if row >= NROW or col >= NCOL:
        continue
    if ibound[row, col] == 0:
        continue
    
    # Accumulate land use and soil weights
    luse_weights = {}
    soil_weights = {}
    
    for dhru_id, cell_frac in dhru_list:
        if dhru_id not in dhru2hru:
            continue
        # Get HRUs for this DHRU
        for hru_id, hru_frac in dhru2hru[dhru_id]:
            if hru_id not in hru_data:
                continue
            luse, soil = hru_data[hru_id]
            weight = cell_frac * hru_frac
            luse_weights[luse] = luse_weights.get(luse, 0) + weight
            soil_weights[soil] = soil_weights.get(soil, 0) + weight
    
    if luse_weights:
        dom_luse = max(luse_weights, key=luse_weights.get)
        dom_soil = max(soil_weights, key=soil_weights.get)
        luse_grid[row, col] = dom_luse
        soil_grid[row, col] = dom_soil
        mapped += 1
    else:
        unmapped_dhru_only += 1

# Count unmapped active cells
for r in range(NROW):
    for c in range(NCOL):
        if ibound[r, c] > 0 and luse_grid[r, c] == '':
            unmapped += 1

print(f"   Mapped: {mapped}, Unmapped (no DHRU chain): {unmapped_dhru_only}, Total unmapped active: {unmapped}")

# -- 6. Summary statistics --
unique_luse = set(luse_grid[luse_grid != ''])
unique_soil = set(soil_grid[soil_grid != ''])
print(f"\n   Land use types in grid: {sorted(unique_luse)}")
print(f"   Soil types in grid: {sorted(unique_soil)}")

# Count per land use
from collections import Counter
luse_counts = Counter(luse_grid[luse_grid != ''].flatten())
print("\n   Land use distribution:")
for luse, count in sorted(luse_counts.items(), key=lambda x: -x[1]):
    print(f"     {luse}: {count} cells ({100*count/mapped:.1f}%)")

soil_counts = Counter(soil_grid[soil_grid != ''].flatten())
print("\n   Soil distribution:")
for soil, count in sorted(soil_counts.items(), key=lambda x: -x[1]):
    print(f"     {soil}: {count} cells ({100*count/mapped:.1f}%)")

# Save arrays for use by the figure script
np.save(r'D:\GMRW\mapping\luse_grid.npy', luse_grid)
np.save(r'D:\GMRW\mapping\soil_grid.npy', soil_grid)
np.save(r'D:\GMRW\mapping\ibound.npy', ibound)
print("\nSaved luse_grid.npy, soil_grid.npy, ibound.npy")
