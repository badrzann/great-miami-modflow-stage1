"""Parse SWAT-MODFLOW linkage files to understand the data structure."""
import os, glob, re

# 1. Parse dhru2grid from swat-modflow files directory
print("=== PARSING dhru2grid ===")
with open(r'D:\GMRW\swat-modflow files\swatmf_dhru2grid.txt') as f:
    header = f.readline().split()
    print(f'Header: {header}')
    
    nz_entries = []
    for cell_idx in range(int(header[0])):
        line = f.readline()
        if not line:
            break
        parts = line.split()
        cid = int(parts[0])
        n = int(parts[1])
        if n > 0:
            dhru_ids = [int(x) for x in f.readline().split()]
            fractions = [float(x) for x in f.readline().split()]
            nz_entries.append((cid, n, dhru_ids, fractions))

print(f'Total non-zero (active) cells: {len(nz_entries)}')
print('First 10:')
for cid, n, dids, fracs in nz_entries[:10]:
    print(f'  Cell {cid}: n={n}, dhrus={dids}, fracs={[round(x,5) for x in fracs]}')

all_dhru_ids = set()
for _, _, dids, _ in nz_entries:
    all_dhru_ids.update(dids)
print(f'Unique DHRU IDs: {len(all_dhru_ids)}, min={min(all_dhru_ids)}, max={max(all_dhru_ids)}')

# 2. Parse dhru2hru - check DHRU ID range
print("\n=== PARSING dhru2hru ===")
with open(r'D:\GMRW\swat-modflow files\swatmf_dhru2hru.txt') as f:
    header = f.readline().split()
    print(f'Header: {header}')
    
    dhru2hru = {}
    for _ in range(int(header[0])):
        line = f.readline().split()
        dhru_id = int(line[0])
        n_hrus = int(line[1])
        third = int(line[2]) if len(line) > 2 else None
        hru_ids = [int(x) for x in f.readline().split()]
        fracs = [float(x) for x in f.readline().split()]
        # Dominant HRU
        dom_idx = fracs.index(max(fracs))
        dhru2hru[dhru_id] = (hru_ids[dom_idx], max(fracs))
    
    print(f'Parsed {len(dhru2hru)} DHRUs')
    print(f'DHRU ID range: {min(dhru2hru.keys())}-{max(dhru2hru.keys())}')

# 3. Check overlap
overlap = all_dhru_ids.intersection(dhru2hru.keys())
print(f'\nDHRU IDs in both files: {len(overlap)} out of {len(all_dhru_ids)} (dhru2grid) and {len(dhru2hru)} (dhru2hru)')

# 4. For the overlapping DHRUs, get luse/soil info
print("\n=== PARSING HRU headers ===")
hru_files = glob.glob(r'D:\GMRW\swat-modflow files\*.hru')
hru_data = {}
for fpath in hru_files:
    with open(fpath) as fh:
        header_line = fh.readline().strip()
    m = re.search(r'HRU:(\d+)\s+Subbasin:(\d+)\s+HRU:(\d+)\s+Luse:(\S+)\s+Soil:\s*(\S+)', header_line)
    if m:
        gid = int(m.group(1))
        hru_data[gid] = (m.group(4), m.group(5))

print(f'Parsed {len(hru_data)} HRU headers')

# 5. Parse .sol files for soil hydrologic group
print("\n=== PARSING .sol files for hydrologic group ===")
sol_files = glob.glob(r'D:\GMRW\swat-modflow files\*.sol')
soil_groups = {}
for fpath in sol_files[:len(sol_files)]:
    with open(fpath) as fh:
        header_line = fh.readline().strip()
        m_soil = re.search(r'Soil:\s*(\S+)', header_line)
        soil_name = m_soil.group(1) if m_soil else None
        for line in fh:
            if 'Soil Hydrologic Group' in line:
                grp = line.split(':')[1].strip()
                if soil_name:
                    soil_groups[soil_name] = grp
                break

print(f'Soil hydrologic groups: {soil_groups}')

# 6. For non-overlapping DHRUs, try alternative approach
# Use grid2dhru from finalresult to map grid cells → subbasins
print("\n=== PARSING grid2dhru (finalresult) ===")
with open(r'D:\GMRW\finalresult\swatmf_run\swatmf_grid2dhru.txt') as f:
    header = f.readline().split()
    print(f'Header: {header}')
    
    # Read first 20 blocks
    grid_cell_subs = {}
    for _ in range(50):
        line = f.readline()
        if not line:
            break
        parts = line.split()
        cell_id = int(parts[0])
        n = int(parts[1])
        if n > 0:
            row2 = [int(x) for x in f.readline().split()]  # row indices
            row3 = [int(x) for x in f.readline().split()]  # col indices
            row4 = [float(x) for x in f.readline().split()]  # fractions
            grid_cell_subs[cell_id] = list(zip(row2, row3, row4))

print(f'Parsed {len(grid_cell_subs)} grid cell entries')
print('Sample entries:')
for cell_id in list(grid_cell_subs.keys())[:5]:
    print(f'  DHRU element {cell_id}: overlaps with {len(grid_cell_subs[cell_id])} MODFLOW cells')
    for r, c, f in grid_cell_subs[cell_id][:3]:
        print(f'    row={r}, col={c}, frac={f:.5f}')
