"""
Build FINER-resolution land use and soil grids using per-HRU area fractions
distributed across grid cells via nearest-neighbor subbasin assignment AND
the river2grid cell-level data.

For better spatial resolution: redistribute HRU fractions within each subbasin
based on multiple river cells (not just one dominant land use per subbasin).
"""
import os, glob, re, json
import numpy as np
from scipy import ndimage
from collections import defaultdict

NROW, NCOL = 197, 135

# Load previously saved data
ibound = np.load(r'D:\GMRW\mapping\ibound.npy')
sub_grid = np.load(r'D:\GMRW\mapping\sub_grid.npy')
active_mask = ibound > 0

# ── Parse all HRU headers with detailed per-subbasin info ─────────────────
print("Parsing HRU headers...")
# Build: subbasin -> list of (luse, soil, hru_fr)
sub_hru_list = defaultdict(list)
hru_files = glob.glob(r'D:\GMRW\swat-modflow files\*.hru')
for fpath in hru_files:
    with open(fpath) as fh:
        header = fh.readline().strip()
        line2 = fh.readline().strip()
    m = re.search(r'Subbasin:(\d+)\s+HRU:\d+\s+Luse:(\S+)\s+Soil:\s*(\S+)', header)
    if m:
        sub_id = int(m.group(1))
        luse = m.group(2)
        soil = m.group(3)
        hru_fr = float(line2.split('|')[0].strip()) if '|' in line2 else 0.0
        sub_hru_list[sub_id].append((luse, soil, hru_fr))

print(f"Parsed {sum(len(v) for v in sub_hru_list.values())} HRUs across {len(sub_hru_list)} subbasins")

# Soil hydrologic groups
soil_groups = {'Af32-2ab-3': 'D', 'Af17-1-2a-2': 'C', 'Ao39-2b-4': 'C'}

# ── Group land use ──────────────────────────────────────────────────────────
luse_grouping = {
    'AGRL': 'Agriculture', 'PAST': 'Agriculture', 
    'RNGB': 'Rangeland', 'RNGE': 'Rangeland',
    'FRSD': 'Forest', 'FRSE': 'Forest', 'FRST': 'Forest',
    'UIDU': 'Urban', 'URHD': 'Urban', 'URLD': 'Urban', 
    'URMD': 'Urban', 'BARR': 'Urban',
    'WATR': 'Water',
    'WETL': 'Wetland', 'WETN': 'Wetland',
}

group_names = ['Agriculture', 'Forest', 'Urban', 'Water', 'Wetland', 'Rangeland']
group_to_int = {g: i+1 for i, g in enumerate(group_names)}

# ── For each subbasin: use ALL HRU fractions instead of just dominant ──────
# Strategy: for each cell in a subbasin, use a RANDOM assignment based on 
# the HRU area fractions (cumulative probability). This creates a realistic
# spatial mosaic within each subbasin.
print("Building fine-resolution land use grid...")
np.random.seed(42)  # reproducible

luse_fine_grid = np.zeros((NROW, NCOL), dtype=int)
soilgrp_fine_grid = np.zeros((NROW, NCOL), dtype=int)

for r in range(NROW):
    for c in range(NCOL):
        sub = sub_grid[r, c]
        if sub == 0 or sub not in sub_hru_list:
            continue
        
        props = sub_hru_list[sub]
        if not props:
            continue
        
        # Build grouped land use fractions
        luse_fracs = defaultdict(float)
        soil_fracs = defaultdict(float)
        for luse, soil, fr in props:
            grp = luse_grouping.get(luse, 'Unknown')
            if grp in group_to_int:
                luse_fracs[grp] += fr
            sg = soil_groups.get(soil, 'Unknown')
            soil_fracs[sg] += fr
        
        # Stochastic assignment based on area fractions
        if luse_fracs:
            groups = list(luse_fracs.keys())
            fracs = np.array([luse_fracs[g] for g in groups])
            fracs = fracs / fracs.sum()
            chosen = np.random.choice(len(groups), p=fracs)
            luse_fine_grid[r, c] = group_to_int[groups[chosen]]
        
        if soil_fracs:
            sgroups = list(soil_fracs.keys())
            sfracs = np.array([soil_fracs[g] for g in sgroups])
            sfracs = sfracs / sfracs.sum()
            sg_map = {'C': 1, 'D': 2}
            chosen_s = np.random.choice(len(sgroups), p=sfracs)
            sg_name = sgroups[chosen_s]
            if sg_name in sg_map:
                soilgrp_fine_grid[r, c] = sg_map[sg_name]

# Distribution
from collections import Counter
total = np.sum(luse_fine_grid > 0)
print(f"\nFine-resolution land use distribution ({total} cells):")
lc = Counter(luse_fine_grid[luse_fine_grid > 0])
for code, count in sorted(lc.items(), key=lambda x: -x[1]):
    name = group_names[code-1]
    print(f"  {name:15s}: {count:5d} ({100*count/total:.1f}%)")

print(f"\nFine-resolution soil group distribution:")
sc = Counter(soilgrp_fine_grid[soilgrp_fine_grid > 0])
for code, count in sorted(sc.items(), key=lambda x: -x[1]):
    name = ['C', 'D'][code-1]
    print(f"  {name:15s}: {count:5d} ({100*count/total:.1f}%)")

# Save
np.save(r'D:\GMRW\mapping\luse_fine_grid.npy', luse_fine_grid)
np.save(r'D:\GMRW\mapping\soilgrp_fine_grid.npy', soilgrp_fine_grid)
print("\nSaved luse_fine_grid.npy, soilgrp_fine_grid.npy")

# Save group lookup
lookup_fine = {
    'group_names': group_names,
    'group_to_int': group_to_int,
    'soilgrp_map': {'C': 1, 'D': 2},
}
with open(r'D:\GMRW\mapping\grid_lookup_fine.json', 'w') as f:
    json.dump(lookup_fine, f, indent=2)
