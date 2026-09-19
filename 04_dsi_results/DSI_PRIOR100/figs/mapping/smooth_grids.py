"""
Apply majority filter to smooth the stochastic land use and soil grids.
This removes salt-and-pepper noise while preserving subbasin-scale patterns.
"""
import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import mode as scipy_mode

def mode_filter_2d(data, size=3):
    """Apply a majority (mode) filter, ignoring zeros."""
    def local_mode(values):
        nz = values[values > 0]
        if len(nz) == 0:
            return 0
        vals, counts = np.unique(nz, return_counts=True)
        return vals[np.argmax(counts)]
    return generic_filter(data.astype(float), local_mode, size=size, mode='constant', cval=0)

# Load
luse = np.load(r'D:\GMRW\mapping\luse_fine_grid.npy')
soil = np.load(r'D:\GMRW\mapping\soilgrp_fine_grid.npy')
ibound = np.load(r'D:\GMRW\mapping\ibound.npy')
sub_grid = np.load(r'D:\GMRW\mapping\sub_grid.npy')

print("Applying 5x5 mode filter to land use grid...")
luse_smooth = mode_filter_2d(luse, size=5).astype(int)
# Ensure only active cells have values
luse_smooth[ibound <= 0] = 0
luse_smooth[sub_grid == 0] = 0

print("Applying 5x5 mode filter to soil grid...")
soil_smooth = mode_filter_2d(soil, size=5).astype(int)
soil_smooth[ibound <= 0] = 0
soil_smooth[sub_grid == 0] = 0

# Distribution check
from collections import Counter
total = np.sum(luse_smooth > 0)
group_names = ['Agriculture', 'Forest', 'Urban', 'Water', 'Wetland', 'Rangeland']
print(f"\nSmoothed land use ({total} cells):")
lc = Counter(luse_smooth[luse_smooth > 0])
for code, count in sorted(lc.items(), key=lambda x: -x[1]):
    print(f"  {group_names[code-1]:15s}: {count:5d} ({100*count/total:.1f}%)")

print(f"\nSmoothed soil groups:")
sc = Counter(soil_smooth[soil_smooth > 0])
for code, count in sorted(sc.items(), key=lambda x: -x[1]):
    name = ['C', 'D'][code-1]
    print(f"  {name:15s}: {count:5d} ({100*count/total:.1f}%)")

np.save(r'D:\GMRW\mapping\luse_smooth_grid.npy', luse_smooth)
np.save(r'D:\GMRW\mapping\soilgrp_smooth_grid.npy', soil_smooth)
print("\nSaved smoothed grids.")
