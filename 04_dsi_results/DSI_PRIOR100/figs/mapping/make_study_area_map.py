import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
NROW, NCOL = 197, 135
LON_MIN, LAT_MIN = -84.855, 39.185
DX, DY = 0.01, 0.01 # Approximate grid spacing based on previous metadata

# --- Data Loading ---
def load_ibound(filepath):
    # Simplified parsing of MODFLOW BAS file for IBOUND
    status = np.zeros((NROW, NCOL))
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Skip header and find start of IBOUND
        start_idx = 0
        for i, line in enumerate(lines):
            if "(FREE)" in line:
                start_idx = i + 1
                break
        
        row_data = []
        for line in lines[start_idx:]:
            # Skip empty lines or header/footer noise
            clean_parts = [x for x in line.split() if x.lstrip('-').isdigit()]
            if not clean_parts: continue
            row_data.extend([int(x) for x in clean_parts])
            if len(row_data) >= NROW * NCOL:
                break
        
        status = np.array(row_data[:NROW*NCOL]).reshape((NROW, NCOL))
    return status

def load_river_cells(filepath):
    river_cells = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                # MODFLOW: layer, row, col. 1-indexed.
                row, col = int(parts[1]), int(parts[2])
                river_cells.append((row-1, col-1))
    return river_cells

def load_gauges(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    # Assumes columns 'row', 'col' and 'well_name' or 'obsnme'
    return df

# paths
swatmf_dir = r"D:\GMRW\finalresult\swatmf_run"
bas_file = os.path.join(swatmf_dir, "modflow_GMRW.bas")
riv_file = os.path.join(swatmf_dir, "modflow_GMRW.riv")
well_file = os.path.join(swatmf_dir, "well_name_mapping.csv")

# Load data
ibound = load_ibound(bas_file)
river_cells = load_river_cells(riv_file)
gauges = load_gauges(well_file)

# Initialize plot array
# 0: Inactive, 1: Active, 2: River
plot_data = np.zeros_like(ibound, dtype=float)
plot_data[ibound != 0] = 1.0
for r, c in river_cells:
    if 0 <= r < NROW and 0 <= c < NCOL:
        plot_data[r, c] = 2.0

# --- Plotting ---
plt.figure(figsize=(10, 12))

# Define Colormap: [White (Inactive), LightGrey (Active), Blue (River)]
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['white', '#f0f0f0', '#0072b2'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = plt.Normalize(-0.5, 2.5)

plt.imshow(plot_data, cmap=cmap, interpolation='nearest', aspect='equal')

# Overlay Grid (Subtle)
for i in range(NCOL + 1):
    plt.axvline(i - 0.5, color='gray', linewidth=0.1, alpha=0.5)
for i in range(NROW + 1):
    plt.axhline(i - 0.5, color='gray', linewidth=0.1, alpha=0.5)

# Plot Gauges
# Group by type if possible, or just plot all as indicators
# Based on user image, red dots/squares for flow gauges
plt.scatter(gauges['col']-1, gauges['row']-1, c='red', s=50, marker='s', label='Flow Gauges', edgecolors='black')

# Add Labels for specific gauges (matching style)
# Pick a few representative ones
for i, row in gauges.head(10).iterrows():
    plt.text(row['col']-1, row['row']-1, f" {row['well_name']}", fontsize=8, verticalalignment='center')

# pilot points hypothetical (if found) - placeholder
# plt.scatter(pp_cols, pp_rows, c='green', s=20, marker='x', label='Pilot Points')

# Aesthetics
plt.title("GMRW Study Area: MODFLOW Grid, River Cells, and Gauges", fontsize=14)
plt.xlabel("Column Number")
plt.ylabel("Row Number")
plt.legend(loc='upper right')

# Invert Y to match MODFLOW (Row 1 at top)
plt.gca().invert_yaxis()

plt.tight_layout()
output_path = os.path.join(swatmf_dir, "gmrw_study_area_map.png")
plt.savefig(output_path, dpi=300)
print(f"Map saved successfully to: {output_path}")
