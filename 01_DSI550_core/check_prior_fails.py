import pandas as pd
import numpy as np

s1 = pd.read_csv('stage1_run/swat_modflow_ies_dsi550_fast_stage1.0.obs.csv', index_col=0)

flow_cols = [c for c in s1.columns if 'sim' in c.lower()]
flow_cal_cols = [c for c in flow_cols if any(str(y) in c for y in range(2003,2018))]
flow_val_cols = [c for c in flow_cols if any(str(y) in c for y in range(2018,2030))]

print(f'=== PRIOR stage1.0.obs ===')
print(f'Total realizations: {len(s1)}')
print(f'Total observation columns: {len(s1.columns)}')
print(f'Flow_cal columns: {len(flow_cal_cols)}')
print(f'Flow_val columns: {len(flow_val_cols)}')

nan_rows = s1.isnull().any(axis=1)
print(f'NaN rows (any NaN in any column): {nan_rows.sum()}')

all_zero_cal = (s1[flow_cal_cols] == 0).all(axis=1)
all_zero_val = (s1[flow_val_cols] == 0).all(axis=1)
print(f'All-zero flow_cal rows (failed cal): {all_zero_cal.sum()}')
print(f'All-zero flow_val rows (failed val): {all_zero_val.sum()}')

# Rows with ANY zero in flow_cal
any_zero_cal = (s1[flow_cal_cols] == 0).any(axis=1)
print(f'Any-zero flow_cal rows: {any_zero_cal.sum()}')

if all_zero_cal.any():
    print(f'Failed realization names: {list(s1[all_zero_cal].index)}')
else:
    print('No failed realizations in prior (0 all-zero rows)!')
    print(f'Flow_cal min per realization (min): {s1[flow_cal_cols].min(axis=1).min():.4f}')
    print(f'Flow_cal max per realization (max): {s1[flow_cal_cols].max(axis=1).max():.4f}')

# Also check stage3.1.obs for comparison
s3 = pd.read_csv('stage3_run/swat_modflow_ies_dsi550_fast_stage3.1.obs.csv', index_col=0)
flow_cal_cols3 = [c for c in s3.columns if 'sim' in c.lower() and any(str(y) in c for y in range(2003,2018))]
all_zero_cal3 = (s3[flow_cal_cols3] == 0).all(axis=1)
print(f'\n=== POSTERIOR stage3.1.obs ===')
print(f'Total realizations: {len(s3)}')
print(f'All-zero flow_cal rows: {all_zero_cal3.sum()}')
