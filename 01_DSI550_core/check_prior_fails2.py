import pandas as pd
import numpy as np

s1 = pd.read_csv('stage1_run/swat_modflow_ies_dsi550_fast_stage1.0.obs.csv', index_col=0)
s3 = pd.read_csv('stage3_run/swat_modflow_ies_dsi550_fast_stage3.1.obs.csv', index_col=0)

flow_cal_cols = [c for c in s1.columns if 'sim' in c.lower() and any(str(y) in c for y in range(2003,2018))]
flow_val_cols = [c for c in s1.columns if 'sim' in c.lower() and any(str(y) in c for y in range(2018,2030))]

print('=== PRIOR stage1.0.obs ===')
print(f'Total rows: {len(s1)}')

# -9999 check (SWAT failure flag)
neg9999_cal = (s1[flow_cal_cols] == -9999).any(axis=1)
neg9999_val = (s1[flow_val_cols] == -9999).any(axis=1)
neg9999_any = (s1[flow_cal_cols + flow_val_cols] <= -999).any(axis=1)
print(f'Rows with -9999 in flow_cal: {neg9999_cal.sum()}')
print(f'Rows with -9999 in flow_val: {neg9999_val.sum()}')
print(f'Rows with any value <= -999 in all flow: {neg9999_any.sum()}')
print(f'  Failed realization names: {list(s1[neg9999_any].index)}')

valid = ~neg9999_any
print(f'Valid prior realizations: {valid.sum()}')
print(f'Failed prior realizations: {(~valid).sum()}')

print()
flow_cal_cols3 = [c for c in s3.columns if 'sim' in c.lower() and any(str(y) in c for y in range(2003,2018))]
flow_val_cols3 = [c for c in s3.columns if 'sim' in c.lower() and any(str(y) in c for y in range(2018,2030))]
neg9999_any3 = (s3[flow_cal_cols3 + flow_val_cols3] <= -999).any(axis=1)
print('=== POSTERIOR stage3.1.obs ===')
print(f'Total rows: {len(s3)}')
print(f'Failed (<=−999): {neg9999_any3.sum()}')
print(f'Valid: {(~neg9999_any3).sum()}')
