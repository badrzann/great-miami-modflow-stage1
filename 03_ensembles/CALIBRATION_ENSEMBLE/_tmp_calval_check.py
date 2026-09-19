import pandas as pd, numpy as np
base = 'D:/GMRW/finalresult/swatmf_run/'

print('=== IES calval iterations (full 2003-2023) ===')
for i in range(6):
    f = f'swat_modflow_ies_calval.{i}.obs.csv'
    df = pd.read_csv(base+f, index_col=0)
    nrows = len(df)
    head_cols = [c for c in df.columns if 'head' in c]
    flow18    = [c for c in df.columns if 'flow_2018' in c or 'flow_2019' in c or 'flow_2020' in c or 'flow_2021' in c or 'flow_2022' in c or 'flow_2023' in c]
    if head_cols:
        w57 = df['head_w057_sim'].replace(-9999, np.nan).dropna().values
        print(f'iter {i}: {nrows} reals | head_w057=[{w57.min():.2f}-{w57.max():.2f}] | val-flow cols={len(flow18)}')
    else:
        print(f'iter {i}: {nrows} reals | NO head cols | val-flow cols={len(flow18)}')

print()
print('=== calval.5 (likely the posterior used in paper) ===')
df = pd.read_csv(base+'swat_modflow_ies_calval.5.obs.csv', index_col=0)
print(f'Rows: {len(df)}')
w57 = df['head_w057_sim'].replace(-9999, np.nan).dropna().values
w44 = df['head_w044_sim'].replace(-9999, np.nan).dropna().values
print(f'head_w057: [{w57.min():.2f}-{w57.max():.2f}] (obs=310.32)')
print(f'head_w044: [{w44.min():.2f}-{w44.max():.2f}] (obs=329.13)')
f18 = df['flow_201804_sim'].replace(-9999, np.nan).dropna().values
print(f'flow_201804: [{f18.min():.1f}-{f18.max():.1f}] (obs=373.78)')
f03 = df['flow_200501_sim'].replace(-9999, np.nan).dropna().values
print(f'flow_200501: [{f03.min():.1f}-{f03.max():.1f}] (obs=662.9)')
