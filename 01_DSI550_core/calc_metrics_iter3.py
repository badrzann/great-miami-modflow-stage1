import pandas as pd
import numpy as np

run_dir = r'D:\nasrin\swatmf_run\fast550_package\stage3_run'
pst_file = run_dir + r'\swat_modflow_ies_dsi550_fast_stage3.pst'

# ---- parse PST obs section ----
obs_rows = []
in_obs = False
with open(pst_file) as f:
    for line in f:
        l = line.strip()
        if l.lower().startswith('* observation data'):
            in_obs = True
            continue
        if in_obs and l.startswith('*'):
            break
        if in_obs and l:
            parts = l.split()
            if len(parts) >= 4:
                obs_rows.append({
                    'obsnme': parts[0].lower(),
                    'obsval': float(parts[1]),
                    'weight': float(parts[2]),
                    'obgnme': parts[3].lower()
                })

pst_obs = pd.DataFrame(obs_rows)
print(f"PST obs loaded: {len(pst_obs)} rows")
print("groups:", list(pst_obs['obgnme'].unique()))


def calc_metrics(obs_csv, pst_obs, label):
    ens = pd.read_csv(obs_csv, index_col=0)
    ens = ens[ens.index.astype(str) != 'BASE']
    ens.columns = [c.lower() for c in ens.columns]
    print(f'\n=== {label} ===')
    for grp in ['flow_cal', 'flow_val']:
        g = pst_obs[(pst_obs['obgnme'] == grp) & (pst_obs['weight'] > 0)]
        cols = [o for o in g['obsnme'] if o in ens.columns]
        if not cols:
            print(f'  {grp}: no matching obs columns')
            continue
        g = g[g['obsnme'].isin(cols)]
        ov = g.set_index('obsnme')['obsval']
        eg = ens[cols]
        valid = eg[(eg > -999).all(axis=1)]
        sm = valid.median(axis=0)
        sl = valid.quantile(0.025, axis=0)
        sh = valid.quantile(0.975, axis=0)
        oa = ov[cols].values
        sa = sm[cols].values
        la = sl[cols].values
        ha = sh[cols].values
        nse  = 1 - np.sum((oa - sa)**2) / np.sum((oa - oa.mean())**2)
        r    = np.corrcoef(oa, sa)[0, 1]
        kge  = 1 - np.sqrt((r-1)**2 + (sa.std()/oa.std()-1)**2 + (sa.mean()/oa.mean()-1)**2)
        pbias = 100 * np.sum(oa - sa) / np.sum(oa)
        r2   = 1 - np.sum((oa - sa)**2) / np.sum((oa - oa.mean())**2)
        picp = np.mean((oa >= la) & (oa <= ha)) * 100
        mpiw = np.mean(ha - la)
        print(f'  {grp} (n_valid={len(valid)}): NSE={nse:.4f}  KGE={kge:.4f}  PBIAS={pbias:.2f}%  R2={r2:.4f}  PICP={picp:.1f}%  MPIW={mpiw:.2f}')


for it, lbl in [('1', 'Iteration 1'), ('2', 'Iteration 2'), ('3', 'Iteration 3')]:
    obs_csv = run_dir + f'\\swat_modflow_ies_dsi550_fast_stage3.{it}.obs.csv'
    calc_metrics(obs_csv, pst_obs, lbl)

# phi actual summary
phi = pd.read_csv(run_dir + r'\swat_modflow_ies_dsi550_fast_stage3.phi.actual.csv')
print('\n=== phi.actual (actual, no noise) ===')
print(phi[['iteration', 'n', 'mean', 'std', 'min', 'max']].to_string(index=False))
