import pandas as pd
import numpy as np

run_dir = r'D:\nasrin\swatmf_run\fast550_package\stage3_run'
pst_file = run_dir + r'\swat_modflow_ies_dsi550_fast_stage3.pst'

# ---- parse PST obs section (keep ALL obs including weight=0 for validation) ----
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
for g in pst_obs['obgnme'].unique():
    sub = pst_obs[pst_obs['obgnme'] == g]
    print(f"  {g}: {len(sub)} obs, weighted>0: {(sub['weight']>0).sum()}")


def calc_metrics(obs_csv, pst_obs, label):
    ens = pd.read_csv(obs_csv, index_col=0)
    ens = ens[ens.index.astype(str) != 'BASE']
    ens.columns = [c.lower() for c in ens.columns]

    results = {}
    # flow_cal: weight>0 (used in calibration)
    # flow_val: weight=0 but still compute metrics (validation, not assimilated)
    group_filters = {
        'flow_cal': pst_obs[pst_obs['obgnme'] == 'flow_cal'],
        'flow_val': pst_obs[pst_obs['obgnme'] == 'flow_val'],
    }

    print(f'\n=== {label} ===')
    for grp, g in group_filters.items():
        cols = [o for o in g['obsnme'] if o in ens.columns]
        if not cols:
            print(f'  {grp}: no matching obs columns')
            continue

        g = g[g['obsnme'].isin(cols)]
        ov = g.set_index('obsnme')['obsval']
        eg = ens[cols]

        # filter failed realizations
        valid = eg[(eg > -999).all(axis=1)]
        n_valid = len(valid)

        sim_med = valid.median(axis=0)
        sim_lo  = valid.quantile(0.025, axis=0)
        sim_hi  = valid.quantile(0.975, axis=0)

        oa = ov[cols].values.astype(float)
        sa = sim_med[cols].values.astype(float)
        la = sim_lo[cols].values.astype(float)
        ha = sim_hi[cols].values.astype(float)

        # NSE / R2
        ss_res = np.sum((oa - sa) ** 2)
        ss_tot = np.sum((oa - oa.mean()) ** 2)
        nse = 1 - ss_res / ss_tot
        r2  = nse  # same formula for median-based

        # RMSE
        rmse = np.sqrt(np.mean((oa - sa) ** 2))

        # KGE
        r = np.corrcoef(oa, sa)[0, 1]
        kge = 1 - np.sqrt((r - 1)**2 + (sa.std() / oa.std() - 1)**2 + (sa.mean() / oa.mean() - 1)**2)

        # PBIAS
        pbias = 100 * np.sum(oa - sa) / np.sum(oa)

        # PICP / MPIW
        picp = np.mean((oa >= la) & (oa <= ha)) * 100
        mpiw = np.mean(ha - la)

        print(f'  {grp} (n={len(cols)} obs, {n_valid} valid reals):')
        print(f'    NSE={nse:.4f}  R2={r2:.4f}  RMSE={rmse:.4f}  KGE={kge:.4f}  PBIAS={pbias:.2f}%  PICP={picp:.1f}%  MPIW={mpiw:.2f}')
        results[grp] = dict(NSE=round(nse,4), R2=round(r2,4), RMSE=round(rmse,4),
                            KGE=round(kge,4), PBIAS=round(pbias,2),
                            PICP=round(picp,1), MPIW=round(mpiw,2), n_valid=n_valid)
    return results

all_results = {}
for it, lbl in [('1', 'Iteration 1'), ('2', 'Iteration 2'), ('3', 'Iteration 3')]:
    obs_csv = run_dir + f'\\swat_modflow_ies_dsi550_fast_stage3.{it}.obs.csv'
    all_results[lbl] = calc_metrics(obs_csv, pst_obs, lbl)

# Summary table
print('\n' + '='*70)
print('SUMMARY TABLE')
print('='*70)
header = f"{'':12} {'NSE':>7} {'R2':>7} {'RMSE':>8} {'KGE':>7} {'PBIAS':>8} {'PICP':>6} {'MPIW':>8}"
print(header)
print('-'*70)
for lbl, grp_res in all_results.items():
    for grp, m in grp_res.items():
        tag = f"{lbl[-1]}_{grp[:8]}"
        print(f"  {tag:12} {m['NSE']:>7.4f} {m['R2']:>7.4f} {m['RMSE']:>8.4f} {m['KGE']:>7.4f} {m['PBIAS']:>7.2f}% {m['PICP']:>5.1f}% {m['MPIW']:>8.2f}")

# Phi trend
phi = pd.read_csv(run_dir + r'\swat_modflow_ies_dsi550_fast_stage3.phi.actual.csv')
print('\nPhi actual trend:')
print(phi[['iteration', 'total_runs', 'mean', 'standard_deviation', 'min', 'max']].to_string(index=False))
