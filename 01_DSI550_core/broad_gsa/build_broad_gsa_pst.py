"""
build_broad_gsa_pst.py
======================
Step 1 of the broad prior-sensitivity screening workflow.

This script:
  1. Defines 34 candidate parameters with literature-based prior ranges
     (13 retained from final calibration + 21 broader candidates)
  2. Generates a 500-member Latin Hypercube Sampling (LHS) prior ensemble
  3. Writes the PEST++ IES control file (swat_modflow_broad_gsa.pst)
  4. Writes the initial prior ensemble CSV (broad_gsa.0.par.csv)
  5. Writes a launch script

Parameter groups span: surface hydrology, ET, soil, groundwater, channel,
snowmelt, HRU physical, and MODFLOW aquifer properties — covering the full
range of commonly cited SWAT-MODFLOW calibration parameters (Arnold et al.,
2012; Abbaspour et al., 2015; Bailey et al., 2016; Pokhrel et al., 2018).

Run from: D:/nasrin/swatmf_run/fast550_package/broad_gsa/
  > python build_broad_gsa_pst.py

After building, launch with:
  > pestpp-ies swat_modflow_broad_gsa.pst /h :4022
  (same worker setup as stage3)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import math, textwrap

RNG_SEED   = 42
N_REAL     = 500
OUT_DIR    = Path(__file__).parent

# ─── 1. Parameter catalogue ──────────────────────────────────────────────────
# Columns: name, group, parchg (log/none), parval1 (base), lower, upper,
#          retained_in_final_13, description
# Sources: SWAT 2012 I/O docs, Arnold et al. (2012), Abbaspour et al. (2015),
#          Bailey et al. (2016 SWAT-MODFLOW), Moriasi et al. (2007)
PARAMS = [
    # ── SWAT surface hydrology (5) ───────────────────────────────────────────
    dict(name="cn2_m",       group="swat_surface", parchg="log",  base=0.933,   lo=0.50,  hi=1.70,  retained=True,
         desc="CN2 multiplier (surface runoff curve number, .mgt)"),
    dict(name="canmx_m",     group="swat_surface", parchg="log",  base=1.0,     lo=0.20,  hi=5.00,  retained=False,
         desc="CANMX multiplier (max canopy storage, .hru)"),
    dict(name="slsubbsn_m",  group="swat_surface", parchg="log",  base=1.0,     lo=0.50,  hi=2.00,  retained=False,
         desc="SLSUBBSN multiplier (avg slope length, .hru)"),
    dict(name="ov_n_m",      group="swat_surface", parchg="log",  base=1.0,     lo=0.50,  hi=3.00,  retained=False,
         desc="OV_N multiplier (Manning's n overland flow, .hru)"),
    dict(name="surlag",      group="swat_surface", parchg="log",  base=7.995,   lo=0.50,  hi=48.0,  retained=True,
         desc="SURLAG - surface runoff lag coefficient, basins.bsn [days]"),

    # ── SWAT evapotranspiration (2) ──────────────────────────────────────────
    dict(name="esco",        group="swat_et",      parchg="none", base=0.732,   lo=0.01,  hi=1.00,  retained=True,
         desc="ESCO - soil evaporation compensation factor, .hru"),
    dict(name="epco",        group="swat_et",      parchg="none", base=1.0,     lo=0.01,  hi=1.00,  retained=False,
         desc="EPCO - plant uptake compensation factor, .hru"),

    # ── SWAT soil hydraulics (4) ─────────────────────────────────────────────
    dict(name="sol_awc_m",   group="swat_soil",    parchg="log",  base=1.0,     lo=0.50,  hi=2.00,  retained=False,
         desc="SOL_AWC multiplier (available water capacity, .sol)"),
    dict(name="sol_k_m",     group="swat_soil",    parchg="log",  base=1.0,     lo=0.25,  hi=4.00,  retained=False,
         desc="SOL_K multiplier (saturated hydraulic conductivity, .sol)"),
    dict(name="sol_bd_m",    group="swat_soil",    parchg="log",  base=1.0,     lo=0.80,  hi=1.20,  retained=False,
         desc="SOL_BD multiplier (bulk density, .sol)"),
    dict(name="ffcb",        group="swat_soil",    parchg="none", base=0.0,     lo=0.00,  hi=1.00,  retained=False,
         desc="FFCB - initial soil water as fraction of field capacity, basins.bsn"),

    # ── SWAT shallow groundwater (9) ────────────────────────────────────────
    dict(name="alpha_bf",    group="swat_gw",      parchg="log",  base=0.0507,  lo=0.001, hi=1.00,  retained=True,
         desc="ALPHA_BF - shallow aquifer baseflow recession constant, .gw [1/day]"),
    dict(name="gw_delay",    group="swat_gw",      parchg="log",  base=14.62,   lo=0.20,  hi=180.0, retained=True,
         desc="GW_DELAY - recharge lag time, .gw [days]"),
    dict(name="gw_revap",    group="swat_gw",      parchg="log",  base=0.02,    lo=0.01,  hi=0.05,  retained=True,
         desc="GW_REVAP - groundwater revap coefficient, .gw"),
    dict(name="gwqmn",       group="swat_gw",      parchg="log",  base=712.67,  lo=1e-6,  hi=4000., retained=True,
         desc="GWQMN - threshold depth for baseflow return, .gw [mm]"),
    dict(name="rchrg_dp",    group="swat_gw",      parchg="none", base=0.0302,  lo=0.00,  hi=0.20,  retained=True,
         desc="RCHRG_DP - deep aquifer percolation fraction, .gw"),
    dict(name="revapmn",     group="swat_gw",      parchg="log",  base=749.74,  lo=200.,  hi=1500., retained=True,
         desc="REVAPMN - revap threshold depth, .gw [mm]"),
    dict(name="lat_ttime",   group="swat_gw",      parchg="none", base=0.0,     lo=0.00,  hi=10.0,  retained=False,
         desc="LAT_TTIME - lateral flow travel time, .hru [days]"),
    dict(name="gw_spyld_m",  group="swat_gw",      parchg="log",  base=1.0,     lo=0.20,  hi=5.00,  retained=False,
         desc="GW_SPYLD multiplier (shallow aquifer specific yield, .gw)"),
    dict(name="alpha_bf_d",  group="swat_gw",      parchg="log",  base=0.01,    lo=0.0005,hi=0.50,  retained=False,
         desc="ALPHA_BF_D - deep aquifer baseflow recession constant, .gw [1/day]"),
    dict(name="deepst_m",    group="swat_gw",      parchg="log",  base=1.0,     lo=0.10,  hi=5.00,  retained=False,
         desc="DEEPST multiplier (initial deep aquifer depth, .gw)"),
    dict(name="shallst_m",   group="swat_gw",      parchg="log",  base=1.0,     lo=0.10,  hi=5.00,  retained=False,
         desc="SHALLST multiplier (initial shallow aquifer depth, .gw)"),

    # ── SWAT channel routing (3) ─────────────────────────────────────────────
    dict(name="ch_k2_m",     group="swat_channel", parchg="log",  base=1.003,   lo=0.005, hi=10.0,  retained=True,
         desc="CH_K2 multiplier (channel effective hydraulic conductivity, .rte)"),
    dict(name="ch_n2_m",     group="swat_channel", parchg="log",  base=1.0,     lo=0.50,  hi=3.00,  retained=False,
         desc="CH_N2 multiplier (Manning's n main channel, .rte)"),
    dict(name="alpha_bnk",   group="swat_channel", parchg="none", base=0.0,     lo=0.00,  hi=1.00,  retained=False,
         desc="ALPHA_BNK - bank storage baseflow alpha factor, .rte [1/day]"),

    # ── SWAT snowmelt (4) ────────────────────────────────────────────────────
    dict(name="sftmp",       group="swat_snow",    parchg="none", base=1.0,     lo=-5.0,  hi=5.0,   retained=False,
         desc="SFTMP - snowfall temperature threshold, basins.bsn [degC]"),
    dict(name="smfmx",       group="swat_snow",    parchg="none", base=4.5,     lo=1.0,   hi=10.0,  retained=False,
         desc="SMFMX - max snowmelt rate (June 21), basins.bsn [mm/degC-day]"),
    dict(name="smfmn",       group="swat_snow",    parchg="none", base=4.5,     lo=0.0,   hi=8.0,   retained=False,
         desc="SMFMN - min snowmelt rate (Dec 21), basins.bsn [mm/degC-day]"),
    dict(name="timp",        group="swat_snow",    parchg="none", base=1.0,     lo=0.01,  hi=1.00,  retained=False,
         desc="TIMP - snow pack temperature lag factor, basins.bsn"),

    # ── SWAT HRU physical (2) ────────────────────────────────────────────────
    dict(name="hru_slp_m",   group="swat_hru",     parchg="log",  base=1.0,     lo=0.50,  hi=2.00,  retained=False,
         desc="HRU_SLP multiplier (average slope steepness, .hru)"),

    # ── MODFLOW aquifer (4) ──────────────────────────────────────────────────
    dict(name="hk_mult",     group="modflow",      parchg="log",  base=2.0,     lo=0.10,  hi=10.0,  retained=True,
         desc="HK multiplier (horizontal hydraulic conductivity, UPW)"),
    dict(name="sy",          group="modflow",      parchg="log",  base=0.15,    lo=0.01,  hi=0.40,  retained=True,
         desc="Sy - specific yield (UPW)"),
    dict(name="ss",          group="modflow",      parchg="log",  base=5e-5,    lo=1e-7,  hi=1e-3,  retained=True,
         desc="Ss - specific storage, UPW [1/m]"),
    dict(name="vka_mult",    group="modflow",      parchg="log",  base=1.0,     lo=0.10,  hi=10.0,  retained=False,
         desc="VKA multiplier (vertical anisotropy HK/VK, UPW)"),
]

# ─── 2. Latin Hypercube Sampling ─────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)

def lhs_sample(n, d):
    """Generate an (n x d) LHS sample in [0, 1]."""
    cut = np.linspace(0, 1, n + 1)
    u   = rng.random((n, d))
    for j in range(d):
        idx = rng.permutation(n)
        u[:, j] = cut[idx] + u[:, j] * (cut[1] - cut[0])
    return u

n_par = len(PARAMS)
lhs   = lhs_sample(N_REAL, n_par)   # (500, 22) uniform [0,1]

# Transform each column from [0,1] → parameter space
par_arr = np.zeros_like(lhs)
for j, p in enumerate(PARAMS):
    lo, hi = p["lo"], p["hi"]
    if p["parchg"] == "log":
        lo_t, hi_t = math.log10(lo), math.log10(hi)
        par_arr[:, j] = 10 ** (lo_t + lhs[:, j] * (hi_t - lo_t))
    else:
        par_arr[:, j] = lo + lhs[:, j] * (hi - lo)

real_names = [f"real_{i:04d}" for i in range(N_REAL)]
par_df = pd.DataFrame(par_arr,
                      index=real_names,
                      columns=[p["name"] for p in PARAMS])
par_df.index.name = "real_name"

# Save LHS ensemble
par_csv = OUT_DIR / "broad_gsa.0.par.csv"
par_df.to_csv(par_csv)
print(f"Saved LHS ensemble: {par_csv}  shape={par_df.shape}")

# ─── 3. Candidate parameter summary table ────────────────────────────────────
summary = pd.DataFrame([{
    "Parameter":       p["name"],
    "Group":           p["group"],
    "Description":     p["desc"],
    "Transform":       p["parchg"],
    "Lower":           p["lo"],
    "Upper":           p["hi"],
    "Base":            p["base"],
    "Retained_final13": "YES" if p["retained"] else "no",
} for p in PARAMS])
summary.to_csv(OUT_DIR / "candidate_parameter_table.csv", index=False)
print("Saved candidate parameter table")

# ─── 4. Write PEST++ IES control file ────────────────────────────────────────
# Reuse observations and instruction files from stage3
STAGE3_DIR  = OUT_DIR.parent / "stage3_run"
PARENT_RUN  = OUT_DIR.parent.parent   # swatmf_run/

# Count parameter groups
n_pargp = len(set(p["group"] for p in PARAMS))

pst_lines = []
pst_lines.append("pcf")
pst_lines.append("* control data")
# Line 1: RSTFLE PESTMODE
pst_lines.append("         norestart          estimation")
# Line 2: NPAR NOBS NPARGP NPRIOR NOBSGP
pst_lines.append(f"        {n_par}       356         {n_pargp}         0         3")
# Line 3: NTPLFLE NINSFLE PRECIS DPOINT NUMCOM JACFILE MESSFILE
pst_lines.append("         1         2              single               point         1")
# Lines 4-9: optimization control (matching stage3 format; NOPTMAX=-1 for prior ensemble eval)
pst_lines.append("   2.000000E+01  -3.000000E+00   3.000000E-01   1.000000E-02        -7")
pst_lines.append("   1.000000E+01   1.000000E+01   1.000000E-03")
pst_lines.append("   1.000000E-01")
# NOPTMAX=-1 → evaluate prior ensemble then stop (no parameter updates)
pst_lines.append("        -1   1.000000E-02         3         3   1.000000E-02         3")
pst_lines.append("         0         0         0")
pst_lines.append("* singular value decomposition")
pst_lines.append("         1")
pst_lines.append(f"        {n_par}    1.000000E-06")
pst_lines.append("1")
pst_lines.append("* parameter groups")
for grp in ["swat_surface", "swat_et", "swat_soil", "swat_gw", "swat_channel", "swat_snow", "swat_hru", "modflow"]:
    pst_lines.append(f"{grp}  relative  0.01  0.0  switch  2.0  parabolic")
pst_lines.append("* parameter data")
for p in PARAMS:
    lo, hi = p["lo"], p["hi"]
    # Match stage3 PST format: PARTRANS=log or none, PARCHG=factor.
    # SCALE=1.0, OFFSET=0.0, DERCOM=1  (same as stage3 which works).
    # OFFSET=0.0 is critical — OFFSET=1.0 would add +1 to every parameter value.
    partrans = "log" if p["parchg"] == "log" else "none"
    pst_lines.append(
        f"{p['name']:<15s}  {partrans:<6s}  factor  {p['base']:<14.6g}  "
        f"{lo:<14.6g}  {hi:<14.6g}  {p['group']}  1.0  0.0  1"
    )
pst_lines.append("* observation groups")
pst_lines.append("flow_cal")
pst_lines.append("flow_val")
pst_lines.append("head")
# Read observation data from stage3 PST
stage3_pst = STAGE3_DIR / "swat_modflow_ies_dsi550_fast_stage3.pst"
in_obs = False
obs_block = []
for ln in stage3_pst.read_text().splitlines():
    s = ln.strip()
    if s.startswith("* observation data"): in_obs = True; continue
    if in_obs and s.startswith("*"): break
    if in_obs and s: obs_block.append(ln)
pst_lines.append("* observation data")
pst_lines.extend(obs_block)
pst_lines.append("* model command line")
pst_lines.append(r"D:\nasrin\swatmf_run\.venv\Scripts\python.exe forward_run_broad.py")
pst_lines.append("* model input/output")
pst_lines.append(f"pest_pars_broad.dat.tpl  pest_pars_broad.dat")
pst_lines.append(f"sim_flow_fp.ins  sim_flow_fp.dat")
pst_lines.append(f"sim_head_fp.ins  sim_head_fp.dat")
pst_lines.append("* prior information")
pst_lines.append("* regularization")

pst_path = OUT_DIR / "swat_modflow_broad_gsa.pst"
pst_path.write_text("\n".join(pst_lines))
print(f"Saved PST file: {pst_path}")

# ─── 5. Write pestpp options section ────────────────────────────────────────
# Append PESTPP options (stage3-compatible v5.2.16 option names)
with open(pst_path, "a") as f:
    f.write(textwrap.dedent(f"""
++ies_par_en(broad_gsa.0.par.csv)
++ies_num_reals({N_REAL})
++ies_no_noise(true)
++ies_verbose_level(1)
++ies_save_binary(false)
++ies_subset_size({N_REAL})
++max_run_fail(50)
++overdue_giveup_fac(6.0)
++overdue_resched_fac(3.0)
++panther_agent_no_ping_timeout_secs(86400)
++panther_agent_restart_on_error(1)
++panther_agent_freeze_on_fail(false)
"""))
print(f"PESTPP options written to {pst_path}")

# ─── 6. Write launch PowerShell script ───────────────────────────────────────
launch = textwrap.dedent(f"""\
# Launch broad prior-sensitivity ensemble (eval-only, 500 realizations)
# Run from: D:\\nasrin\\swatmf_run\\fast550_package\\broad_gsa\\
# Step 1: Copy worker setup (if needed)
#   Each worker must have: forward_run_broad.py, pest_pars_broad.dat.tpl,
#   all SWAT model files, SWAT-MODFLOW3.exe, pestpp-ies.exe

# Start master
Start-Process pestpp-ies.exe -ArgumentList "swat_modflow_broad_gsa.pst /h :4022" -NoNewWindow

# Workers (run from each worker directory)
# Foreach ($w in 1..24) {{
#   $dir = "D:\\nasrin\\swatmf_run\\worker$w"
#   Start-Process pestpp-ies.exe -ArgumentList "swat_modflow_broad_gsa.pst /h localhost:4022" -WorkingDirectory $dir -NoNewWindow
# }}
Write-Host "Broad GSA ensemble launch script generated."
Write-Host "Estimated runtime: ~{N_REAL // 24 * 5} minutes with 24 workers"
""")
(OUT_DIR / "launch_broad_gsa.ps1").write_text(launch)
print("Saved launch script: launch_broad_gsa.ps1")

# ─── 7. Print summary ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BROAD GSA ENSEMBLE BUILD COMPLETE")
print("="*70)
print(f"  Candidate parameters : {n_par}")
print(f"    Retained (final 13): {sum(p['retained'] for p in PARAMS)}")
print(f"    New candidates     : {sum(not p['retained'] for p in PARAMS)}")
print(f"  LHS realizations     : {N_REAL}")
print(f"  Output folder        : {OUT_DIR}")
print("\nFiles created:")
print(f"  broad_gsa.0.par.csv           — LHS prior ensemble")
print(f"  candidate_parameter_table.csv  — parameter catalogue")
print(f"  swat_modflow_broad_gsa.pst     — PEST++ IES control file")
print(f"  launch_broad_gsa.ps1           — launch script")
print("\nNext steps:")
print("  1. Copy forward_run_broad.py and pest_pars_broad.dat.tpl to each worker")
print("  2. Run launch_broad_gsa.ps1")
print("  3. After completion, run analyze_broad_gsa.py")
