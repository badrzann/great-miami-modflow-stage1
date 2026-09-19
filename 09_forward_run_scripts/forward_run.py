#!/usr/bin/env python3
"""
forward_run.py  —  SWAT–MODFLOW forward model runner for PESTPP-IES
------------------------------------------------------------------------
1. Read current parameter values from pest_pars.dat
2. Restore SWAT input files from _baseline_bak/
3. Apply parameter modifications in-place
4. Run SWAT-MODFLOW3.exe  (live stdout, with timer + output.rch size check)
5. Extract simulated FLOW_OUTcms for Reach 5, calibration period
6. Write to sim_flow.dat  (one value per line, 96 monthly values)
------------------------------------------------------------------------
Simulation period : 2001-2010 (NBYR=10)
Warm-up           : 2001-2002 (NYSKIP=2, suppressed from output)
Calibration window: 2003-01 → 2010-12  (96 months, all output years)
------------------------------------------------------------------------
"""
import os, sys, re, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
EXE        = Path("SWAT-MODFLOW3.exe")
PAR_FILE   = Path("pest_pars.dat")
SIM_FILE   = Path("sim_flow.dat")
RCH_NO     = 5
SIM_START  = "2003-01"    # First monthly record in output.rch (NYSKIP=2 suppresses 2001-2002)
CAL_START  = "2003-01"
CAL_END    = "2010-12"    # 8-year calibration window (NBYR=10, NYSKIP=2)
N_OBS      = 96           # 96 months: 2003-01 → 2010-12

# ── 1. Read parameters ─────────────────────────────────────────────────────────
def read_params(path):
    p = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            p[parts[0].lower()] = float(parts[1])
    return p

# ── 2. SWAT file value setter ──────────────────────────────────────────────────
def set_param(text, keyword, new_val):
    pattern = (r'([ \t]*)([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)'
               r'([ \t]+\|[ \t]*' + re.escape(keyword) + r'[ \t]*:)')
    def repl(m):
        orig = m.group(2)
        dec = max(len(orig.rstrip('0').split('.')[-1]), 4) if '.' in orig else 4
        formatted = f"{new_val:.{dec}f}".rjust(len(orig))
        return m.group(1) + formatted + m.group(3)
    return re.subn(pattern, repl, text, flags=re.IGNORECASE)

def multiply_param(text, keyword, mult):
    pattern = (r'([ \t]*)([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)'
               r'([ \t]+\|[ \t]*' + re.escape(keyword) + r'[ \t]*:)')
    def repl(m):
        val = float(m.group(2)) * mult
        if keyword.upper() == "CN2":
            val = max(35.0, min(98.0, val))
        orig = m.group(2)
        dec = max(len(orig.rstrip('0').split('.')[-1]), 2) if '.' in orig else 4
        formatted = f"{val:.{dec}f}".rjust(len(orig))
        return m.group(1) + formatted + m.group(3)
    return re.subn(pattern, repl, text, flags=re.IGNORECASE)

# ── 3. Apply parameters to model files ────────────────────────────────────────
def apply_params(params, run_dir=Path(".")):
    run_dir = Path(run_dir)
    bak_dir = run_dir / "_baseline_bak"
    for ext in [".mgt", ".gw", ".hru", ".rte"]:
        for bak in bak_dir.glob(f"*{ext}"):
            shutil.copy2(bak, run_dir / bak.name)
    shutil.copy2(bak_dir / "basins.bsn", run_dir / "basins.bsn")

    gw_params = {"ALPHA_BF": params.get("alpha_bf"), "GW_DELAY": params.get("gw_delay"),
                 "GWQMN": params.get("gwqmn"), "GW_REVAP": params.get("gw_revap"),
                 "RCHRG_DP": params.get("rchrg_dp"), "REVAPMN": params.get("revapmn")}
    for fp in run_dir.glob("*.gw"):
        text = fp.read_text(encoding="latin-1")
        changed = False
        for kw, val in gw_params.items():
            if val is not None:
                text, n = set_param(text, kw, val)
                if n: changed = True
        if changed:
            fp.write_text(text, encoding="latin-1")

    esco = params.get("esco")
    if esco is not None:
        for fp in run_dir.glob("*.hru"):
            text = fp.read_text(encoding="latin-1")
            text, n = set_param(text, "ESCO", esco)
            if n: fp.write_text(text, encoding="latin-1")

    cn2_m = params.get("cn2_m")
    if cn2_m is not None:
        for fp in run_dir.glob("*.mgt"):
            text = fp.read_text(encoding="latin-1")
            text, n = multiply_param(text, "CN2", cn2_m)
            if n: fp.write_text(text, encoding="latin-1")

    ch_k2_m = params.get("ch_k2_m")
    if ch_k2_m is not None:
        for fp in run_dir.glob("*.rte"):
            text = fp.read_text(encoding="latin-1")
            text, n = multiply_param(text, "CH_K2", ch_k2_m)
            if n: fp.write_text(text, encoding="latin-1")

    surlag = params.get("surlag")
    if surlag is not None:
        bsn_path = run_dir / "basins.bsn"
        text = bsn_path.read_text(encoding="latin-1")
        text, _ = set_param(text, "SURLAG", surlag)
        bsn_path.write_text(text, encoding="latin-1")

# ── 4. Read output.rch and extract calibration-period flow ────────────────────
def read_sim_flow(run_dir=Path(".")):
    df = pd.read_csv(
        Path(run_dir) / "output.rch",
        skiprows=9, sep=r"\s+", header=None,
        usecols=[0, 1, 3, 6],
        names=["LABEL", "RCH", "MON", "FLOW_OUTcms"],
        encoding="latin-1", on_bad_lines="skip",
    )
    df["RCH"]         = pd.to_numeric(df["RCH"],         errors="coerce")
    df["MON"]         = pd.to_numeric(df["MON"],         errors="coerce")
    df["FLOW_OUTcms"] = pd.to_numeric(df["FLOW_OUTcms"], errors="coerce")
    rch5  = df[(df["RCH"] == RCH_NO) & (df["MON"].between(1, 12))].reset_index(drop=True)
    dates = pd.date_range(SIM_START, periods=len(rch5), freq="MS")
    rch5.index = dates
    return rch5["FLOW_OUTcms"].rename("sim_cms").loc[CAL_START:CAL_END]

# ── 5. Pre-run validation — required SWAT input files ─────────────────────────
#   These files are read by SWAT-MODFLOW3.exe at startup.  If any are missing
#   or 0-bytes the Fortran runtime will crash with:
#       forrtl: severe (24): end-of-file during read
#   This list was determined empirically from the crash on unit 171 (septwq.dat)
#   and by checking which .dat files SWAT opens.
REQUIRED_SWAT_FILES = [
    "file.cio",       # master control
    "basins.bsn",     # basin parameters
    "plant.dat",      # crop / plant database
    "till.dat",       # tillage database
    "urban.dat",      # urban land-use database
    "septwq.dat",     # septic system database
    "pest.dat",       # pesticide database (SWAT built-in, not PEST++)
    "fert.dat",       # fertilizer database
]

def validate_required_files(run_dir: Path) -> None:
    """Abort early if any required SWAT input file is missing or empty."""
    missing = []
    empty   = []
    for fname in REQUIRED_SWAT_FILES:
        fp = run_dir / fname
        if not fp.exists():
            missing.append(fname)
        elif fp.stat().st_size == 0:
            empty.append(fname)
    if missing or empty:
        msg_parts = []
        if missing:
            msg_parts.append(f"  MISSING: {', '.join(missing)}")
        if empty:
            msg_parts.append(f"  EMPTY (0 bytes): {', '.join(empty)}")
        print("\n" + "="*60, flush=True)
        print("FATAL: Required SWAT input files are missing or empty!", flush=True)
        for m in msg_parts:
            print(m, flush=True)
        print("\nFix: copy these files from a working model directory:", flush=True)
        print(f"  e.g.  copy 'D:\\GMRW\\swatmf_run\\<file>' → '{run_dir}\\'", flush=True)
        print("="*60 + "\n", flush=True)
        sys.exit(24)  # match the Fortran error code for traceability
    else:
        print(f"  ✓ Pre-run check: {len(REQUIRED_SWAT_FILES)} required files OK", flush=True)

# ── 6. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_dir = Path(".")

    # ── Validate required input files BEFORE anything else ────────────────
    validate_required_files(run_dir)

    params  = read_params(run_dir / PAR_FILE)
    print(f"  Parameters: { {k: f'{v:.4f}' for k,v in params.items()} }", flush=True)

    apply_params(params, run_dir)
    print("  Parameters applied to model files.", flush=True)

    rch_path        = run_dir / "output.rch"
    rch_size_before = rch_path.stat().st_size if rch_path.exists() else -1
    t_start         = time.time()
    print(f"  Running SWAT-MODFLOW3.exe  [{time.strftime('%H:%M:%S')}]", flush=True)

    proc = subprocess.Popen(
        [str(run_dir / EXE)], cwd=str(run_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    last_line = ""
    try:
        for line in proc.stdout:
            stripped = line.rstrip()
            if stripped:
                print(f"  [SWAT] {stripped}", flush=True)
                last_line = stripped
    except Exception as e:
        print(f"  [WARN] Stream interrupted: {e}", flush=True)

    proc.wait()
    elapsed = time.time() - t_start
    print(f"  Finished in {elapsed:.1f}s  (exit={proc.returncode})  [{time.strftime('%H:%M:%S')}]",
          flush=True)

    if rch_path.exists():
        delta = rch_path.stat().st_size - rch_size_before
        if delta == 0:
            print("  [WARN] output.rch UNCHANGED — model may not have run!", flush=True)
        else:
            print(f"  output.rch delta: +{delta:,} bytes", flush=True)

    if proc.returncode != 0:
        print(f"  ERROR: exit code {proc.returncode}. Last line: {last_line}", flush=True)
        Path(SIM_FILE).write_text("\n".join(["-9999.0"] * N_OBS) + "\n")
        sys.exit(0)

    try:
        sim_cal = read_sim_flow(run_dir)
        n = len(sim_cal)
        if n < N_OBS:
            extra   = pd.Series([-9999.0] * (N_OBS - n))
            sim_cal = pd.concat([sim_cal, extra])
        elif n > N_OBS:
            sim_cal = sim_cal.iloc[:N_OBS]
        values = sim_cal.values
    except Exception as e:
        print(f"  ERROR reading output.rch: {e}", flush=True)
        values = np.full(N_OBS, -9999.0)

    with open(run_dir / SIM_FILE, "w") as f:
        for v in values:
            f.write(f"{v:.6f}\n")
    print(f"  Wrote {len(values)} values to {SIM_FILE}  (CAL: {CAL_START} → {CAL_END})",
          flush=True)
