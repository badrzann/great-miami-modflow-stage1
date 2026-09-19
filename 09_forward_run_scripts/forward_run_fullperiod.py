#!/usr/bin/env python3
"""
forward_run_fullperiod.py  Ã¢â¬â  Joint SWAT+MODFLOW forward runner (full period)
------------------------------------------------------------------------------
Identical to forward_run_v5.py except:
  CAL_END      = "2023-12"   (was "2010-12")
  N_OBS_FLOW   = 252         (was 96)
  SIM_FLOW     = "sim_flow_fp.dat"  (separate output file)

This runner extracts 252 monthly flow values (2003-01 Ã¢â â 2023-12) from
output.rch. The SWAT-MODFLOW model is already configured for NBYR=23 in
file.cio so no changes to the model itself are needed.

Observations
  sim_flow_fp.dat Ã¢â¬â 252 monthly Reach-5 streamflow (2003-01 Ã¢â â 2023-12)
  sim_head_fp.dat Ã¢â¬â 104 time-averaged groundwater heads (mean of saved steps)

Parameters (13 Ã¢â¬â same as v5/v6):
  SWAT  (10) : alpha_bf, ch_k2_m, cn2_m, esco, gw_delay, gw_revap,
               gwqmn, surlag, rchrg_dp, revapmn
  MODFLOW (3): hk_mult, sy, ss
------------------------------------------------------------------------------
"""
import os, sys, re, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ã¢ââ¬Ã¢ââ¬ Configuration Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
EXE          = Path("SWAT-MODFLOW3.exe")
PAR_FILE     = Path("pest_pars.dat")
SIM_FLOW     = Path("sim_flow_fp.dat")      # full-period flow output
SIM_HEAD     = Path("sim_head_fp.dat")      # full-period head output
UPW_FILE     = Path("modflow_GMRW.upw")
HED_FILE     = Path("modflow_GMRW.hed")
WELL_CSV     = Path(r"D:\nasrin\obs_data\head_mean_104.csv")

RCH_NO       = 5
SIM_START    = "2003-01"
CAL_START    = "2003-01"
CAL_END      = "2023-12"           # Ã¢â Â CHANGED from "2010-12"
N_OBS_FLOW   = 252                 # Ã¢â Â CHANGED from 96
N_OBS_HEAD   = 104
HDRY         = -999.0

# Ã¢ââ¬Ã¢ââ¬ 1. Read parameters Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
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

# Ã¢ââ¬Ã¢ââ¬ 2. SWAT file value setter/multiplier Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
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

# Ã¢ââ¬Ã¢ââ¬ 3a. Apply SWAT parameters Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
def apply_swat_params(params, run_dir=Path(".")):
    run_dir = Path(run_dir)
    bak_dir = run_dir / "_baseline_bak"
    for ext in [".mgt", ".gw", ".hru", ".rte"]:
        for bak in bak_dir.glob(f"*{ext}"):
            shutil.copy2(bak, run_dir / bak.name)
    shutil.copy2(bak_dir / "basins.bsn", run_dir / "basins.bsn")

    gw_params = {
        "ALPHA_BF": params.get("alpha_bf"),
        "GW_DELAY": params.get("gw_delay"),
        "GWQMN":    params.get("gwqmn"),
        "GW_REVAP": params.get("gw_revap"),
        "RCHRG_DP": params.get("rchrg_dp"),
        "REVAPMN":  params.get("revapmn"),
    }
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
            text, _ = set_param(text, "ESCO", esco)
            fp.write_text(text, encoding="latin-1")

    cn2_m = params.get("cn2_m")
    if cn2_m is not None:
        for fp in run_dir.glob("*.mgt"):
            text = fp.read_text(encoding="latin-1")
            text, _ = multiply_param(text, "CN2", cn2_m)
            fp.write_text(text, encoding="latin-1")

    ch_k2_m = params.get("ch_k2_m")
    if ch_k2_m is not None:
        for fp in run_dir.glob("*.rte"):
            text = fp.read_text(encoding="latin-1")
            text, _ = multiply_param(text, "CH_K2", ch_k2_m)
            fp.write_text(text, encoding="latin-1")

    surlag = params.get("surlag")
    if surlag is not None:
        bsn = run_dir / "basins.bsn"
        text = bsn.read_text(encoding="latin-1")
        text, _ = set_param(text, "SURLAG", surlag)
        bsn.write_text(text, encoding="latin-1")

# Ã¢ââ¬Ã¢ââ¬ 3b. Apply MODFLOW parameters to modflow_GMRW.upw Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
def apply_modflow_params(params, run_dir=Path(".")):
    run_dir = Path(run_dir)
    bak_upw = run_dir / "_baseline_bak" / "modflow_GMRW.upw"
    upw_path = run_dir / UPW_FILE
    shutil.copy2(bak_upw, upw_path)

    hk_mult = params.get("hk_mult", 1.0)
    sy_val  = params.get("sy",      None)
    ss_val  = params.get("ss",      None)

    lines = upw_path.read_text(encoding="latin-1").splitlines(keepends=True)
    out = []
    in_hk = False
    hk_header_seen = False

    for line in lines:
        if not hk_header_seen and re.search(r'INTERNAL.*HK', line, re.IGNORECASE):
            hk_header_seen = True
            in_hk = True
            out.append(line)
            continue

        if in_hk and re.search(r'VKA|SS\b|SY\b', line, re.IGNORECASE):
            in_hk = False

        if in_hk:
            if abs(hk_mult - 1.0) > 1e-9:
                new_tokens = []
                for tok in line.split():
                    try:
                        v = float(tok)
                        new_tok = str(round(v * hk_mult, 4)) if v != 0.0 else tok
                    except ValueError:
                        new_tok = tok
                    new_tokens.append(new_tok)
                line = " ".join(new_tokens) + "\n"
            out.append(line)
            continue

        if ss_val is not None and re.match(r'\s*CONSTANT\s+[\d.eE+\-]+\s+SS\b', line, re.IGNORECASE):
            line = f"CONSTANT {ss_val:.8g}            SS (specific storage)\n"
        if sy_val is not None and re.match(r'\s*CONSTANT\s+[\d.eE+\-]+\s+SY\b', line, re.IGNORECASE):
            line = f"CONSTANT {sy_val:.8g}              SY (specific yield)\n"

        out.append(line)

    upw_path.write_text("".join(out), encoding="latin-1")
    print(f"  UPW: hk_mult={hk_mult:.4f}  sy={sy_val}  ss={ss_val}", flush=True)

# Ã¢ââ¬Ã¢ââ¬ 4. Parse modflow_GMRW.hed (formatted ASCII, (213F10.2) LABEL) Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
def parse_hed_file(hed_path, nrow=197, ncol=135):
    frames = []
    lines  = Path(hed_path).read_text(encoding="latin-1").splitlines()
    n      = len(lines)
    i      = 0

    while i < n:
        parts = lines[i].split()
        if len(parts) >= 7:
            try:
                float(parts[0])
                float(parts[2])
                stripped = [p for p in parts if not p.startswith('(')]
                tag = " ".join(stripped[4:-3]).strip().upper()
                if "HEAD" in tag:
                    nc = int(stripped[-3])
                    nr = int(stripped[-2])
                    il = int(stripped[-1])
                    i += 1
                    arr = np.full((nr, nc), np.nan)
                    for r in range(nr):
                        if i >= n:
                            break
                        dline = lines[i]
                        for c in range(nc):
                            chunk = dline[c*10:(c+1)*10]
                            if not chunk.strip():
                                break
                            try:
                                v = float(chunk)
                                arr[r, c] = np.nan if abs(v - HDRY) < 1.0 else v
                            except ValueError:
                                pass
                        i += 1
                    frames.append(arr)
                    continue
            except (ValueError, IndexError):
                pass
        i += 1

    return frames

# Ã¢ââ¬Ã¢ââ¬ 5. Extract simulated heads at 104 well locations Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
def extract_sim_heads(run_dir=Path(".")):
    hed_path = Path(run_dir) / HED_FILE
    wells    = pd.read_csv(WELL_CSV)

    frames = parse_hed_file(hed_path)
    if not frames:
        print("  [WARN] No head frames parsed from .hed file!", flush=True)
        return np.full(N_OBS_HEAD, -9999.0)

    stack = np.stack(frames, axis=0)
    with np.errstate(all="ignore"):
        mean_head = np.nanmean(stack, axis=0)

    sim_heads = np.full(N_OBS_HEAD, -9999.0)
    for idx, row in wells.iterrows():
        r = int(row["row"])  - 1
        c = int(row["clo"])  - 1
        h = mean_head[r, c]
        sim_heads[idx] = h if np.isfinite(h) else -9999.0

    valid = np.sum(sim_heads > -9000)
    print(f"  Heads extracted: {valid}/{N_OBS_HEAD} valid  "
          f"(mean={np.mean(sim_heads[sim_heads > -9000]):.2f} m)", flush=True)
    return sim_heads

# Ã¢ââ¬Ã¢ââ¬ 6. Read output.rch Ã¢â¬â full-period streamflow extraction Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
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
    print(f"  output.rch: {len(rch5)} monthly rows for RCH {RCH_NO} "
          f"({rch5.index[0].strftime('%Y-%m')} Ã¢â â {rch5.index[-1].strftime('%Y-%m')})",
          flush=True)
    return rch5["FLOW_OUTcms"].rename("sim_cms").loc[CAL_START:CAL_END]

# Ã¢ââ¬Ã¢ââ¬ 7. Required-file validation Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
REQUIRED_FILES = [
    "file.cio", "basins.bsn", "plant.dat", "till.dat",
    "urban.dat", "septwq.dat", "pest.dat", "fert.dat",
]

def validate_required_files(run_dir):
    missing = [f for f in REQUIRED_FILES if not (run_dir / f).exists()]
    empty   = [f for f in REQUIRED_FILES if (run_dir / f).exists()
               and (run_dir / f).stat().st_size == 0]
    if missing or empty:
        print(f"FATAL: Missing={missing}  Empty={empty}", flush=True)
        sys.exit(24)
    bak_upw = run_dir / "_baseline_bak" / "modflow_GMRW.upw"
    if not bak_upw.exists():
        print(f"FATAL: _baseline_bak/modflow_GMRW.upw not found Ã¢â¬â run build_ies_pst_fullperiod.py first",
              flush=True)
        sys.exit(24)
    print(f"  [OK] Pre-run check OK ({len(REQUIRED_FILES)} SWAT files + UPW backup)", flush=True)


# -- 8. Main ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_dir = Path(".")
        validate_required_files(run_dir)

        params = read_params(run_dir / PAR_FILE)
        print(f"  Parameters: { {k: f'{v:.5g}' for k,v in params.items()} }", flush=True)

        apply_swat_params(params, run_dir)
        apply_modflow_params(params, run_dir)
        print("  All parameters applied.", flush=True)

        rch_size_before = (run_dir / "output.rch").stat().st_size \
            if (run_dir / "output.rch").exists() else -1
        t_start = time.time()
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
            print(f"  [WARN] Stream error: {e}", flush=True)

        proc.wait()
        elapsed = time.time() - t_start
        print(f"  Done in {elapsed:.1f}s (exit={proc.returncode}) [{time.strftime('%H:%M:%S')}]",
              flush=True)

        if (run_dir / "output.rch").exists():
            delta = (run_dir / "output.rch").stat().st_size - rch_size_before
            print(f"  output.rch delta: {delta:+,} bytes", flush=True)

        if proc.returncode != 0:
            print(f"  ERROR: exit={proc.returncode}.  Last output: {last_line}", flush=True)
            SIM_FLOW.write_text("\n".join(["-9999.0"] * N_OBS_FLOW) + "\n")
            SIM_HEAD.write_text("\n".join(["-9999.0"] * N_OBS_HEAD) + "\n")
            sys.exit(0)

        # -- Extract streamflow -----------------------------------------------
        flow_vals = np.full(N_OBS_FLOW, -9999.0)
        try:
            sim_fp = read_sim_flow(run_dir)
            n = len(sim_fp)
            if n < N_OBS_FLOW:
                print(f"  [WARN] Only {n} flow months extracted (expected {N_OBS_FLOW})!", flush=True)
                padded = list(sim_fp.values) + [-9999.0] * (N_OBS_FLOW - n)
                flow_vals = np.array(padded)
            elif n > N_OBS_FLOW:
                flow_vals = sim_fp.values[:N_OBS_FLOW]
            else:
                flow_vals = sim_fp.values
        except Exception as e:
            print(f"  ERROR reading output.rch: {str(e)[:200]}", flush=True)

        SIM_FLOW.write_text("\n".join(f"{v:.6f}" for v in flow_vals) + "\n")
        valid_f = flow_vals[flow_vals > -9000]
        if len(valid_f):
            print(f"  sim_flow_fp.dat: {N_OBS_FLOW} values written  "
                  f"(range {valid_f.min():.1f}-{valid_f.max():.1f} m3/s)", flush=True)
        else:
            print(f"  sim_flow_fp.dat: {N_OBS_FLOW} values written  (all -9999)", flush=True)

        # -- Extract heads ----------------------------------------------------
        head_vals = np.full(N_OBS_HEAD, -9999.0)
        try:
            head_vals = extract_sim_heads(run_dir)
        except Exception as e:
            print(f"  ERROR reading .hed file: {str(e)[:200]}", flush=True)

        SIM_HEAD.write_text("\n".join(f"{v:.4f}" for v in head_vals) + "\n")
        print(f"  sim_head_fp.dat: {N_OBS_HEAD} values written", flush=True)
        print("  [DONE] forward_run_fullperiod.py complete.", flush=True)

    except SystemExit:
        raise
    except Exception as _fatal:
        print(f"  FATAL unhandled exception: {_fatal}", flush=True)
        try:
            SIM_FLOW.write_text("\n".join(["-9999.0"] * N_OBS_FLOW) + "\n")
            SIM_HEAD.write_text("\n".join(["-9999.0"] * N_OBS_HEAD) + "\n")
        except Exception:
            pass
    sys.exit(0)
