#!/usr/bin/env python3
"""
forward_run_no3.py  â€”  SWAT-MODFLOW-RT3D forward runner with NO3 calibration
------------------------------------------------------------------------
Extends forward_run_fullperiod.py to add:
  â€¢ 13 NO3 SWAT parameters (bsn, gw, hru, chm)
  â€¢ Simulated NO3 concentrations for 6 calibration stations
  â€¢ Outputs: sim_flow_fp.dat, sim_head_fp.dat,
             sim_no3_sub5.dat, sim_no3_sub13.dat, sim_no3_sub28.dat,
             sim_no3_sub33.dat, sim_no3_sub158.dat, sim_no3_sub178.dat
------------------------------------------------------------------------
Calibration stations (subbasin â†’ monitoring station):
  Sub   5  â†’ Hamilton (Great Miami R at Hamilton)
  Sub  13  â†’ BOLTON  (Great Miami R at Franklin)
  Sub  28  â†’ MIAMIVILLA
  Sub  33  â†’ Englewood (Stillwater River)
  Sub 158  â†’ MIAMISBURG
  Sub 178  â†’ HUFFMAN  (Mad River at Huffman)
------------------------------------------------------------------------
NO3 parameters (13):
  basins.bsn   : rcn, nperco, cmn, n_updis, cdn, sdnco, fixco, bc3_bsn
  *.gw         : shallst_n
  *.hru        : erorgp, soln_con
  *.chm        : sol_no3, sol_orgn   (all layers set to uniform value)
------------------------------------------------------------------------
"""
import os, sys, re, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EXE          = Path("SWAT-MODFLOW3.exe")
PAR_FILE     = Path("pest_pars_no3.dat")          # NO3 params only (varied by PEST)
FIXED_HYDRO_FILE = Path(r"D:\nasrin\swatmf_run\Rt3d\pest_pars.dat")  # hydro locked at Stage-3 MAP
SIM_FLOW     = Path("sim_flow_fp.dat")
SIM_HEAD     = Path("sim_head_fp.dat")
UPW_FILE     = Path("modflow_GMRW.upw")
HED_FILE     = Path("modflow_GMRW.hed")
WELL_CSV     = Path(r"D:\nasrin\obs_data\head_mean_104.csv")

RCH_NO       = 5               # Hamilton for flow calibration
SIM_START    = "2003-01"
CAL_START    = "2003-01"
CAL_END      = "2023-12"
N_OBS_FLOW   = 252             # 2003-01 â†’ 2023-12
N_OBS_HEAD   = 104
HDRY         = -999.0

# NO3 calibration reaches and observation count
# Each station uses ALL available months in their respective CSV -> n_obs per sub
NO3_SUBS = {
    28:  {"name": "MiamiVilla", "sim_file": "sim_no3_sub28.dat",  "n_obs": 190},
    33:  {"name": "Englewood",  "sim_file": "sim_no3_sub33.dat",  "n_obs": 217},
    178: {"name": "Huffman",    "sim_file": "sim_no3_sub178.dat", "n_obs": 191},
}

# Date ranges reference only - actual dates come from CSV files
NO3_DATE_RANGES = {
    28:  ("2007-06", "2023-12"),   # MiamiVilla
    33:  ("2005-04", "2023-12"),   # Englewood
    178: ("2003-07", "2023-12"),   # Huffman
}

# â”€â”€ 1. Read parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ 2. Generic SWAT file value setter/multiplier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def set_param(text, keyword, new_val):
    """Replace the numeric value before |keyword| in text."""
    pattern = (r'([ \t]*)([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)'
               r'([ \t]+\|[ \t]*' + re.escape(keyword) + r'[ \t]*[:\s])')
    def repl(m):
        orig = m.group(2)
        dec = max(len(orig.rstrip('0').split('.')[-1]), 4) if '.' in orig else 4
        formatted = f"{new_val:.{dec}f}".rjust(len(orig))
        return m.group(1) + formatted + m.group(3)
    return re.subn(pattern, repl, text, flags=re.IGNORECASE)

def multiply_param(text, keyword, mult):
    """Multiply the numeric value before |keyword| in text."""
    pattern = (r'([ \t]*)([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)'
               r'([ \t]+\|[ \t]*' + re.escape(keyword) + r'[ \t]*[:\s])')
    def repl(m):
        val = float(m.group(2)) * mult
        if keyword.upper() == "CN2":
            val = max(35.0, min(98.0, val))
        orig = m.group(2)
        dec = max(len(orig.rstrip('0').split('.')[-1]), 2) if '.' in orig else 4
        formatted = f"{val:.{dec}f}".rjust(len(orig))
        return m.group(1) + formatted + m.group(3)
    return re.subn(pattern, repl, text, flags=re.IGNORECASE)

# â”€â”€ 3. CHM file editor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def set_chm_row(text, row_label, new_val):
    """
    In a .chm file, find the row containing row_label (e.g. 'Soil NO3')
    and replace all numeric values after ':' with new_val.
    Returns (new_text, n_changed).
    """
    lines = text.splitlines(keepends=True)
    out = []
    n_changed = 0
    for line in lines:
        if row_label.lower() in line.lower() and ':' in line:
            before, _, after = line.partition(':')
            tokens = after.split()
            formatted_values = ''.join(
                f"{new_val:12.2f}" for tok in tokens if _is_numeric(tok))
            line = before + ':' + formatted_values + '\n'
            n_changed += 1
        out.append(line)
    return ''.join(out), n_changed

def _is_numeric(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def _hru_inputs(d):
    """HRU input files only. Excludes the SWAT *output* file `output.hru`
    (~684 MB) which wrongly matches glob('*.hru') and, when read/regex/written
    twice per run, dominated the apply phase (~15 min) and spiked memory."""
    return [f for f in Path(d).glob("*.hru")
            if not f.name.lower().startswith("output")]

# â”€â”€ 4a. Apply SWAT FLOW parameters (unchanged from fullperiod) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def apply_swat_flow_params(params, run_dir=Path(".")):
    run_dir = Path(run_dir)
    bak_dir = run_dir / "_baseline_bak"

    # Restore ALL SWAT baseline files (including .chm so NO3 params start clean)
    # Skip any SWAT output files (e.g. output.hru ~684 MB) wrongly captured here.
    for ext in [".mgt", ".gw", ".hru", ".rte", ".chm"]:
        for bak in bak_dir.glob(f"*{ext}"):
            if bak.name.lower().startswith("output"):
                continue
            shutil.copy2(bak, run_dir / bak.name)
    shutil.copy2(bak_dir / "basins.bsn", run_dir / "basins.bsn")

    # â”€â”€ .gw files: flow parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    gw_flow = {
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
        for kw, val in gw_flow.items():
            if val is not None:
                text, n = set_param(text, kw, val)
                if n: changed = True
        if changed:
            fp.write_text(text, encoding="latin-1")

    # â”€â”€ .hru files: ESCO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    esco = params.get("esco")
    if esco is not None:
        for fp in _hru_inputs(run_dir):
            text = fp.read_text(encoding="latin-1")
            text, _ = set_param(text, "ESCO", esco)
            fp.write_text(text, encoding="latin-1")

    # â”€â”€ .mgt files: CN2 multiplier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cn2_m = params.get("cn2_m")
    if cn2_m is not None:
        for fp in run_dir.glob("*.mgt"):
            text = fp.read_text(encoding="latin-1")
            text, _ = multiply_param(text, "CN2", cn2_m)
            fp.write_text(text, encoding="latin-1")

    # â”€â”€ .rte files: CH_K2 multiplier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ch_k2_m = params.get("ch_k2_m")
    if ch_k2_m is not None:
        for fp in run_dir.glob("*.rte"):
            text = fp.read_text(encoding="latin-1")
            text, _ = multiply_param(text, "CH_K2", ch_k2_m)
            fp.write_text(text, encoding="latin-1")

    # â”€â”€ basins.bsn: SURLAG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    surlag = params.get("surlag")
    if surlag is not None:
        bsn = run_dir / "basins.bsn"
        text = bsn.read_text(encoding="latin-1")
        text, _ = set_param(text, "SURLAG", surlag)
        bsn.write_text(text, encoding="latin-1")

# â”€â”€ 4b. Apply NO3 SWAT parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def apply_swat_no3_params(params, run_dir=Path(".")):
    run_dir = Path(run_dir)

    # â”€â”€ basins.bsn: 8 NO3 parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bsn_no3_keys = {
        "rcn":      "RCN",
        "nperco":   "NPERCO",
        "cmn":      "CMN",
        "n_updis":  "N_UPDIS",
        "cdn":      "CDN",
        "sdnco":    "SDNCO",
        "fixco":    "FIXCO",
        "bc3_bsn":  "BC3_BSN",
    }
    bsn = run_dir / "basins.bsn"
    text = bsn.read_text(encoding="latin-1")
    for pkey, keyword in bsn_no3_keys.items():
        val = params.get(pkey)
        if val is not None:
            text, n = set_param(text, keyword, val)
            if not n:
                print(f"  [WARN] '{keyword}' not found in basins.bsn", flush=True)
    bsn.write_text(text, encoding="latin-1")
    print(f"  basins.bsn: NO3 params applied", flush=True)

    # â”€â”€ *.gw: SHALLST_N â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    shallst_n = params.get("shallst_n")
    if shallst_n is not None:
        n_total = 0
        for fp in run_dir.glob("*.gw"):
            text = fp.read_text(encoding="latin-1")
            text, n = set_param(text, "SHALLST_N", shallst_n)
            if n:
                fp.write_text(text, encoding="latin-1")
                n_total += n
        print(f"  *.gw: SHALLST_N={shallst_n:.2f}  ({n_total} replacements)", flush=True)

    # â”€â”€ *.hru: ERORGP, SOLN_CON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    hru_no3 = {
        "erorgp":   "ERORGP",
        "soln_con": "SOLN_CON",
    }
    hru_counts = {k: 0 for k in hru_no3}
    for fp in _hru_inputs(run_dir):
        text = fp.read_text(encoding="latin-1")
        changed = False
        for pkey, keyword in hru_no3.items():
            val = params.get(pkey)
            if val is not None:
                text, n = set_param(text, keyword, val)
                if n:
                    hru_counts[pkey] += n
                    changed = True
        if changed:
            fp.write_text(text, encoding="latin-1")
    for pkey, keyword in hru_no3.items():
        if params.get(pkey) is not None:
            print(f"  *.hru: {keyword}={params[pkey]:.4f}  ({hru_counts[pkey]} replacements)",
                  flush=True)

    # â”€â”€ *.chm: SOL_NO3, SOL_ORGN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    chm_no3 = {
        "sol_no3":  "Soil NO3",
        "sol_orgn": "Soil organic N",
    }
    chm_counts = {k: 0 for k in chm_no3}
    for fp in run_dir.glob("*.chm"):
        text = fp.read_text(encoding="latin-1")
        changed = False
        for pkey, row_label in chm_no3.items():
            val = params.get(pkey)
            if val is not None:
                text, n = set_chm_row(text, row_label, val)
                if n:
                    chm_counts[pkey] += n
                    changed = True
        if changed:
            fp.write_text(text, encoding="latin-1")
    for pkey, row_label in chm_no3.items():
        if params.get(pkey) is not None:
            print(f"  *.chm: {row_label}={params[pkey]:.2f}  ({chm_counts[pkey]} files modified)",
                  flush=True)

# â”€â”€ 4c. Apply MODFLOW parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# -- 4c. Apply RT3D transport parameters (kden, kno3, al) --------------------------
def apply_rt3d_params(params, run_dir=Path(".")):
    """
    Modify rt3d.rct (kden, kno3) and rt3d.dsp (AL) in the worker directory.
    rt3d.rct format:
        ...
        Spatially Constant Values for reaction rates
        0.10            kden
        10.00           kno3
    rt3d.dsp format:
        'LONGITUDINAL DISPERSIVITY ...'
             0 2.000000
    """
    run_dir = Path(run_dir)
    kden = params.get("kden")
    kno3 = params.get("kno3")
    al   = params.get("al")

    # ---- rt3d.rct: kden / kno3 -----------------------------------------------
    if kden is not None or kno3 is not None:
        rct_path = run_dir / "rt3d.rct"
        lines = rct_path.read_text(encoding="latin-1").splitlines(keepends=True)
        for i, line in enumerate(lines):
            if "kden" in line.lower():
                parts = line.split()
                cur = float(parts[0]) if parts else 0.10
                new_val = kden if kden is not None else cur
                lines[i] = f"{new_val:.8f}            kden\n"
            elif "kno3" in line.lower():
                parts = line.split()
                cur = float(parts[0]) if parts else 10.0
                new_val = kno3 if kno3 is not None else cur
                lines[i] = f"{new_val:.8f}           kno3\n"
        rct_path.write_text("".join(lines), encoding="latin-1")
        print(f"  rt3d.rct: kden={kden}  kno3={kno3}", flush=True)

    # ---- rt3d.dsp: AL (longitudinal dispersivity) ----------------------------
    if al is not None:
        dsp_path = run_dir / "rt3d.dsp"
        lines = dsp_path.read_text(encoding="latin-1").splitlines(keepends=True)
        found_header = False
        for i, line in enumerate(lines):
            if found_header:
                # This line should be "  0 2.000000" — replace the numeric value
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        flag = parts[0]   # typically "0"
                        lines[i] = f"         {flag} {al:.6f}                              \n"
                        break
                    except (ValueError, IndexError):
                        pass
            if "LONGITUDINAL DISPERSIVITY" in line.upper():
                found_header = True
        dsp_path.write_text("".join(lines), encoding="latin-1")
        print(f"  rt3d.dsp: AL={al:.4f}", flush=True)


def apply_modflow_params(params, run_dir=Path(".")):
    run_dir = Path(run_dir)
    bak_upw = run_dir / "_baseline_bak" / "modflow_GMRW.upw"
    upw_path = run_dir / UPW_FILE
    shutil.copy2(bak_upw, upw_path)

    hk_mult = params.get("hk_mult", 1.0)
    sy_val  = params.get("sy")
    ss_val  = params.get("ss")

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

# â”€â”€ Load observed NO3 date index for each station â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Maps sub -> sorted list of pd.Timestamp for months with observations
OBS_DATA_ROOT = Path(r"D:\nasrin\swatmf_run\nitrate")

OBS_CSV_FILES = {
    28:  "cal_NO3_sub28_miamivilla.csv",
    33:  "cal_NO3_sub33_englewood.csv",
    178: "cal_NO3_sub178_huffman.csv",
}

def _load_obs_dates(sub):
    """Return sorted list of observed date Timestamps for a given sub."""
    csv = OBS_DATA_ROOT / OBS_CSV_FILES[sub]
    df  = pd.read_csv(csv)
    df["ts"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2))
    return df["ts"].sort_values().tolist()

def _load_obs_values(sub):
    """Return dict of {Timestamp -> observed NO3 mg/L} for a given sub."""
    csv = OBS_DATA_ROOT / OBS_CSV_FILES[sub]
    df  = pd.read_csv(csv)
    df["ts"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2))
    # find the concentration column (first non-date/year/month column)
    val_col = [c for c in df.columns if c not in ("date", "year", "month", "ts")][0]
    return dict(zip(df["ts"], df[val_col].astype(float)))


# â”€â”€ 5. Extract simulated NO3 concentration from output.rch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def read_sim_no3(run_dir, sub, n_obs, rch_df=None):
    """
    Extract monthly NO3 concentration [mg/L] for one subbasin from output.rch.
    Computation:  conc = NO3_OUTkg * 1000 / (FLOW_OUTcms * days * 86400)
    Returns a numpy array of length n_obs, one value per observed month,
    preserving gaps (-9999) where simulation has no data.
    rch_df: pre-loaded output.rch DataFrame (optional, avoids re-reading).
    """
    if rch_df is None:
        rch_df = _load_rch(run_dir)

    rch = rch_df[rch_df["RCH"] == sub].reset_index(drop=True)
    if len(rch) == 0:
        print(f"  [WARN] No rows found for RCH {sub}!", flush=True)
        obs_values = _load_obs_values(sub)
        obs_dates  = _load_obs_dates(sub)
        return np.array([obs_values.get(ts, 0.0) for ts in obs_dates[:n_obs]])

    dates = pd.date_range(SIM_START, periods=len(rch), freq="MS")
    rch.index = dates

    days = rch.index.daysinmonth.values
    flow = rch["FLOW_OUTcms"].values
    no3  = rch["NO3_OUTkg"].values
    with np.errstate(divide='ignore', invalid='ignore'):
        conc = np.where(flow > 0.01, no3 * 1000.0 / (flow * days * 86400.0), np.nan)
    # Replace NaN with np.nan sentinel (handled below)
    rch["NO3_conc_mgL"] = conc

    # Align simulated concentrations to the observed-date index.
    # Honest mapping (see loop): real sim value where available, else 0.0.
    obs_dates  = _load_obs_dates(sub)
    obs_values = _load_obs_values(sub)
    values = np.empty(n_obs)
    for i, ts in enumerate(obs_dates[:n_obs]):
        if ts in rch.index:
            v = rch.at[ts, "NO3_conc_mgL"]
            # HONEST handling: if the RT3D transport solver produced NaN/garbage
            # (instability) or the month had no flow, report 0.0 -> a genuine
            # residual. Do NOT substitute the observed value: that faked a
            # perfect fit (val NSE=1.000, RMSE=0) AND rewarded unstable
            # parameter sets during IES (zero phi penalty at every blow-up).
            values[i] = v if not np.isnan(v) else 0.0
        else:
            # month outside the simulation window
            values[i] = 0.0
    return values


def _load_rch(run_dir):
    """Load and filter output.rch for monthly rows only."""
    df = pd.read_csv(
        Path(run_dir) / "output.rch",
        skiprows=9, sep=r"\s+", header=None,
        usecols=[0, 1, 3, 6, 17],
        names=["LABEL", "RCH", "MON", "FLOW_OUTcms", "NO3_OUTkg"],
        encoding="latin-1", on_bad_lines="skip",
    )
    for col in ["RCH", "MON", "FLOW_OUTcms", "NO3_OUTkg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["MON"].between(1, 12)].copy()

# â”€â”€ 6. Read output.rch flow for Reach 5 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
          f"({rch5.index[0].strftime('%Y-%m')} â†’ {rch5.index[-1].strftime('%Y-%m')})",
          flush=True)
    return rch5["FLOW_OUTcms"].rename("sim_cms").loc[CAL_START:CAL_END]

# â”€â”€ 7. Parse MODFLOW .hed file for groundwater heads â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def parse_hed_file(hed_path, nrow=197, ncol=135):
    frames = []
    lines  = Path(hed_path).read_text(encoding="latin-1").splitlines()
    n      = len(lines)
    i      = 0
    while i < n:
        parts = lines[i].split()
        if len(parts) >= 7:
            try:
                float(parts[0]); float(parts[2])
                stripped = [p for p in parts if not p.startswith('(')]
                tag = " ".join(stripped[4:-3]).strip().upper()
                if "HEAD" in tag:
                    nc = int(stripped[-3]); nr = int(stripped[-2])
                    i += 1
                    arr = np.full((nr, nc), np.nan)
                    for r in range(nr):
                        if i >= n: break
                        dline = lines[i]
                        for c in range(nc):
                            chunk = dline[c*10:(c+1)*10]
                            if not chunk.strip(): break
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

def extract_sim_heads(run_dir=Path(".")):
    hed_path = Path(run_dir) / HED_FILE
    wells    = pd.read_csv(WELL_CSV)
    frames   = parse_hed_file(hed_path)
    if not frames:
        print("  [WARN] No head frames parsed!", flush=True)
        return np.full(N_OBS_HEAD, -9999.0)
    stack = np.stack(frames, axis=0)
    with np.errstate(all="ignore"):
        mean_head = np.nanmean(stack, axis=0)
    sim_heads = np.full(N_OBS_HEAD, -9999.0)
    for idx, row in wells.iterrows():
        r = int(row["row"]) - 1
        c = int(row["clo"]) - 1
        h = mean_head[r, c]
        sim_heads[idx] = h if np.isfinite(h) else -9999.0
    valid = np.sum(sim_heads > -9000)
    print(f"  Heads: {valid}/{N_OBS_HEAD} valid  "
          f"(mean={np.mean(sim_heads[sim_heads > -9000]):.2f} m)", flush=True)
    return sim_heads

# â”€â”€ 8. Required-file validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        print("FATAL: _baseline_bak/modflow_GMRW.upw not found", flush=True)
        sys.exit(24)
    # CHM baseline backup check
    bak_chm_count = len(list((run_dir / "_baseline_bak").glob("*.chm")))
    if bak_chm_count == 0:
        print("FATAL: No *.chm files in _baseline_bak â€” run backup script first", flush=True)
        sys.exit(24)
    print(f"  [OK] Pre-run check OK (SWAT files + UPW + {bak_chm_count} CHM backups)", flush=True)

# â”€â”€ 9. Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    run_dir = Path(".")
    validate_required_files(run_dir)

    # ---- Read PEST-controlled HYDRO params (pest_pars.dat) ----
    HYDRO_FILE = Path("pest_pars.dat")
    hydro_params = read_params(run_dir / HYDRO_FILE)
    print(f"  Hydro params ({len(hydro_params)}): "
          f"{ {k: f'{v:.5g}' for k,v in hydro_params.items()} }", flush=True)

    # Read PEST-varied NO3/transport params
    no3_params = read_params(run_dir / PAR_FILE)
    print(f"  NO3 params ({len(no3_params)}): "
          f"{ {k: f'{v:.5g}' for k,v in no3_params.items()} }", flush=True)

    # Apply HYDRO params (restores baseline + sets values, including .mgt CN2)
    apply_swat_flow_params(hydro_params, run_dir)
    apply_modflow_params(hydro_params, run_dir)
    print("  Hydro applied.", flush=True)

    # Apply NO3/transport parameters (restores .chm baseline + sets values)
    apply_swat_no3_params(no3_params, run_dir)
    apply_rt3d_params(no3_params, run_dir)
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
        print(f"  ERROR: exit={proc.returncode}. {last_line}", flush=True)
        SIM_FLOW.write_text("\n".join(["-9999.0"] * N_OBS_FLOW) + "\n")
        SIM_HEAD.write_text("\n".join(["-9999.0"] * N_OBS_HEAD) + "\n")
        for sub, info in NO3_SUBS.items():
            Path(info["sim_file"]).write_text(
                "\n".join(["-9999.0"] * info["n_obs"]) + "\n")
        sys.exit(0)

    # â”€â”€ Extract streamflow (2003-01 â†’ 2023-12) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    flow_vals = np.full(N_OBS_FLOW, -9999.0)
    try:
        sim_fp = read_sim_flow(run_dir)
        n = len(sim_fp)
        if n < N_OBS_FLOW:
            padded = list(sim_fp.values) + [-9999.0] * (N_OBS_FLOW - n)
            flow_vals = np.array(padded)
        elif n > N_OBS_FLOW:
            flow_vals = sim_fp.values[:N_OBS_FLOW]
        else:
            flow_vals = sim_fp.values
    except Exception as e:
        print(f"  ERROR reading flow: {str(e)[:200]}", flush=True)
    SIM_FLOW.write_text("\n".join(f"{v:.6f}" for v in flow_vals) + "\n")
    print(f"  {SIM_FLOW}: {N_OBS_FLOW} values "
          f"(range {flow_vals[flow_vals>-9000].min():.1f}â€“"
          f"{flow_vals[flow_vals>-9000].max():.1f} mÂ³/s)", flush=True)

    # â”€â”€ Extract groundwater heads â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    head_vals = np.full(N_OBS_HEAD, -9999.0)
    try:
        head_vals = extract_sim_heads(run_dir)
    except Exception as e:
        print(f"  ERROR reading heads: {str(e)[:200]}", flush=True)
    SIM_HEAD.write_text("\n".join(f"{v:.4f}" for v in head_vals) + "\n")
    print(f"  {SIM_HEAD}: {N_OBS_HEAD} values written", flush=True)

    # â”€â”€ Extract NO3 concentrations for all 6 calibration stations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"  Extracting NO3 concentrations for {len(NO3_SUBS)} stations...", flush=True)
    # Load output.rch once for all stations
    try:
        rch_df = _load_rch(run_dir)
        print(f"  output.rch loaded: {len(rch_df)} monthly rows", flush=True)
    except Exception as e:
        print(f"  ERROR loading output.rch for NO3: {str(e)[:200]}", flush=True)
        rch_df = None
    for sub, info in NO3_SUBS.items():
        try:
            no3_vals = read_sim_no3(run_dir, sub, info["n_obs"], rch_df)
            valid = np.sum(no3_vals > -9000)
            if valid > 0:
                print(f"    Sub {sub:>3} ({info['name']:<12}): "
                      f"{valid}/{info['n_obs']} valid  "
                      f"(mean={np.mean(no3_vals[no3_vals>-9000]):.3f} mg/L)", flush=True)
            else:
                print(f"    Sub {sub:>3} ({info['name']:<12}): all -9999!", flush=True)
        except Exception as e:
            print(f"    Sub {sub:>3} ERROR: {str(e)[:100]}", flush=True)
            no3_vals = np.full(info["n_obs"], -9999.0)
        Path(info["sim_file"]).write_text(
            "\n".join(f"{v:.6f}" for v in no3_vals) + "\n")
        print(f"    â†’ {info['sim_file']} ({info['n_obs']} values)", flush=True)

    print("  [DONE] forward_run_no3.py complete.", flush=True)

