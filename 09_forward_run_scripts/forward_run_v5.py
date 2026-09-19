#!/usr/bin/env python3
"""
forward_run_v5.py  â€”  Joint SWAT+MODFLOW forward runner for PESTPP-IES v5
------------------------------------------------------------------------
Extends forward_run.py with MODFLOW parameter support and head extraction.

Parameters applied
  SWAT  (10) : alpha_bf, ch_k2_m, cn2_m, esco, gw_delay, gw_revap,
               gwqmn, surlag, rchrg_dp, revapmn
  MODFLOW (3): hk_mult (global HK multiplier), sy, ss

Observations
  sim_flow.dat â€” 96 monthly Reach-5 streamflow (2003-01 â†’ 2010-12)
  sim_head.dat â€” 85 time-averaged groundwater heads (mean of saved steps)

MODFLOW grid : 1 layer Ã— 197 rows Ã— 135 cols  (modflow_GMRW.upw)
Head output  : modflow_GMRW.hed (formatted ASCII, (213F10.2) LABEL)
------------------------------------------------------------------------
"""
import os, sys, re, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

# Force UTF-8 stdout/stderr â€” prevents UnicodeEncodeError on Windows CP1252 terminals
# and when running as a subprocess under PESTPP-IES
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EXE          = Path("SWAT-MODFLOW3.exe")
PAR_FILE     = Path("pest_pars.dat")
SIM_FLOW     = Path("sim_flow.dat")
SIM_HEAD     = Path("sim_head.dat")
UPW_FILE     = Path("modflow_GMRW.upw")
HED_FILE     = Path("modflow_GMRW.hed")
WELL_CSV     = Path(r"D:\nasrin\obs_data\head_mean_104.csv")

RCH_NO       = 5
SIM_START    = "2003-01"
CAL_START    = "2003-01"
CAL_END      = "2010-12"
N_OBS_FLOW   = 96
N_OBS_HEAD   = 104
HDRY         = -999.0          # MODFLOW HDRY â€” inactive cell marker

# â”€â”€ 1. Read parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ 2. SWAT file value setter/multiplier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ 3a. Apply SWAT parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ 3b. Apply MODFLOW parameters to modflow_GMRW.upw â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def apply_modflow_params(params, run_dir=Path(".")):
    """Restore UPW from backup, then apply hk_mult / sy / ss."""
    run_dir = Path(run_dir)
    bak_upw = run_dir / "_baseline_bak" / "modflow_GMRW.upw"
    upw_path = run_dir / UPW_FILE

    # Restore original UPW (mandatory so each run starts from baseline)
    shutil.copy2(bak_upw, upw_path)

    hk_mult = params.get("hk_mult", 1.0)
    sy_val  = params.get("sy",      None)
    ss_val  = params.get("ss",      None)

    lines = upw_path.read_text(encoding="latin-1").splitlines(keepends=True)
    out = []
    in_hk = False  # True while reading HK data rows
    hk_header_seen = False
    row_count = 0

    for line in lines:
        # Detect start of HK INTERNAL block
        if not hk_header_seen and re.search(r'INTERNAL.*HK', line, re.IGNORECASE):
            hk_header_seen = True
            in_hk = True
            out.append(line)
            continue

        # Detect VKA â€” signals end of HK block
        if in_hk and re.search(r'VKA|SS\b|SY\b', line, re.IGNORECASE):
            in_hk = False

        if in_hk:
            # Modify HK row: multiply non-zero values by hk_mult
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

        # Replace SS / SY CONSTANT lines
        if ss_val is not None and re.match(r'\s*CONSTANT\s+[\d.eE+\-]+\s+SS\b', line, re.IGNORECASE):
            line = f"CONSTANT {ss_val:.8g}            SS (specific storage)\n"
        if sy_val is not None and re.match(r'\s*CONSTANT\s+[\d.eE+\-]+\s+SY\b', line, re.IGNORECASE):
            line = f"CONSTANT {sy_val:.8g}              SY (specific yield)\n"

        out.append(line)

    upw_path.write_text("".join(out), encoding="latin-1")
    print(f"  UPW: hk_mult={hk_mult:.4f}  sy={sy_val}  ss={ss_val}", flush=True)

# â”€â”€ 4. Parse modflow_GMRW.hed (formatted ASCII, (213F10.2) LABEL) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def parse_hed_file(hed_path, nrow=197, ncol=135):
    """
    Parse all saved time step head arrays from a MODFLOW formatted .hed file.

    Returns
    -------
    list of 2-D numpy arrays (nrow Ã— ncol), one per saved time step.
    Inactive cells (HDRY) are set to np.nan.
    """
    frames = []
    lines  = Path(hed_path).read_text(encoding="latin-1").splitlines()
    n      = len(lines)
    i      = 0

    while i < n:
        # Look for header: KSTP KPER PERTIM TOTIM TEXT NCOL NROW ILAY [FORMAT]
        # MODFLOW formatted head output may append e.g. (213F10.2) as last token
        parts = lines[i].split()
        if len(parts) >= 7:
            try:
                float(parts[0])   # KSTP
                float(parts[2])   # PERTIM
                # Strip optional trailing Fortran format string like (213F10.2)
                stripped = [p for p in parts if not p.startswith('(')]
                tag = " ".join(stripped[4:-3]).strip().upper()
                if "HEAD" in tag:
                    nc = int(stripped[-3])   # NCOL
                    nr = int(stripped[-2])   # NROW
                    il = int(stripped[-1])   # ILAY
                    i += 1
                    arr = np.full((nr, nc), np.nan)
                    for r in range(nr):
                        if i >= n:
                            break
                        dline = lines[i]
                        # parse fixed-width F10.2 (10 chars per value)
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

# â”€â”€ 5. Extract simulated heads at 85 well locations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def extract_sim_heads(run_dir=Path(".")):
    """
    Read modflow_GMRW.hed, average over all saved time steps,
    extract head at each well cell. Returns array length N_OBS_HEAD.
    """
    hed_path = Path(run_dir) / HED_FILE
    wells    = pd.read_csv(WELL_CSV)   # columns: well_id, head_mean_m, layer, row, clo

    frames = parse_hed_file(hed_path)
    if not frames:
        print("  [WARN] No head frames parsed from .hed file!", flush=True)
        return np.full(N_OBS_HEAD, -9999.0)

    # Stack and mean-average over time steps (ignore nan = inactive)
    stack = np.stack(frames, axis=0)   # shape: (n_frames, nrow, ncol)
    with np.errstate(all="ignore"):
        mean_head = np.nanmean(stack, axis=0)   # shape: (nrow, ncol)

    sim_heads = np.full(N_OBS_HEAD, -9999.0)
    for idx, row in wells.iterrows():
        r = int(row["row"])  - 1   # 1-based â†’ 0-based
        c = int(row["clo"])  - 1
        h = mean_head[r, c]
        sim_heads[idx] = h if np.isfinite(h) else -9999.0

    valid = np.sum(sim_heads > -9000)
    print(f"  Heads extracted: {valid}/{N_OBS_HEAD} valid  (mean={np.mean(sim_heads[sim_heads > -9000]):.2f} m)",
          flush=True)
    return sim_heads

# â”€â”€ 6. Read output.rch â€” streamflow extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ 7. Required-file validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        print(f"FATAL: _baseline_bak/modflow_GMRW.upw not found â€” run build_ies_pst_v5.py first",
              flush=True)
        sys.exit(24)
    print(f"  [OK] Pre-run check OK ({len(REQUIRED_FILES)} SWAT files + UPW backup)",
          flush=True)

# â”€â”€ 8. Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    run_dir = Path(".")
    validate_required_files(run_dir)

    params = read_params(run_dir / PAR_FILE)
    print(f"  Parameters: { {k: f'{v:.5g}' for k,v in params.items()} }", flush=True)

    apply_swat_params(params, run_dir)
    apply_modflow_params(params, run_dir)
    print("  All parameters applied.", flush=True)

    rch_size_before = (run_dir / "output.rch").stat().st_size if (run_dir / "output.rch").exists() else -1
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

    # â”€â”€ Write dummy outputs on failure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if proc.returncode != 0:
        print(f"  ERROR: exit={proc.returncode}. Last: {last_line}", flush=True)
        SIM_FLOW.write_text("\n".join(["-9999.0"] * N_OBS_FLOW) + "\n")
        SIM_HEAD.write_text("\n".join(["-9999.0"] * N_OBS_HEAD) + "\n")
        sys.exit(0)

    # â”€â”€ Extract streamflow â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    flow_vals = np.full(N_OBS_FLOW, -9999.0)   # pre-init: ensures always defined
    try:
        sim_cal = read_sim_flow(run_dir)
        n = len(sim_cal)
        if n < N_OBS_FLOW:
            sim_cal = pd.concat([sim_cal, pd.Series([-9999.0] * (N_OBS_FLOW - n))])
        elif n > N_OBS_FLOW:
            sim_cal = sim_cal.iloc[:N_OBS_FLOW]
        flow_vals = sim_cal.values
    except Exception as e:
        print(f"  ERROR reading output.rch: {str(e)[:200]}", flush=True)
        flow_vals = np.full(N_OBS_FLOW, -9999.0)

    with open(run_dir / SIM_FLOW, "w") as f:
        for v in flow_vals:
            f.write(f"{v:.6f}\n")
    print(f"  Wrote {N_OBS_FLOW} flow values to {SIM_FLOW}", flush=True)

    # â”€â”€ Extract groundwater heads â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    head_vals = np.full(N_OBS_HEAD, -9999.0)   # pre-init: ensures always defined
    try:
        head_vals = extract_sim_heads(run_dir)
    except Exception as e:
        print(f"  ERROR extracting heads: {str(e)[:200]}", flush=True)
        head_vals = np.full(N_OBS_HEAD, -9999.0)

    with open(run_dir / SIM_HEAD, "w") as f:
        for v in head_vals:
            f.write(f"{v:.4f}\n")
    print(f"  Wrote {N_OBS_HEAD} head values to {SIM_HEAD}", flush=True)

