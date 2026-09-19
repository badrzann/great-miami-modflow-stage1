#!/usr/bin/env python3
"""
forward_run_broad.py
====================
Extended forward run for the broad 34-parameter prior-sensitivity ensemble.

Extends forward_run_fullperiod.py with 21 additional candidate parameters:
  Surface:  canmx_m, slsubbsn_m, ov_n_m
  ET:       epco
  Soil:     sol_awc_m, sol_k_m, sol_bd_m, ffcb
  GW:       lat_ttime, gw_spyld_m, alpha_bf_d, deepst_m, shallst_m
  Channel:  ch_n2_m, alpha_bnk
  Snow:     sftmp, smfmx, smfmn, timp
  HRU phys: hru_slp_m
  MODFLOW:  vka_mult (on top of retained hk_mult, sy, ss)

Reads:  pest_pars_broad.dat   (34 parameters, written by PEST++ from .tpl)
Writes: sim_flow_fp.dat        252 monthly flow values (2003-01 to 2023-12)
        sim_head_fp.dat        104 time-averaged groundwater heads
"""
import os, sys, re, shutil, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# __ Configuration _____________________________________________________________
EXE        = Path("SWAT-MODFLOW3.exe")
PAR_FILE   = Path("pest_pars_broad.dat")
SIM_FLOW   = Path("sim_flow_fp.dat")
SIM_HEAD   = Path("sim_head_fp.dat")
UPW_FILE   = Path("modflow_GMRW.upw")
HED_FILE   = Path("modflow_GMRW.hed")
WELL_CSV   = Path(r"D:\nasrin\obs_data\head_mean_104.csv")

RCH_NO     = 5
SIM_START  = "2003-01"
CAL_START  = "2003-01"
CAL_END    = "2023-12"
N_OBS_FLOW = 252
N_OBS_HEAD = 104
HDRY       = -999.0

# __ 1. Parameter reader _______________________________________________________
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

# __ 2. SWAT pipe-format set / multiply (|KEYWORD:) ___________________________
def set_param(text, keyword, new_val):
    pattern = (r"([ \t]*)([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)"
               r"([ \t]+\|[ \t]*" + re.escape(keyword) + r"[ \t]*:)")
    def repl(m):
        orig = m.group(2)
        dec = max(len(orig.rstrip("0").split(".")[-1]), 4) if "." in orig else 4
        return m.group(1) + f"{new_val:.{dec}f}".rjust(len(orig)) + m.group(3)
    return re.subn(pattern, repl, text, flags=re.IGNORECASE)

def multiply_param(text, keyword, mult):
    pattern = (r"([ \t]*)([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)"
               r"([ \t]+\|[ \t]*" + re.escape(keyword) + r"[ \t]*:)")
    def repl(m):
        val = float(m.group(2)) * mult
        if keyword.upper() == "CN2":
            val = max(35.0, min(98.0, val))
        orig = m.group(2)
        dec = max(len(orig.rstrip("0").split(".")[-1]), 2) if "." in orig else 4
        return m.group(1) + f"{val:.{dec}f}".rjust(len(orig)) + m.group(3)
    return re.subn(pattern, repl, text, flags=re.IGNORECASE)

# __ 3. ArcSWAT 2012 .sol array-line multiplier ________________________________
def multiply_sol_line(text, label_substr, mult):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if label_substr.lower() in line.lower() and ":" in line:
            colon_pos = line.index(":")
            prefix = line[:colon_pos + 1]
            values_str = line[colon_pos + 1:]
            def repl_num(m):
                val = float(m.group(0)) * mult
                return f"{val:.5f}".rjust(len(m.group(0)))
            lines[i] = prefix + re.sub(r"\d+\.\d+", repl_num, values_str)
    return "\n".join(lines)

# __ 4. Apply all 34 SWAT parameters ___________________________________________
def apply_params_broad(params, run_dir=Path(".")):
    run_dir = Path(run_dir)
    bak_dir = run_dir / "_baseline_bak"

    for ext in [".mgt", ".gw", ".hru", ".rte", ".sol"]:
        for bak in bak_dir.glob(f"*{ext}"):
            shutil.copy2(bak, run_dir / bak.name)
    shutil.copy2(bak_dir / "basins.bsn", run_dir / "basins.bsn")

    # .gw
    gw_set  = {"ALPHA_BF": params.get("alpha_bf"), "GW_DELAY": params.get("gw_delay"),
                "GWQMN": params.get("gwqmn"), "GW_REVAP": params.get("gw_revap"),
                "RCHRG_DP": params.get("rchrg_dp"), "REVAPMN": params.get("revapmn"),
                "ALPHA_BF_D": params.get("alpha_bf_d")}
    gw_mult = {"GW_SPYLD": params.get("gw_spyld_m"), "DEEPST": params.get("deepst_m"),
               "SHALLST": params.get("shallst_m")}
    for fp in run_dir.glob("*.gw"):
        text = fp.read_text(encoding="latin-1"); changed = False
        for kw, val in gw_set.items():
            if val is not None:
                text, n = set_param(text, kw, val); changed = changed or bool(n)
        for kw, val in gw_mult.items():
            if val is not None:
                text, n = multiply_param(text, kw, val); changed = changed or bool(n)
        if changed: fp.write_text(text, encoding="latin-1")

    # .hru
    hru_set  = {"ESCO": params.get("esco"), "EPCO": params.get("epco"),
                "LAT_TTIME": params.get("lat_ttime")}
    hru_mult = {"CANMX": params.get("canmx_m"), "SLSUBBSN": params.get("slsubbsn_m"),
                "OV_N": params.get("ov_n_m"), "HRU_SLP": params.get("hru_slp_m")}
    for fp in run_dir.glob("*.hru"):
        if fp.name == "output.hru": continue
        text = fp.read_text(encoding="latin-1"); changed = False
        for kw, val in hru_set.items():
            if val is not None:
                text, n = set_param(text, kw, val); changed = changed or bool(n)
        for kw, val in hru_mult.items():
            if val is not None:
                text, n = multiply_param(text, kw, val); changed = changed or bool(n)
        if changed: fp.write_text(text, encoding="latin-1")

    # .mgt
    cn2_m = params.get("cn2_m")
    if cn2_m is not None:
        for fp in run_dir.glob("*.mgt"):
            text = fp.read_text(encoding="latin-1")
            text, n = multiply_param(text, "CN2", cn2_m)
            if n: fp.write_text(text, encoding="latin-1")

    # .sol (ArcSWAT 2012 array-line format)
    sol_mults = [("Ave. AW Incl. Rock Frag", params.get("sol_awc_m")),
                 ("Ksat. (est.)",             params.get("sol_k_m")),
                 ("Bulk Density Moist",        params.get("sol_bd_m"))]
    for fp in run_dir.glob("*.sol"):
        text = fp.read_text(encoding="latin-1"); changed = False
        for label, val in sol_mults:
            if val is not None:
                new_text = multiply_sol_line(text, label, val)
                if new_text != text:
                    text = new_text; changed = True
        if changed: fp.write_text(text, encoding="latin-1")

    # .rte
    rte_mult = {"CH_K2": params.get("ch_k2_m"), "CH_N2": params.get("ch_n2_m")}
    rte_set  = {"ALPHA_BNK": params.get("alpha_bnk")}
    for fp in run_dir.glob("*.rte"):
        text = fp.read_text(encoding="latin-1"); changed = False
        for kw, val in rte_mult.items():
            if val is not None:
                text, n = multiply_param(text, kw, val); changed = changed or bool(n)
        for kw, val in rte_set.items():
            if val is not None:
                text, n = set_param(text, kw, val); changed = changed or bool(n)
        if changed: fp.write_text(text, encoding="latin-1")

    # basins.bsn
    bsn_set = {"SURLAG": params.get("surlag"), "SFTMP": params.get("sftmp"),
               "SMFMX": params.get("smfmx"),  "SMFMN": params.get("smfmn"),
               "TIMP":  params.get("timp"),    "FFCB":  params.get("ffcb")}
    bsn_path = run_dir / "basins.bsn"
    text = bsn_path.read_text(encoding="latin-1"); changed = False
    for kw, val in bsn_set.items():
        if val is not None:
            text, n = set_param(text, kw, val); changed = changed or bool(n)
    if changed: bsn_path.write_text(text, encoding="latin-1")

# __ 5. Apply MODFLOW UPW parameters ___________________________________________
def apply_modflow_params(params, run_dir=Path(".")):
    run_dir  = Path(run_dir)
    upw_path = run_dir / UPW_FILE
    shutil.copy2(run_dir / "_baseline_bak" / "modflow_GMRW.upw", upw_path)

    hk_mult  = params.get("hk_mult",  1.0)
    vka_mult = params.get("vka_mult", 1.0)
    sy_val   = params.get("sy",  None)
    ss_val   = params.get("ss",  None)

    lines  = upw_path.read_text(encoding="latin-1").splitlines(keepends=True)
    out    = []
    in_hk  = False
    in_vka = False
    hk_seen = False

    for line in lines:
        if not hk_seen and re.search(r"INTERNAL.*HK\b", line, re.IGNORECASE):
            hk_seen = True; in_hk = True; out.append(line); continue
        if re.search(r"INTERNAL.*VKA\b", line, re.IGNORECASE):
            in_hk = False; in_vka = True; out.append(line); continue
        if in_vka and re.search(r"\b(SS|SY)\b", line, re.IGNORECASE):
            in_vka = False
        if in_hk and re.search(r"\b(VKA|SS|SY)\b", line, re.IGNORECASE):
            in_hk = False

        if in_hk and abs(hk_mult - 1.0) > 1e-9:
            toks = [str(round(float(t)*hk_mult,4)) if t.replace(".","",1).replace("-","",1).replace("e","",1).replace("+","",1).replace("E","",1).isdigit() or "." in t else t for t in line.split()]
            line = " ".join(toks) + "\n"; out.append(line); continue
        if in_vka and abs(vka_mult - 1.0) > 1e-9:
            toks = [str(round(float(t)*vka_mult,4)) if t.replace(".","",1).replace("-","",1).replace("e","",1).replace("+","",1).replace("E","",1).isdigit() or "." in t else t for t in line.split()]
            line = " ".join(toks) + "\n"; out.append(line); continue

        if ss_val is not None and re.match(r"\s*CONSTANT\s+[\d.eE+\-]+\s+SS\b", line, re.IGNORECASE):
            line = f"CONSTANT {ss_val:.8g}            SS (specific storage)\n"
        if sy_val is not None and re.match(r"\s*CONSTANT\s+[\d.eE+\-]+\s+SY\b", line, re.IGNORECASE):
            line = f"CONSTANT {sy_val:.8g}              SY (specific yield)\n"
        out.append(line)

    upw_path.write_text("".join(out), encoding="latin-1")
    print(f"  UPW: hk_mult={hk_mult:.4f} vka_mult={vka_mult:.4f} sy={sy_val} ss={ss_val}", flush=True)

# __ 6. Parse MODFLOW .hed file ________________________________________________
def parse_hed_file(hed_path):
    frames = []
    lines  = Path(hed_path).read_text(encoding="latin-1").splitlines()
    n = len(lines); i = 0
    while i < n:
        parts = lines[i].split()
        if len(parts) >= 7:
            try:
                float(parts[0]); float(parts[2])
                stripped = [p for p in parts if not p.startswith("(")]
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
                    frames.append(arr); continue
            except (ValueError, IndexError):
                pass
        i += 1
    return frames

# __ 7. Extract simulated heads ________________________________________________
def extract_sim_heads(run_dir=Path(".")):
    wells  = pd.read_csv(WELL_CSV)
    frames = parse_hed_file(Path(run_dir) / HED_FILE)
    if not frames:
        print("  [WARN] No head frames parsed!", flush=True)
        return np.full(N_OBS_HEAD, -9999.0)
    with np.errstate(all="ignore"):
        mean_head = np.nanmean(np.stack(frames, axis=0), axis=0)
    sim_heads = np.full(N_OBS_HEAD, -9999.0)
    for idx, row in wells.iterrows():
        h = mean_head[int(row["row"])-1, int(row["clo"])-1]
        sim_heads[idx] = h if np.isfinite(h) else -9999.0
    valid = np.sum(sim_heads > -9000)
    print(f"  Heads: {valid}/{N_OBS_HEAD} valid (mean={np.mean(sim_heads[sim_heads>-9000]):.2f} m)", flush=True)
    return sim_heads

# __ 8. Read output.rch ________________________________________________________
def read_sim_flow(run_dir=Path(".")):
    df = pd.read_csv(Path(run_dir)/"output.rch", skiprows=9, sep=r"\s+", header=None,
                     usecols=[0,1,3,6], names=["LABEL","RCH","MON","FLOW_OUTcms"],
                     encoding="latin-1", on_bad_lines="skip")
    for col in ["RCH","MON","FLOW_OUTcms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rch5 = df[(df["RCH"]==RCH_NO) & (df["MON"].between(1,12))].reset_index(drop=True)
    rch5.index = pd.date_range(SIM_START, periods=len(rch5), freq="MS")
    print(f"  output.rch: {len(rch5)} months for RCH {RCH_NO}", flush=True)
    return rch5["FLOW_OUTcms"].rename("sim_cms").loc[CAL_START:CAL_END]

# __ 9. Validate required files ________________________________________________
REQUIRED_FILES = ["file.cio","basins.bsn","plant.dat","till.dat",
                  "urban.dat","septwq.dat","pest.dat","fert.dat"]

def validate_required_files(run_dir):
    missing = [f for f in REQUIRED_FILES if not (run_dir/f).exists()]
    empty   = [f for f in REQUIRED_FILES if (run_dir/f).exists() and (run_dir/f).stat().st_size==0]
    if missing or empty:
        print(f"FATAL: Missing={missing}  Empty={empty}", flush=True); sys.exit(24)
    if not (run_dir/"_baseline_bak"/"modflow_GMRW.upw").exists():
        print("FATAL: _baseline_bak/modflow_GMRW.upw not found!", flush=True); sys.exit(24)
    print(f"  [OK] Pre-run check: {len(REQUIRED_FILES)} SWAT files + UPW backup", flush=True)

# __ 10. Main __________________________________________________________________
if __name__ == "__main__":
    try:
        run_dir = Path(".")
        validate_required_files(run_dir)

        params = read_params(run_dir / PAR_FILE)
        print(f"  Parameters ({len(params)}): "
              f"{ {k: f'{v:.4g}' for k,v in list(params.items())[:6]} }...", flush=True)

        apply_params_broad(params, run_dir)
        apply_modflow_params(params, run_dir)
        print("  All 34 parameters applied.", flush=True)

        rch_size_before = (run_dir/"output.rch").stat().st_size \
            if (run_dir/"output.rch").exists() else -1
        t_start = time.time()
        print(f"  Running SWAT-MODFLOW3.exe  [{time.strftime('%H:%M:%S')}]", flush=True)

        proc = subprocess.Popen([str(run_dir/EXE)], cwd=str(run_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        last_line = ""
        try:
            for line in proc.stdout:
                s = line.rstrip()
                if s: last_line = s
        except Exception as e:
            print(f"  [WARN] {e}", flush=True)
        proc.wait()
        elapsed = time.time() - t_start
        print(f"  Done in {elapsed:.1f}s (exit={proc.returncode}) [{time.strftime('%H:%M:%S')}]", flush=True)

        if (run_dir/"output.rch").exists():
            delta = (run_dir/"output.rch").stat().st_size - rch_size_before
            print(f"  output.rch delta: {delta:+,} bytes", flush=True)

        if proc.returncode != 0:
            print(f"  ERROR exit={proc.returncode}. Last: {last_line}", flush=True)
            SIM_FLOW.write_text("\n".join(["-9999.0"]*N_OBS_FLOW)+"\n")
            SIM_HEAD.write_text("\n".join(["-9999.0"]*N_OBS_HEAD)+"\n")
            sys.exit(0)

        # Streamflow
        flow_vals = np.full(N_OBS_FLOW, -9999.0)
        try:
            sim_fp = read_sim_flow(run_dir)
            n = len(sim_fp)
            if   n < N_OBS_FLOW: flow_vals = np.array(list(sim_fp.values)+[-9999.0]*(N_OBS_FLOW-n))
            elif n > N_OBS_FLOW: flow_vals = sim_fp.values[:N_OBS_FLOW]
            else:                 flow_vals = sim_fp.values
        except Exception as e:
            print(f"  ERROR rch: {e}", flush=True)
        SIM_FLOW.write_text("\n".join(f"{v:.6f}" for v in flow_vals)+"\n")
        vf = flow_vals[flow_vals>-9000]
        if len(vf): print(f"  sim_flow_fp.dat: {N_OBS_FLOW} vals ({vf.min():.1f}-{vf.max():.1f} m3/s)", flush=True)

        # Heads
        head_vals = np.full(N_OBS_HEAD, -9999.0)
        try: head_vals = extract_sim_heads(run_dir)
        except Exception as e: print(f"  ERROR hed: {e}", flush=True)
        SIM_HEAD.write_text("\n".join(f"{v:.4f}" for v in head_vals)+"\n")
        print(f"  sim_head_fp.dat: {N_OBS_HEAD} values written.", flush=True)
        print("  [DONE] forward_run_broad.py complete.", flush=True)

    except SystemExit:
        raise
    except Exception as _fatal:
        print(f"  FATAL: {_fatal}", flush=True)
        try:
            SIM_FLOW.write_text("\n".join(["-9999.0"]*N_OBS_FLOW)+"\n")
            SIM_HEAD.write_text("\n".join(["-9999.0"]*N_OBS_HEAD)+"\n")
        except Exception:
            pass
    sys.exit(0)