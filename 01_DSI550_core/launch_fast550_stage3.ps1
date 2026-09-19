param(
    [int]$Port = 4013,
    [int]$NumWorkers = 24
)

$ErrorActionPreference = "Stop"
$pkgDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelDir = Split-Path -Parent $pkgDir
$exe = Join-Path $modelDir "pestpp-ies.exe"
$pstName = "swat_modflow_ies_dsi550_fast_stage3.pst"
$stage1Dir = Join-Path $pkgDir "stage1_run"
$stage2Dir = Join-Path $pkgDir "stage2_run"
$stage3Dir = Join-Path $pkgDir "stage3_run"
$priorParEn  = "swat_modflow_ies_dsi550_fast_stage1.0.par.csv"
$restartParEn = "swat_modflow_ies_dsi550_fast_stage2.3.par.csv"
$restartObsEn = "swat_modflow_ies_dsi550_fast_stage2.3.obs.csv"

if (-not (Test-Path $exe)) { throw "pestpp-ies.exe not found at $exe" }
if ($NumWorkers -lt 1 -or $NumWorkers -gt 24) { throw "NumWorkers must be between 1 and 24 (available worker dirs)." }
if (Test-Path $stage3Dir) { throw "Run directory already exists: $stage3Dir. Rename/remove it to avoid overwrite." }
if (-not (Test-Path (Join-Path $stage2Dir $restartParEn))) { throw "Missing Stage 2 ensemble: $stage2Dir\$restartParEn" }
if (-not (Test-Path (Join-Path $stage2Dir $restartObsEn))) { throw "Missing Stage 2 ensemble: $stage2Dir\$restartObsEn" }
if (-not (Test-Path (Join-Path $stage1Dir $priorParEn)))  { throw "Missing prior ensemble: $stage1Dir\$priorParEn" }

$running = Get-Process pestpp-ies -ErrorAction SilentlyContinue
if ($running) { throw "pestpp-ies processes are already running. Stop them before launching." }

New-Item -ItemType Directory -Path $stage3Dir | Out-Null

# Copy PST, restart ensembles, prior ensemble, and interface files
Copy-Item (Join-Path $pkgDir $pstName)            (Join-Path $stage3Dir $pstName)       -Force
Copy-Item (Join-Path $stage2Dir $restartParEn)    (Join-Path $stage3Dir $restartParEn)  -Force
Copy-Item (Join-Path $stage2Dir $restartObsEn)    (Join-Path $stage3Dir $restartObsEn)  -Force
Copy-Item (Join-Path $stage1Dir $priorParEn)      (Join-Path $stage3Dir $priorParEn)    -Force
Copy-Item (Join-Path $modelDir "pest_pars.dat.tpl") (Join-Path $stage3Dir "pest_pars.dat.tpl") -Force
Copy-Item (Join-Path $modelDir "sim_flow_fp.ins")   (Join-Path $stage3Dir "sim_flow_fp.ins")   -Force
Copy-Item (Join-Path $modelDir "sim_head_fp.ins")   (Join-Path $stage3Dir "sim_head_fp.ins")   -Force

# Ensure each worker has the Stage 3 PST
for ($i = 1; $i -le $NumWorkers; $i++) {
    $wdir = Join-Path $modelDir ("worker" + $i)
    if (-not (Test-Path $wdir)) { throw "Missing worker directory: $wdir" }
    Copy-Item (Join-Path $pkgDir $pstName) (Join-Path $wdir $pstName) -Force
}

Write-Host "Launching $NumWorkers workers on localhost:$Port ..." -ForegroundColor Cyan
for ($i = 1; $i -le $NumWorkers; $i++) {
    $wdir = Join-Path $modelDir ("worker" + $i)
    Start-Process -FilePath $exe -WorkingDirectory $wdir -ArgumentList "$pstName /h localhost:$Port" -WindowStyle Minimized
}

Start-Sleep -Seconds 3
Write-Host "Starting Stage 3 master in $stage3Dir" -ForegroundColor Cyan
Write-Host "Command: $exe $pstName /h :$Port" -ForegroundColor DarkGray

Set-Location $stage3Dir
& $exe $pstName /h :$Port

Write-Host "Stage 3 complete. Check: $stage3Dir" -ForegroundColor Green
