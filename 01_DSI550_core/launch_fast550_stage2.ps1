param(
    [int]$Port = 4012,
    [int]$NumWorkers = 8
)

$ErrorActionPreference = "Stop"
$pkgDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelDir = Split-Path -Parent $pkgDir
$exe = Join-Path $modelDir "pestpp-ies.exe"
$pstName = "swat_modflow_ies_dsi550_fast_stage2.pst"
$stage1Dir = Join-Path $pkgDir "stage1_run"
$stage2Dir = Join-Path $pkgDir "stage2_run"
$parEn = "swat_modflow_ies_dsi550_fast_stage1.1.par.csv"
$obsEn = "swat_modflow_ies_dsi550_fast_stage1.1.obs.csv"
$priorParEn = "swat_modflow_ies_dsi550_fast_stage1.0.par.csv"

if (-not (Test-Path $exe)) { throw "pestpp-ies.exe not found at $exe" }
if ($NumWorkers -ne 8) { throw "This workflow is configured for exactly 8 workers. Use -NumWorkers 8." }
if (Test-Path $stage2Dir) { throw "Run directory already exists: $stage2Dir. Rename/remove it to avoid overwrite." }
if (-not (Test-Path (Join-Path $stage1Dir $parEn))) { throw "Missing Stage 1 ensemble file: $stage1Dir\\$parEn" }
if (-not (Test-Path (Join-Path $stage1Dir $obsEn))) { throw "Missing Stage 1 ensemble file: $stage1Dir\\$obsEn" }
if (-not (Test-Path (Join-Path $stage1Dir $priorParEn))) { throw "Missing Stage 1 prior ensemble file: $stage1Dir\\$priorParEn" }

$running = Get-Process pestpp-ies -ErrorAction SilentlyContinue
if ($running) { throw "pestpp-ies processes are already running. Stop them before launching this run." }

New-Item -ItemType Directory -Path $stage2Dir | Out-Null

# Copy control, restart ensemble, and interface files into clean Stage 2 run directory
Copy-Item (Join-Path $pkgDir $pstName) (Join-Path $stage2Dir $pstName) -Force
Copy-Item (Join-Path $stage1Dir $parEn) (Join-Path $stage2Dir $parEn) -Force
Copy-Item (Join-Path $stage1Dir $obsEn) (Join-Path $stage2Dir $obsEn) -Force
Copy-Item (Join-Path $stage1Dir $priorParEn) (Join-Path $stage2Dir $priorParEn) -Force
Copy-Item (Join-Path $modelDir "pest_pars.dat.tpl") (Join-Path $stage2Dir "pest_pars.dat.tpl") -Force
Copy-Item (Join-Path $modelDir "sim_flow_fp.ins") (Join-Path $stage2Dir "sim_flow_fp.ins") -Force
Copy-Item (Join-Path $modelDir "sim_head_fp.ins") (Join-Path $stage2Dir "sim_head_fp.ins") -Force

# Ensure each worker has the Stage 2 pst file
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
Write-Host "Starting Stage 2 master in $stage2Dir" -ForegroundColor Cyan
Write-Host "Command: $exe $pstName /h :$Port" -ForegroundColor DarkGray

Set-Location $stage2Dir
& $exe $pstName /h :$Port

Write-Host "Stage 2 complete. Check: $stage2Dir" -ForegroundColor Green
