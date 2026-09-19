param(
    [int]$Port = 4011,
    [int]$NumWorkers = 8
)

$ErrorActionPreference = "Stop"
$pkgDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelDir = Split-Path -Parent $pkgDir
$exe = Join-Path $modelDir "pestpp-ies.exe"
$pstName = "swat_modflow_ies_dsi550_fast_stage1.pst"
$runDir = Join-Path $pkgDir "stage1_run"

if (-not (Test-Path $exe)) { throw "pestpp-ies.exe not found at $exe" }
if ($NumWorkers -ne 8) { throw "This workflow is configured for exactly 8 workers. Use -NumWorkers 8." }
if (Test-Path $runDir) { throw "Run directory already exists: $runDir. Rename/remove it to avoid overwrite." }

$running = Get-Process pestpp-ies -ErrorAction SilentlyContinue
if ($running) { throw "pestpp-ies processes are already running. Stop them before launching this run." }

New-Item -ItemType Directory -Path $runDir | Out-Null

# Copy control and interface files to a clean master run directory
Copy-Item (Join-Path $pkgDir $pstName) (Join-Path $runDir $pstName) -Force
Copy-Item (Join-Path $modelDir "pest_pars.dat.tpl") (Join-Path $runDir "pest_pars.dat.tpl") -Force
Copy-Item (Join-Path $modelDir "sim_flow_fp.ins") (Join-Path $runDir "sim_flow_fp.ins") -Force
Copy-Item (Join-Path $modelDir "sim_head_fp.ins") (Join-Path $runDir "sim_head_fp.ins") -Force

# Ensure each worker has the Stage 1 pst file
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
Write-Host "Starting Stage 1 master in $runDir" -ForegroundColor Cyan
Write-Host "Command: $exe $pstName /h :$Port" -ForegroundColor DarkGray

Set-Location $runDir
& $exe $pstName /h :$Port

Write-Host "Stage 1 complete. Check: $runDir" -ForegroundColor Green
