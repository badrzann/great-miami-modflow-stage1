# restart_prior100.ps1 — Kill old master and restart PESTPP-IES prior100
# Run from D:\GMRW\swatmf_run

# ── 1. Kill any existing PESTPP-IES processes ──
Write-Host "Stopping old PESTPP-IES processes..."
Get-Process pestpp-ies -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

# ── 2. Clean up old run files so master starts fresh ──
$master = "D:\GMRW\swatmf_run"
Remove-Item "$master\prior100.rmr" -ErrorAction SilentlyContinue
Remove-Item "$master\prior100.rec" -ErrorAction SilentlyContinue
Remove-Item "$master\phi.actual.csv" -ErrorAction SilentlyContinue
Remove-Item "$master\swat_modflow_ies_prior100.0.obs.csv" -ErrorAction SilentlyContinue

# ── 3. Start master ──
Write-Host "Starting PESTPP-IES master on port 4018..."
$masterProc = Start-Process -FilePath "$master\pestpp-ies.exe" `
    -ArgumentList "swat_modflow_ies_prior100.pst /h :4018" `
    -WorkingDirectory $master `
    -PassThru -WindowStyle Normal
Write-Host "  Master PID: $($masterProc.Id)"
Start-Sleep -Seconds 5

# ── 4. Start workers (real_* and worker_* directories) ──
$realWorkers = Get-ChildItem "D:\GMRW\swatmf_run\fp_workers\real_*" -Directory | Sort-Object Name
$wkWorkers   = Get-ChildItem "D:\GMRW\swatmf_run\fp_workers\worker_*" -Directory | Sort-Object Name
$workers = @($realWorkers) + @($wkWorkers)
Write-Host "Starting $($workers.Count) workers..."
foreach ($w in $workers) {
    $wp = Start-Process -FilePath "$master\pestpp-ies.exe" `
        -ArgumentList "..\..\swat_modflow_ies_prior100.pst /h localhost:4018" `
        -WorkingDirectory $w.FullName `
        -PassThru -WindowStyle Minimized
    Write-Host "  Worker $($w.Name) PID=$($wp.Id)"
    Start-Sleep -Milliseconds 500
}

Write-Host "`nAll processes started. Monitor progress with:"
Write-Host "  Get-Content D:\GMRW\swatmf_run\prior100.rmr -Tail 20"
Write-Host "  Get-Content D:\GMRW\swatmf_run\phi.actual.csv"
