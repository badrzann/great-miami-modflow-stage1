# Launch broad prior-sensitivity ensemble (eval-only, 500 realizations)
# Run from: D:\nasrin\swatmf_run\fast550_package\broad_gsa\
# Step 1: Copy worker setup (if needed)
#   Each worker must have: forward_run_broad.py, pest_pars_broad.dat.tpl,
#   all SWAT model files, SWAT-MODFLOW3.exe, pestpp-ies.exe

# Start master
Start-Process pestpp-ies.exe -ArgumentList "swat_modflow_broad_gsa.pst /h :4022" -NoNewWindow

# Workers (run from each worker directory)
# Foreach ($w in 1..24) {
#   $dir = "D:\nasrin\swatmf_run\worker$w"
#   Start-Process pestpp-ies.exe -ArgumentList "swat_modflow_broad_gsa.pst /h localhost:4022" -WorkingDirectory $dir -NoNewWindow
# }
Write-Host "Broad GSA ensemble launch script generated."
Write-Host "Estimated runtime: ~100 minutes with 24 workers"
