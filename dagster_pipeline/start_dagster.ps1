# Starts the Dagster UI with a fixed DAGSTER_HOME, so run history stays in .dagster_home
# and `dagster dev` does not leave a new .tmp_dagster_home_* folder in the repo on every start.
$root = Split-Path $PSScriptRoot -Parent
$env:DAGSTER_HOME = Join-Path $root ".dagster_home"
New-Item -ItemType Directory -Force $env:DAGSTER_HOME | Out-Null
Set-Location $root
$venvDagster = Join-Path $root "venv\Scripts\dagster.exe"
if (Test-Path $venvDagster) { & $venvDagster dev @args } else { & dagster dev @args }
