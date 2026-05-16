#!/usr/bin/env pwsh
# Start the static dashboard server (PowerShell helper)
Set-Location -Path "$PSScriptRoot\frontend"
python -m http.server 5500
