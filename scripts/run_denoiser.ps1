#!/usr/bin/env pwsh

# Activate the virtual environment
$envPath = Join-Path $PSScriptRoot "noise\Scripts\activate.ps1"
if (Test-Path $envPath) {
    . $envPath
} else {
    Write-Error "Virtual environment not found at: $envPath"
    exit 1
}

# Run the denoiser script with the virtual environment's Python
$denoiserPath = Join-Path $PSScriptRoot "audio_denoiser.py"
if (Test-Path $denoiserPath) {
    python $denoiserPath
} else {
    Write-Error "Denoiser script not found at: $denoiserPath"
    exit 1
}
