<#
.SYNOPSIS
Bootstrap a local development environment for this repository.

.DESCRIPTION
Creates a virtual environment (default: .venv) and installs project dependencies
from pyproject.toml using editable mode with dev extras.

.NOTES
Purpose: root bootstrap entrypoint for Windows/PowerShell users.
Dependency source of truth: pyproject.toml (`.[dev]` extras).
Last reviewed: 2026-02-20

.EXAMPLE
.\setup.ps1

.EXAMPLE
.\setup.ps1 -VenvDir .venv311

.EXAMPLE
.\setup.ps1 -NoVenv
#>

param(
    [string]$VenvDir = ".venv",
    [switch]$NoVenv
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if ($NoVenv) {
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"
    Write-Host "Setup complete (no virtual environment)." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python not found at $VenvPython"
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e ".[dev]"

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
