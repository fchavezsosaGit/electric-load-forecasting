<#
.SYNOPSIS
Bootstrap a local development environment for this repository.

.DESCRIPTION
Creates a virtual environment (default: .venv) and installs project dependencies
from pyproject.toml (runtime + dev groups) through scripts/bootstrap_env.py.

.NOTES
Purpose: root bootstrap entrypoint for Windows/PowerShell users.
Dependency source of truth: pyproject.toml via scripts/bootstrap_env.py.
Last reviewed: 2026-03-06

.EXAMPLE
.\setup.ps1

.EXAMPLE
.\setup.ps1 -VenvDir .venv311

.EXAMPLE
.\setup.ps1 -NoVenv
#>

param(
    [string]$VenvDir = ".venv",
    [switch]$NoVenv,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
$BootstrapScript = "scripts/bootstrap_env.py"

& $PythonExe --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Python executable not found: $PythonExe"
}

if ($NoVenv) {
    & $PythonExe $BootstrapScript --install --with-dev --check
    if ($LASTEXITCODE -ne 0) { throw "Dependency bootstrap failed (no-venv mode)." }
    Write-Host "Setup complete (no virtual environment)." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $VenvDir)) {
    & $PythonExe -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python not found at $VenvPython"
}

& $VenvPython $BootstrapScript --install --with-dev --check
if ($LASTEXITCODE -ne 0) { throw "Dependency bootstrap failed in virtual environment." }

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
