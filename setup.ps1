<#
.SYNOPSIS
Bootstrap a local development environment for this repository.

.DESCRIPTION
Creates a virtual environment (default: .venv) and installs project dependencies
from pyproject.toml (runtime + dev groups).

.NOTES
Purpose: root bootstrap entrypoint for Windows/PowerShell users.
Dependency source of truth: pyproject.toml (`project.dependencies` + `project.optional-dependencies.dev`).
Last reviewed: 2026-03-04

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
$InstallDepsScript = @"
import pathlib
import subprocess
import sys
import tomllib

cfg = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))
project = cfg.get('project', {})
deps = list(project.get('dependencies', []))
deps.extend(project.get('optional-dependencies', {}).get('dev', []))
if not deps:
    raise RuntimeError('No dependencies declared in pyproject.toml')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *deps])
"@
$DependencySmokeScript = "import importlib.util; modules=['numpy','pandas','matplotlib','seaborn','sklearn','statsmodels','jupyter','fastparquet']; missing=[m for m in modules if importlib.util.find_spec(m) is None]; assert not missing, f'Missing dependencies after install: {missing}'; assert importlib.util.find_spec('pyarrow') or importlib.util.find_spec('fastparquet'), 'Missing parquet backend (pyarrow or fastparquet)'"

if ($NoVenv) {
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (no-venv mode)." }
    python -c $InstallDepsScript
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (no-venv mode)." }
    python -c $DependencySmokeScript
    if ($LASTEXITCODE -ne 0) { throw "Dependency smoke check failed (no-venv mode)." }
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
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed in virtual environment." }
& $VenvPython -c $InstallDepsScript
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed in virtual environment." }
& $VenvPython -c $DependencySmokeScript
if ($LASTEXITCODE -ne 0) { throw "Dependency smoke check failed in virtual environment." }

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
