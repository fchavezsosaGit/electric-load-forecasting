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
    [string]$PythonExe = "",
    [switch]$WithAcceleration
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
$BootstrapScript = "scripts/bootstrap_env.py"

function Invoke-CommandParts {
    param(
        [string[]]$Command,
        [string[]]$Arguments
    )
    $prefix = @()
    if ($Command.Count -gt 1) {
        $prefix = $Command[1..($Command.Count - 1)]
    }
    & $Command[0] @($prefix + $Arguments)
}

function Test-CommandParts {
    param([string[]]$Command)
    try {
        Invoke-CommandParts -Command $Command -Arguments @("--version") *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Resolve-PythonCommand {
    param([string]$Requested)
    if ($Requested) {
        return @($Requested)
    }

    $candidates = @(
        @("py", "-3.13"),
        @("py", "-3.12"),
        @("python")
    )
    foreach ($candidate in $candidates) {
        if (Test-CommandParts -Command $candidate) {
            return $candidate
        }
    }
    throw "Python executable not found. Pass -PythonExe or install Python 3.12/3.13."
}

function Test-VenvHealthy {
    param([string]$VenvPythonPath)
    if (-not (Test-Path $VenvPythonPath)) {
        return $false
    }
    try {
        & $VenvPythonPath -c "import sys; print(sys.executable)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

$PythonCommand = Resolve-PythonCommand -Requested $PythonExe
$BootstrapArgs = @("--install", "--with-dev", "--check")
if ($WithAcceleration) {
    $BootstrapArgs += "--with-acceleration"
}

if ($NoVenv) {
    Invoke-CommandParts -Command $PythonCommand -Arguments (@($BootstrapScript) + $BootstrapArgs)
    if ($LASTEXITCODE -ne 0) { throw "Dependency bootstrap failed (no-venv mode)." }
    Write-Host "Setup complete (no virtual environment)." -ForegroundColor Green
    exit 0
}

if ((Test-Path $VenvDir) -and -not (Test-VenvHealthy -VenvPythonPath (Join-Path $VenvDir "Scripts/python.exe"))) {
    Write-Host "Existing virtual environment is not runnable on this machine. Recreating $VenvDir..." -ForegroundColor Yellow
    Remove-Item $VenvDir -Recurse -Force
}

if (-not (Test-Path $VenvDir)) {
    Invoke-CommandParts -Command $PythonCommand -Arguments @("-m", "venv", $VenvDir)
}

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python not found at $VenvPython"
}

& $VenvPython $BootstrapScript @BootstrapArgs
if ($LASTEXITCODE -ne 0) { throw "Dependency bootstrap failed in virtual environment." }

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
