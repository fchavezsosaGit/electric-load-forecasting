<#
.SYNOPSIS
Run the repository end-to-end verification workflow.

.DESCRIPTION
Executes `scripts/run_e2e.py` from the repository root using the selected
Python interpreter.

.EXAMPLE
.\run_e2e.ps1 --mode full

.EXAMPLE
.\run_e2e.ps1 -PythonExe py -- --mode quick
#>

param(
    [string]$PythonExe = "python",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

& $PythonExe --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Python executable not found: $PythonExe"
}

& $PythonExe scripts/run_e2e.py @RemainingArgs
exit $LASTEXITCODE
