$ErrorActionPreference = 'Stop'
# WSL can launch PowerShell without the usual Windows PATHEXT variable.
if (-not (($env:PATHEXT -split ';') -contains '.EXE')) {
    $env:PATHEXT = "$env:PATHEXT;.EXE;.COM;.BAT;.CMD"
}
Set-Location $PSScriptRoot
$hwPython = Join-Path $PSScriptRoot '.venv-win\Scripts\python.exe'

if (-not (Test-Path $hwPython)) {
    $candidates = @()
    foreach ($name in @('python3.12', 'python3.11', 'python')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $candidates += $command.Source }
    }
    $candidates += Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\python3.12.exe'
    $basePython = $null
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path $candidate)) { continue }
        & $candidate -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)'
        if ($LASTEXITCODE -eq 0) { $basePython = $candidate; break }
    }
    if (-not $basePython) { throw 'Install Python 3.11 or 3.12 and run this script again.' }
    & $basePython -m venv .venv-win
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Windows environment.' }
    & $hwPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the project requirements.' }
}

& $hwPython main.py @args
exit $LASTEXITCODE
