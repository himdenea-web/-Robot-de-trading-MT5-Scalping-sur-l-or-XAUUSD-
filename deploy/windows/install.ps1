# One-time setup: Python virtual environment and dependencies.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not installed or not on PATH. Install Python 3.11+ from python.org (tick 'Add to PATH')."
}
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example: fill in the MT5 login, Telegram and phase settings."
}
New-Item -ItemType Directory -Force logs | Out-Null
Write-Host "Done. Next: edit .env, then run:  .\.venv\Scripts\python.exe -m goldbot --check"
