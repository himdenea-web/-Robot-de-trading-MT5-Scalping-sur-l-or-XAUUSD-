# Watchdog: runs goldbot and restarts it if it exits with an error.
# A clean stop (STOP file, or Ctrl+C) ends the watchdog too.
# Usage:  powershell -ExecutionPolicy Bypass -File deploy\windows\run_goldbot.ps1 [-Mode trade]
param(
    [ValidateSet("alert", "trade")] [string] $Mode = "",
    [int] $RestartDelaySeconds = 30
)
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
New-Item -ItemType Directory -Force logs | Out-Null
$watchLog = Join-Path $root "logs\watchdog.log"

function Write-Watch($msg) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $msg
    Add-Content -Path $watchLog -Value $line
    Write-Host $line
}

$botArgs = @("-m", "goldbot")
if ($Mode) { $botArgs += @("--mode", $Mode) }

while ($true) {
    Write-Watch "starting goldbot $($botArgs -join ' ')"
    & $python @botArgs
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Watch "goldbot stopped cleanly (exit 0); watchdog ends"
        break
    }
    if ($code -eq 2 -or $code -eq 3) {
        Write-Watch "goldbot refused to start (exit $code: configuration problem or already running); watchdog ends"
        break
    }
    Write-Watch "goldbot exited with code $code; restarting in $RestartDelaySeconds s"
    Start-Sleep -Seconds $RestartDelaySeconds
}
