# Registers a Windows scheduled task that starts the goldbot watchdog when you log on,
# and restarts it if it stops. MetaTrader 5 needs a desktop session, so the task runs in
# your user session (on a VPS: disconnect from Remote Desktop, do not sign out).
# Usage (normal PowerShell, no admin needed):
#   powershell -ExecutionPolicy Bypass -File deploy\windows\register_task.ps1 -Mode trade
param(
    [ValidateSet("alert", "trade")] [string] $Mode = "alert",
    [string] $TaskName = "goldbot"
)
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$script = Join-Path $root "deploy\windows\run_goldbot.ps1"
$arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$script`" -Mode $Mode"

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT1M"     # give the MT5 terminal time to start and log in
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "goldbot XAUUSD bot (watchdog)" -Force | Out-Null
Write-Host "Task '$TaskName' registered (mode $Mode). Start it now with:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Remove it with:  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
