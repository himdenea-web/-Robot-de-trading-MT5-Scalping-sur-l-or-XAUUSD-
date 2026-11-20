# Laptop only: keep the machine awake on mains power so the bot never sleeps mid-trade.
# Run once in an elevated PowerShell (Run as administrator).
#   - never sleep or hibernate on AC power
#   - closing the lid does nothing on AC power
# Battery settings are left unchanged. Undo with:  powercfg /restoredefaultschemes
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
Write-Host "Done: no sleep, no hibernate and lid close ignored while plugged in."
Write-Host "Also set Windows Update active hours to cover 03:00-17:00 New York time, and keep the clock synced (Settings > Time > Sync now)."
