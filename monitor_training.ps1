# Monitor SRGAN Training Progress
# Usage: .\monitor_training.ps1

Write-Host "========================================" -ForegroundColor Green
Write-Host "SRGAN Training Monitor" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Find latest log file
$latestLog = Get-ChildItem "logs\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $latestLog) {
    Write-Host "No log file found. Training may not have started yet." -ForegroundColor Yellow
    exit
}

Write-Host "Monitoring: $($latestLog.Name)" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""

# Monitor with tail
Get-Content $latestLog.FullName -Wait -Tail 20
