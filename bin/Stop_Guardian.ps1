# Stop_Guardian.ps1  ? no-wmic / CIM / ASCII-only (avoid encoding issues)
[CmdletBinding()]
param()
$ErrorActionPreference = 'Continue'

Write-Host '== Stop guardian & collectors ==' -ForegroundColor Cyan

# 1) stop guardian via PID file
$pidf = 'D:\SQuant_Pro\tasks\collector_guard.pid'
if (Test-Path $pidf) {
  try {
    $guardPid = [int]((Get-Content $pidf -ErrorAction Stop)[0])
    Write-Host ("Stopping guardian PID={0} ..." -f $guardPid) -ForegroundColor Yellow
    Stop-Process -Id $guardPid -Force -ErrorAction SilentlyContinue
  } catch {
    Write-Host ("Guard PID handling error: {0}" -f $_.Exception.Message) -ForegroundColor DarkYellow
  }
  try { Remove-Item $pidf -Force -ErrorAction SilentlyContinue } catch {}
} else {
  Write-Host 'No PID file (probably not running).' -ForegroundColor DarkGray
}

# 2) kill residual child processes (collector and log-writer)
try {
  $procs = Get-CimInstance Win32_Process
  $pyTargets  = $procs | Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -match 'collector_rt\.py' }
  $cmdTargets = $procs | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'collector_task\.log' }

  foreach($p in $pyTargets)  { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {} }
  foreach($p in $cmdTargets) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {} }

  Write-Host ("Killed python={0} cmd={1}" -f $pyTargets.Count, $cmdTargets.Count) -ForegroundColor Gray
} catch {
  Write-Host ("CIM enumeration failed: {0}" -f $_.Exception.Message) -ForegroundColor DarkYellow
}

Write-Host '[CLEAN] Stop sequence finished.' -ForegroundColor Green
