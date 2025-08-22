param(
  [string]$DB        = "D:\quant_system_v2\data\market_data.db",
  [string]$Symbols   = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,APTUSDT",
  [string]$Intervals = "5m,15m,30m,1h,2h,4h,1d",
  [int]$Readers      = 24,
  [int]$HttpParallel = 8
)
$ErrorActionPreference = "Stop"

$ROOT = "D:\SQuant_Pro"
$LOG  = Join-Path $ROOT "logs"
$TASK = Join-Path $ROOT "tasks"
$PIDF = Join-Path $TASK "collector_guard.pid"
$COL  = "D:\SQuant_Pro\bin\collector_rt.py"

# ?? python
$PY = "python"
if (Test-Path "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe") {
  $PY = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe"
}
New-Item -ItemType Directory -Force $LOG,$TASK | Out-Null

# ?????????????
if (Test-Path $PIDF) {
  try {
    $old = [int](Get-Content $PIDF -ErrorAction SilentlyContinue)[0]
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
      Write-Host "[GUARD] ???? (PID=$old)" -ForegroundColor Yellow
      exit 0
    }
  } catch {}
}
Set-Content -Path $PIDF -Encoding ASCII -Value $PID

# ??????? collector_task.log ?? cmd/python
Get-WmiObject Win32_Process -Filter "name='cmd.exe'"    | ? { $_.CommandLine -match 'collector_task\.log' } | % { try { Stop-Process -Id $_.ProcessId -Force } catch {} }
Get-WmiObject Win32_Process -Filter "name='python.exe'" | ? { $_.CommandLine -match 'collector_rt\.py'   } | % { try { Stop-Process -Id $_.ProcessId -Force } catch {} }

if (!(Test-Path $COL)) { Write-Host "[ERR] ?? collector_rt.py?" $COL -ForegroundColor Red; exit 1 }
if (!(Test-Path $DB))  { Write-Host "[ERR] ??????" $DB -ForegroundColor Red; exit 1 }

function Open-LogWriter([string]$basePath) {
  # ????????????????????????
  $target = $basePath
  try {
    $fs = New-Object System.IO.FileStream($target, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
  } catch {
    $ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
    $target = ($basePath -replace '\.log$','') + "_$ts.log"
    $fs = New-Object System.IO.FileStream($target, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
  }
  $sw = New-Object System.IO.StreamWriter($fs)
  $sw.AutoFlush = $true
  return @{ Writer = $sw; Path = $target }
}

Write-Host "[GUARD] ??????..." -ForegroundColor Cyan
while ($true) {
  # ???????????????????
  $argStr = @(
    "-u", '"' + $COL + '"',
    "--db", '"' + $DB + '"',
    "--symbols", '"' + $Symbols + '"',
    "--intervals", '"' + $Intervals + '"',
    "--max-readers", $Readers,
    "--http-parallel", $HttpParallel,
    "--bootstrap-days", 0,
    "--loop", 300
  ) -join " "

  # ??????
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $PY
  $psi.Arguments = $argStr
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError  = $true
  $psi.CreateNoWindow = $true

  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $psi
  [void]$p.Start()
  Write-Host ("[GUARD] collector started, PID={0}" -f $p.Id) -ForegroundColor Green

  # ???????/???
  $base = Join-Path $LOG "collector_task.log"
  $log  = Open-LogWriter $base
  $sw   = $log.Writer
  Write-Host ("[GUARD] logging -> {0}" -f $log.Path) -ForegroundColor DarkGray

  try {
    while (-not $p.HasExited) {
      if (-not $p.StandardOutput.EndOfStream) { $sw.WriteLine($p.StandardOutput.ReadLine()) }
      if (-not $p.StandardError.EndOfStream)  { $sw.WriteLine($p.StandardError.ReadLine())  }
      Start-Sleep -Milliseconds 200
    }
    $sw.WriteLine(("[exit] code {0} at {1:yyyy-MM-dd HH:mm:ss}" -f $p.ExitCode, (Get-Date)))
  } finally {
    if ($sw) { try { $sw.Flush(); $sw.Close(); $sw.Dispose() } catch {} }
  }

  Write-Host ("[GUARD] collector exited (code {0}). 10s ???..." -f $p.ExitCode) -ForegroundColor Yellow
  Start-Sleep 10
}
