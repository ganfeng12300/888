param([string]$PythonHint='')

# === Start_Guardian.ps1 (ASCII) ===
$ErrorActionPreference='Stop'
$ROOT='D:\SQuant_Pro'
$BIN = Join-Path $ROOT 'bin'
$LOGDIR = Join-Path $ROOT 'logs'
$TASKS = Join-Path $ROOT 'tasks'
$PIDFILE = Join-Path $TASKS 'collector_guard.pid'

if(-not(Test-Path $LOGDIR)){ New-Item -ItemType Directory -Path $LOGDIR -Force|Out-Null }
if(-not(Test-Path $TASKS)){ New-Item -ItemType Directory -Path $TASKS -Force|Out-Null }

# ---- Python?? ----
$PY=$PythonHint
if(-not $PY -or -not(Test-Path $PY)){
  $cands=@(
    "$env:LocalAppData\Programs\Python\Python39\python.exe",
    "$env:LocalAppData\Programs\Python\Python310\python.exe",
    "$env:LocalAppData\Programs\Python\Python311\python.exe",
    "python"
  )
  foreach($c in $cands){ try{ $v=& $c -V 2>$null; if($LASTEXITCODE -eq 0 -or $v){ $PY=$c; break } }catch{} }
}
if(-not $PY){ Write-Host "[GUARD] No Python found" -ForegroundColor Red; exit 1 }

try{
  $chk=& $PY -c "import sqlite3,sys;print('sqlite3-ok|'+sys.version)" 2>&1
  Write-Host "[GUARD] Python: $PY" -ForegroundColor Green
  Write-Host "[GUARD] Self-check: $chk" -ForegroundColor DarkGreen
}catch{
  Write-Host "[GUARD] Python failed" -ForegroundColor Red
  exit 1
}

# ---- ???? ----
$DB   = 'D:\quant_system_v2\data\market_data.db'
$SYMS = 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,APTUSDT'
$TFS  = '5m,15m,30m,1h,2h,4h,1d'
$LOOP = '1'
# ------------------

$ts   = Get-Date -Format 'yyyyMMdd_HHmmss'
$GLOG = Join-Path $LOGDIR ("guardian_{0}.log"     -f $ts)
$ELOG = Join-Path $LOGDIR ("guardian_{0}.err.log" -f $ts)

function Log($m,[string]$c='Gray'){
  $line=("["+([DateTime]::UtcNow.ToString('u'))+"] "+$m)
  try{ Add-Content -Path $GLOG -Value $line -Encoding ASCII }catch{}
  Write-Host $m -ForegroundColor $c
}

try{ Set-Content -Path $PIDFILE -Value $PID -Encoding ASCII -Force }catch{}

function Backoff([int]$n){ if($n -lt 5){2*$n}else{10} }

$attempt=0
while($true){
  if(Test-Path (Join-Path $TASKS 'guardian.stop')){ Log "Stop signal detected" 'Yellow'; break }

  $pyArgs = @(
    (Join-Path $BIN 'collector_rt.py'),
    '--db', $DB,
    '--symbols', $SYMS,
    '--intervals', $TFS,
    '--loop', $LOOP
  )
  Log ("Launch: {0} {1}" -f $PY,($pyArgs -join ' ')) 'Cyan'

  $p = Start-Process -FilePath $PY -ArgumentList $pyArgs `
       -NoNewWindow -PassThru -WorkingDirectory $ROOT `
       -RedirectStandardOutput $GLOG -RedirectStandardError $ELOG

  $p.WaitForExit()
  $code=$p.ExitCode
  Log ("[GUARD] Exit code={0}" -f $code) 'Yellow'

  if($code -in 0,2){ $attempt=0; Start-Sleep -Seconds 5; continue }
  $attempt++; $sl=Backoff $attempt
  Log ("[GUARD] Abnormal exit. Retry #$attempt after ${sl}s") 'Magenta'
  Start-Sleep -Seconds $sl
}

try{ Remove-Item $PIDFILE -Force -ErrorAction SilentlyContinue }catch{}
Log "[GUARD] Stopped and cleaned PID." 'Green'
