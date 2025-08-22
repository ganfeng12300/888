param(
  [string]$PythonHint='C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe',
  [int]$BatchSize=10,                 # 姣忔壒鏈€澶氬灏戜釜浜ゆ槗瀵?  [string]$UniverseTier='all',        # 鐢?all/T1/T2/T3
  [string]$Intervals='5m,15m,30m,1h,2h,4h,1d',
  [string]$DB='D:\quant_system_v2\data\market_data.db'
)

$ErrorActionPreference='Stop'
$ROOT='D:\SQuant_Pro'; $BIN=Join-Path $ROOT 'bin'; $LOGDIR=Join-Path $ROOT 'logs'; $TASKS=Join-Path $ROOT 'tasks'
$PIDFILE=Join-Path $TASKS 'collector_guard.pid'
$UJSON =Join-Path $ROOT 'data\universe\symbols_latest.json'
$UTXT  =Join-Path $ROOT ("data\universe\{0}.txt" -f $UniverseTier)

mkdir $LOGDIR -Force|Out-Null; mkdir $TASKS -Force|Out-Null
if(!(Test-Path $UTXT)){ throw "universe list missing: $UTXT. Run ops_refresh_universe.ps1 first." }

# Python 妫€娴?$PY=$PythonHint; if(-not(Test-Path $PY)){ $PY='python' }
$chk=& $PY -c "import sqlite3,sys;print('sqlite3-ok|'+sys.version)" 2>&1
Write-Host "[GUARD] Python: $PY"
Write-Host "[GUARD] Self-check: $chk"

# 鏃ュ織
$ts=Get-Date -Format 'yyyyMMdd_HHmmss'
$GLOG=Join-Path $LOGDIR ("guardian_univ_{0}.log" -f $ts)
$ELOG=Join-Path $LOGDIR ("guardian_univ_{0}.err.log" -f $ts)
Set-Content -Path $PIDFILE -Value $PID -Encoding ascii -Force

function Log($m,[string]$c='Gray'){ $line="[{0}] {1}" -f ([DateTime]::UtcNow.ToString('u')),$m; Add-Content $GLOG $line -Encoding ascii; Write-Host $m -Foreground $c }

# 璇诲竵姹犲苟鍒嗘壒
$SYMS = Get-Content $UTXT | Where-Object { $_ -match '^[A-Z0-9]+' }
function Chunks($arr,[int]$n){ for($i=0;$i -lt $arr.Count;$i+=$n){ ,($arr[$i..([Math]::Min($i+$n-1,$arr.Count-1))]) } }

function Run-Collector([string[]]$syms){
  $symStr = ($syms -join ',')
  $args = @((Join-Path $BIN 'collector_rt.py'),'--db',$DB,'--symbols',$symStr,'--intervals',$Intervals,'--loop','1')
  Log ("Launch: {0} {1}" -f $PY,($args -join ' ')) 'Cyan'
  $p = Start-Process -FilePath $PY -ArgumentList $args -NoNewWindow -PassThru -WorkingDirectory $ROOT -RedirectStandardOutput $GLOG -RedirectStandardError $ELOG
  $p.WaitForExit(); $code=$p.ExitCode
  Log ("[GUARD] Exit code={0}" -f $code) 'Yellow'
}

# 涓诲惊鐜細T1/T2/T3/all 鎵规杞
while($true){
  if(Test-Path (Join-Path $TASKS 'guardian.stop')){ Log "Stop signal detected" 'Yellow'; break }
  foreach($chunk in (Chunks $SYMS $BatchSize)){
    Run-Collector $chunk
    Start-Sleep -Seconds 2  # 杞昏妭娴?  }
  # 瀹屾垚涓€杞悗锛岀煭浼?+ 閲嶆柊璇诲彇甯佹睜锛堝姩鎬侊級
  Start-Sleep -Seconds 5
  if(Test-Path $UTXT){ $SYMS = Get-Content $UTXT | Where-Object { $_ -match '^[A-Z0-9]+' } }
}
Remove-Item $PIDFILE -Force -ErrorAction SilentlyContinue
Log "[GUARD] Stopped and cleaned PID." 'Green'



