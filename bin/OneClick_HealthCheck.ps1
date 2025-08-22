# OneClick_HealthCheck.ps1  — 机构终极版 v2.3
# 产物：health_report.json / health_report.html / issues_summary.txt
# 可选参数：
#   -Root1 "D:\quant_system_pro (3)\quant_system_pro"
#   -Root2 "D:\SQuant_Pro"
#   -DbPath "D:\quant_system_v2\data\market_data.db"
#   -OutDir "D:\SQuant_Pro\healthcheck"
#   -LogsDir "D:\SQuant_Pro\logs"
#   -PythonCmd "auto|python|py -3|自定义路径"
#   -DbProbeTimeoutSec 300
#   -DbQuick           （开关：仅抽样前12个交易对）
[CmdletBinding()]
param(
  [string]$Root1   = "D:\quant_system_pro (3)\quant_system_pro",
  [string]$Root2   = "D:\SQuant_Pro",
  [string]$DbPath  = "D:\quant_system_v2\data\market_data.db",
  [string]$OutDir  = "D:\SQuant_Pro\healthcheck",
  [string]$LogsDir = "D:\SQuant_Pro\logs",
  [string]$PythonCmd = "auto",
  [int]   $DbProbeTimeoutSec = 300,
  [switch]$DbQuick
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try { [Net.ServicePointManager]::SecurityProtocol =
        [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls11 -bor [Net.SecurityProtocolType]::Tls } catch {}

# ---------- 工具函数 ----------
function New-Result { param([string]$id,[string]$name,[string]$status,[string]$details,[string]$evidence,[string]$suggestion)
  [pscustomobject]@{ id=$id; name=$name; status=$status; details=$details; evidence=$evidence; suggestion=$suggestion } }
function Has-Prop($o,$n){
  if($null -eq $o){ return $false }
  if($o -is [System.Collections.IDictionary]){ return $o.Contains($n) }
  return ($o.PSObject.Properties[$n] -ne $null)
}
function Get-Prop($o,$n){
  if($null -eq $o){ return $null }
  if($o -is [System.Collections.IDictionary]){ return $o[$n] }
  $p=$o.PSObject.Properties[$n]; if($p){ return $p.Value } else { return $null }
}
function Read-JsonRobust([string]$path,[int]$retries=12,[int]$sleepMs=250){
  for($i=0;$i -le $retries;$i++){
    if(Test-Path $path){
      try{
        $before=(Get-Item $path).Length
        Start-Sleep -Milliseconds ([math]::Max(50,[int]($sleepMs/2)))
        $after=(Get-Item $path).Length
        if($before -eq $after){
          $raw = Get-Content $path -Raw -Encoding UTF8
          if($raw -and $raw.Trim().Length -gt 0){
            try{ return ($raw | ConvertFrom-Json -ErrorAction Stop) }catch{}
          }
        }
      }catch{}
    }
    Start-Sleep -Milliseconds $sleepMs
  }
  return $null
}

# ---------- Universe ----------
$SYMS_ALL = @(
 'BTCUSDT','ETHUSDT','OPUSDT','RNDRUSDT','ARBUSDT','FETUSDT','SOLUSDT','LDOUSDT','INJUSDT','AVAXUSDT',
 'APTUSDT','NEARUSDT','ATOMUSDT','SUIUSDT','AAVEUSDT','LINKUSDT','DOGEUSDT','BNBUSDT','XRPUSDT','SEIUSDT',
 'TIAUSDT','DYDXUSDT','CFXUSDT','ENSUSDT','FLOWUSDT','MINAUSDT','BLURUSDT','LRCUSDT','CRVUSDT','KAVAUSDT',
 'UNIUSDT','FILUSDT','EOSUSDT','GALAUSDT','PEPEUSDT','MASKUSDT','IMXUSDT','CHZUSDT','GMTUSDT','ZILUSDT',
 'BCHUSDT','ETCUSDT','TRXUSDT','SANDUSDT','MANAUSDT','WOOUSDT','STXUSDT','1INCHUSDT'
)
$TFS  = @('5m','15m','30m','1h','2h','4h','1d')  # exclude 1m
$SYMS = if($DbQuick){ $SYMS_ALL | Select-Object -First 12 } else { $SYMS_ALL }

$APIs = @(
  @{ name='Binance_Futures_Ping'; url='https://fapi.binance.com/fapi/v1/ping'; timeout=7 },
  @{ name='Binance_Spot_Ping';    url='https://api.binance.com/api/v3/ping';  timeout=7 },
  @{ name='OKX_Time';             url='https://www.okx.com/api/v5/public/time'; timeout=7 },
  @{ name='Bitget_Time_Mix';      url='https://api.bitget.com/api/mix/v1/market/time'; timeout=7 },
  @{ name='Bitget_Time_Spot';     url='https://api.bitget.com/api/spot/v1/public/time'; timeout=7 }
)

# ---------- Output paths ----------
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ReportJson = Join-Path $OutDir "health_report.json"
$ReportHtml = Join-Path $OutDir "health_report.html"
$IssuesTxt  = Join-Path $OutDir "issues_summary.txt"
$DbProbePy  = Join-Path $OutDir "hc_db_probe.py"
$DbProbeOut = Join-Path $OutDir "db_probe_out.json"

$results = New-Object System.Collections.Generic.List[object]
$summary = @{
  started_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'")
  root1 = $Root1; root2 = $Root2; db = $DbPath; outdir = $OutDir; quick = [bool]$DbQuick
}

# ---------- Resolve Python ----------
function Resolve-Python {
  param([string]$Cmd)
  if($Cmd -and $Cmd -ne "auto"){
    try { if(Get-Command $Cmd -ErrorAction Stop){ return @{ Cmd=$Cmd } } } catch {}
    return $null
  }
  try { if(Get-Command python -ErrorAction Stop){ return @{ Cmd="python" } } } catch {}
  try { if(Get-Command py      -ErrorAction Stop){ return @{ Cmd="py -3" } } } catch {}
  try { if(Get-Command py      -ErrorAction Stop){ return @{ Cmd="py" } } } catch {}
  return $null
}
$pySpec = Resolve-Python $PythonCmd
if($pySpec){
  $summary.python_cmd = $pySpec.Cmd
  try { $summary.python_cmd_path = (Get-Command ($pySpec.Cmd.Split()[0]) -ErrorAction SilentlyContinue).Source } catch {}
} else {
  $summary.python_cmd = "unresolved"
}

Write-Host "[1/7] Environment checks..." -ForegroundColor Cyan
# ---------- System profile ----------
try {
  $os = Get-CimInstance Win32_OperatingSystem
  $cpu = Get-CimInstance Win32_Processor
  $cs  = Get-CimInstance Win32_ComputerSystem
  $logical  = ($cpu | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
  $physical = ($cpu | Measure-Object -Property NumberOfCores -Sum).Sum
  $memGB = [math]::Round($cs.TotalPhysicalMemory/1GB,2)
  $summary.system = @{
    os=$os.Caption; build=$os.BuildNumber; cpu_name=($cpu|Select-Object -First 1).Name
    cores_physical=$physical; cores_logical=$logical; memory_gb=$memGB
  }
  $results.Add((New-Result "env.sys" "System profile" "OK" "OS/CPU/RAM captured" "$($os.Caption) CPU=$logical/$physical RAM=$memGB GB" "None"))
} catch {
  $results.Add((New-Result "env.sys" "System profile" "WARN" "Failed to capture OS/CPU/RAM" "$($_.Exception.Message)" "Proceed but verify resources"))
}

# ---------- Python runner (A6-SHIM immune) ----------
function Invoke-Py {
  param([string]$Code,[int]$TimeoutSec=20)
  if(-not $pySpec){ return @{ out=""; err="python not found"; code=9001 } }
  $tmp = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName()+".py")
  Set-Content -LiteralPath $tmp -Encoding UTF8 -Value $Code
  try {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $cmdParts = $pySpec.Cmd.Split(' ',2)
    $exe = $cmdParts[0]; $preArgs = ""; if($cmdParts.Count -gt 1){ $preArgs = $cmdParts[1] }
    $psi.FileName = $exe
    $psi.Arguments = ($preArgs + " -S -u `"$tmp`"").Trim()  # -S: no site; -u: unbuffered
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute = $false
    $psi.EnvironmentVariables["PYTHONNOUSERSITE"] = "1"
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    $p.Start() | Out-Null
    if(-not $p.WaitForExit($TimeoutSec*1000)){ try{$p.Kill()}catch{}; throw "python timeout" }
    return @{ out=$p.StandardOutput.ReadToEnd(); err=$p.StandardError.ReadToEnd(); code=$p.ExitCode }
  } finally { try { Remove-Item $tmp -Force -ErrorAction SilentlyContinue } catch {} }
}

# ---------- Python presence ----------
$pyOK = $false
try {
  $r = Invoke-Py "import sys,json; print(json.dumps({'ok':True,'ver':sys.version.split()[0]}))" 10
  $obj = $null; if($r.out){ $obj = $r.out | ConvertFrom-Json -ErrorAction SilentlyContinue }
  if($obj -and $obj.ok){ $pyOK=$true; $summary.python_version=$obj.ver
    $results.Add((New-Result "env.py" "Python" "OK" "Python detected" "version=$($obj.ver)" "None"))
  } else {
    $results.Add((New-Result "env.py" "Python" "FAIL" "Python not detected" ("out="+$r.out+" err="+$r.err) "Ensure Python on PATH"))
  }
} catch {
  $results.Add((New-Result "env.py" "Python" "FAIL" "Python not detected" "$($_.Exception.Message)" "Ensure Python on PATH"))
}

# ---------- Optional deps (soft) ----------
if($pyOK){
  try {
    $depCode = @"
import importlib, json
mods = ['sqlite3','pandas','numpy','torch','lightgbm']
res = {}
for m in mods:
    try:
        importlib.import_module(m)
        if m=='torch':
            import torch
            cuda = bool(getattr(torch,'cuda',None) and torch.cuda.is_available())
            res[m] = {'ok':True,'cuda':cuda}
        else:
            res[m] = {'ok':True}
    except Exception as e:
        res[m] = {'ok':False,'err':str(e)}
print(json.dumps(res, ensure_ascii=False))
"@
    $dep = Invoke-Py $depCode 25
    $depObj = $null
    if($dep.out){ $depObj = $dep.out | ConvertFrom-Json -ErrorAction SilentlyContinue }
    $summary.deps = $depObj
    $okcnt = 0
    if ($depObj -is [System.Collections.IDictionary]) { $okcnt = @($depObj.GetEnumerator() | Where-Object { $_.Value.ok }).Count }
    elseif ($depObj)                                   { $okcnt = @($depObj.PSObject.Properties | Where-Object { $_.Value.ok }).Count }
    $results.Add((New-Result "env.deps" "Python deps (soft check)" "OK" "Detected $okcnt modules" ($dep.out.Trim()) "Install missing ones only if needed"))
  } catch {
    $results.Add((New-Result "env.deps" "Python deps (soft check)" "WARN" "Dependency probe failed" "$($_.Exception.Message)" "Skip; not blocking"))
  }
}

# ---------- DB files ----------
try {
  $dbExists = Test-Path $DbPath
  $wal = "$DbPath-wal"; $shm="$DbPath-shm"
  $dbInfo = @{}
  $dbInfo.exists = $dbExists
  if ($dbExists) { $dbInfo.size_bytes = (Get-Item $DbPath).Length } else { $dbInfo.size_bytes = 0 }
  $dbInfo.wal_exists = Test-Path $wal; if (Test-Path $wal) { $dbInfo.wal_size_bytes = (Get-Item $wal).Length } else { $dbInfo.wal_size_bytes = 0 }
  $dbInfo.shm_exists = Test-Path $shm; if (Test-Path $shm) { $dbInfo.shm_size_bytes = (Get-Item $shm).Length } else { $dbInfo.shm_size_bytes = 0 }
  $summary.db_files = $dbInfo
  if($dbExists){ $results.Add((New-Result "env.db" "DB files" "OK" "DB located" "DB=$DbPath; size=$($dbInfo.size_bytes)" "None")) }
  else { $results.Add((New-Result "env.db" "DB files" "FAIL" "DB not found" "DB=$DbPath" "Verify DB path")) }
} catch {
  $results.Add((New-Result "env.db" "DB files" "WARN" "Unable to stat DB" "$($_.Exception.Message)" "Check permissions/path"))
}

Write-Host "[2/7] Network/API probes..." -ForegroundColor Cyan
# ---------- Network ----------
$apiResults = @()
foreach($api in $APIs){
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  try{
    $resp = Invoke-WebRequest -Uri $api.url -Method Get -TimeoutSec $api.timeout -UseBasicParsing
    $sw.Stop(); $ms = $sw.ElapsedMilliseconds
    $apiResults += @{ name=$api.name; url=$api.url; ok=$true; status=$resp.StatusCode; ms=$ms }
    $results.Add((New-Result ("net."+($api.name)) $api.name "OK" "Reachable" ("status="+$resp.StatusCode+" time_ms="+$ms) "None"))
  } catch {
    $sw.Stop()
    $apiResults += @{ name=$api.name; url=$api.url; ok=$false; status=0; ms=$sw.ElapsedMilliseconds; err=$_.Exception.Message }
    $results.Add((New-Result ("net."+($api.name)) $api.name "WARN" "Unreachable or slow" ("err="+$_.Exception.Message) "Check network or regional blocking"))
  }
}
$summary.api = $apiResults

Write-Host "[3/7] Path/config consistency..." -ForegroundColor Cyan
# ---------- Code/Config scan ----------
function Find-String {
  param([string[]]$Paths,[string]$Pattern)
  $hits = @()
  foreach($p in $Paths){
    $files = @()
    $files += Get-ChildItem -Path $p -Recurse -Include *.bat,*.ps1,*.yaml,*.yml,*.ini,*.json,*.py -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    $self = $MyInvocation.MyCommand.Path
    $files = $files | Where-Object { $_ -ne $self }    # exclude self
    if($files -and $files.Count -gt 0){
      $found = Select-String -Path $files -Pattern $Pattern -SimpleMatch -ErrorAction SilentlyContinue
      if($found){ $hits = @($hits) + @($found) }
    }
  }
  return ,$hits
}
$wmicHits=@(); $oneMinHits=@(); $legacyPathHits=@()
try { $wmicHits      = @(Find-String -Paths @($Root1,$Root2) -Pattern "wmic") } catch {}
try { $oneMinHits    = @(Find-String -Paths @($Root1,$Root2) -Pattern " 1m") } catch {}
try {
  $legacyPathHits = @(Find-String -Paths @($Root1,$Root2) -Pattern "D:\888")
  $tmp = @(Find-String -Paths @($Root1,$Root2) -Pattern "D:\777")
  if($tmp.Count -gt 0){ $legacyPathHits = @($legacyPathHits) + @($tmp) }
} catch {}
$summary.path_scan = @{
  wmic_refs   = ($wmicHits | Select-Object -First 10 | ForEach-Object { @{ Path=$_.Path; Line=$_.LineNumber; Text=$_.Line.Trim() } })
  one_min_refs= ($oneMinHits | Select-Object -First 10 | ForEach-Object { @{ Path=$_.Path; Line=$_.LineNumber; Text=$_.Line.Trim() } })
  legacy_refs = ($legacyPathHits | Select-Object -First 10 | ForEach-Object { @{ Path=$_.Path; Line=$_.LineNumber; Text=$_.Line.Trim() } })
}
$results.Add((New-Result "cfg.wmic"   "wmic usage"               ($(if($wmicHits.Count -gt 0){"WARN"}else{"OK"}))   "Detect deprecated wmic references" ("hits="+$($wmicHits.Count))   "Replace with PowerShell Get-CimInstance/Stop-Process"))
$results.Add((New-Result "cfg.1m"     "1m timeframe references"  ($(if($oneMinHits.Count -gt 0){"WARN"}else{"OK"})) "1m should be excluded per policy"  ("hits="+$($oneMinHits.Count)) "Purge 1m in configs; keep 5m..1d only"))
$results.Add((New-Result "cfg.legacy" "Legacy path references"   ($(if($legacyPathHits.Count -gt 0){"WARN"}else{"OK"})) "Hardcoded legacy paths detected" ("hits="+$($legacyPathHits.Count)) "Normalize to D:\quant_system_v2 or D:\SQuant_Pro"))

Write-Host "[4/7] Guardian/logs sanity..." -ForegroundColor Cyan
# ---------- Guardian & Logs ----------
$pidFile = Join-Path $Root2 "tasks\collector_guard.pid"
$pidFileExists = Test-Path $pidFile
$guardianOk = $false; $guardianNote = ""
if($pidFileExists){
  try {
    $pidRaw = (Get-Content $pidFile -ErrorAction Stop | Select-Object -First 1).Trim()
    if ($pidRaw -match '(\d+)') {
      $guardPid = [int]$Matches[1]
      $proc = Get-Process -Id $guardPid -ErrorAction SilentlyContinue
      if($proc){ $guardianOk = $true; $guardianNote = "PID="+$guardPid } else { $guardianNote = "PID file stale" }
    } else { $guardianNote = "Invalid PID content: " + $pidRaw }
  } catch { $guardianNote = "Invalid PID file: $($_.Exception.Message)" }
} else { $guardianNote = "PID file absent (may be stopped or not created)" }
$results.Add((New-Result "guard.pid" "Guardian PID" ($(if($guardianOk){"OK"}else{"WARN"})) "PID status" $guardianNote "If running, ensure it writes PID atomically; if stopped, ignore"))

# latest log
$latestLog = $null
try {
  $latestLog = Get-ChildItem -Path (Join-Path $LogsDir "collector_task*.log") -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Desc | Select-Object -First 1
} catch {}
if($latestLog){
  $evi = "latest="+$latestLog.FullName+" lastWrite="+$latestLog.LastWriteTime.ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'")
  $results.Add((New-Result "logs.latest" "Latest collector log" "OK" "Found rotating or fixed log" $evi "Tail newest when verifying realtime"))
} else {
  $results.Add((New-Result "logs.latest" "Latest collector log" "WARN" "No matching logs found" "pattern=collector_task*.log" "Check log path/permissions"))
}

# hashtable anomaly in logs
$htHits=@()
try{
  $logFiles = Get-ChildItem -Path (Join-Path $LogsDir "*.log") -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
  if($logFiles){ $h = Select-String -Path $logFiles -Pattern "System\.Collections\.Hashtable" -ErrorAction SilentlyContinue }
  if($h){ $htHits = $h }
}catch{}
$results.Add((New-Result "logs.ht" "Hashtable path anomaly" ($(if($htHits){"WARN"}else{"OK"})) "Detect wrong object-to-string path usage" ("hits="+$($htHits.Count)) "Ensure log path variables are strings only"))

Write-Host "[5/7] Database probe (read-only)..." -ForegroundColor Cyan
# ---------- DB Probe（写文件避免管道阻塞 + 聚合查询） ----------
$pyCode = @"
import sys, json, sqlite3, os
from datetime import datetime, timezone

OUT = r'$DbProbeOut'
DB  = r'$DbPath'
IS_QUICK = bool($([int]([bool]$DbQuick)))

def write(o):
    try:
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(o, f, ensure_ascii=False)
    except Exception as e:
        print(json.dumps({'error': f'write_failed: {e}'}))

try:
    import json as _json_check
    SYMS = _json_check.loads(r'$($SYMS | ConvertTo-Json -Compress)')
    TFS  = _json_check.loads(r'$($TFS  | ConvertTo-Json -Compress)')
except Exception as e:
    write({'error': f'config_json_failed: {e}'}); sys.exit(0)

def connect_ro(path):
    try:
        uri = f'file:{path}?mode=ro'
        con = sqlite3.connect(uri, uri=True, timeout=15, isolation_level=None)
        try:
            con.execute('PRAGMA query_only=ON')
            con.execute('PRAGMA temp_store=2')        # MEMORY
            con.execute('PRAGMA cache_size=-200000')  # ~200MB
        except Exception:
            pass
        return con
    except Exception as e:
        write({'error': f'connect_failed: {e}'}); sys.exit(0)

bars_per_day = {'5m':288,'15m':96,'30m':48,'1h':24,'2h':12,'4h':6,'1d':1}
now = datetime.now(timezone.utc); now_ts = int(now.timestamp())
cut30  = now_ts - 30*24*3600
cut90  = now_ts - 90*24*3600
cut365 = now_ts - 365*24*3600

res = {'now_ts': now_ts, 'pairs': [], 'agg': {}, 'quick': IS_QUICK}

con = connect_ro(DB)
try:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = set(r[0] for r in rows)
except Exception as e:
    write({'error': f'tables_query_failed: {e}'}); sys.exit(0)

if not tables:
    write({'error': 'no_tables_found'}); sys.exit(0)

def tbl(sym, tf): return f"{sym}_{tf}"

SQL = "SELECT MAX(timestamp), " \
      "SUM(CASE WHEN timestamp>=? THEN 1 ELSE 0 END), " \
      "SUM(CASE WHEN timestamp>=? THEN 1 ELSE 0 END), " \
      "SUM(CASE WHEN timestamp>=? THEN 1 ELSE 0 END) FROM '{}'"

pairs = []
for s in SYMS:
    for tf in TFS:
        name = tbl(s, tf)
        if name not in tables:
            pairs.append({'symbol': s, 'tf': tf, 'exists': False}); continue
        try:
            q = SQL.format(name)
            row = cur.execute(q, (cut30, cut90, cut365)).fetchone()
            latest, c30, c90, c365 = row if row else (None, None, None, None)
            info = {'symbol': s, 'tf': tf, 'exists': True}
            info['latest'] = int(latest) if latest is not None else None
            info['counts'] = {'30': c30, '90': c90, '365': c365}
            bpd = bars_per_day.get(tf)
            cov = {}
            for days, cnt in info['counts'].items():
                cov[days] = None if (cnt is None or bpd is None) else round(min(1.0, (cnt or 0)/(int(days)*bpd)), 4)
            info['coverage'] = cov
            info['future_k'] = (info['latest'] is not None and info['latest'] > now_ts + 120)
            info['last_age_sec'] = (now_ts - info['latest']) if info['latest'] is not None else None
            pairs.append(info)
        except Exception as e:
            pairs.append({'symbol': s, 'tf': tf, 'exists': False, 'error': f'probe_failed: {e}'})

res['pairs'] = pairs

try:
    ok30 = sum(1 for p in pairs if p.get('exists') and p.get('coverage',{}).get('30') is not None and p['coverage']['30'] >= 0.95)
    ok90 = sum(1 for p in pairs if p.get('exists') and p.get('coverage',{}).get('90') is not None and p['coverage']['90'] >= 0.95)
    ok365= sum(1 for p in pairs if p.get('exists') and p.get('coverage',{}).get('365') is not None and p['coverage']['365']>= 0.95)
    tot  = sum(1 for p in pairs if p.get('exists'))
    fut  = sum(1 for p in pairs if p.get('exists') and p.get('future_k'))
    lag5m= [p for p in pairs if p.get('exists') and p.get('tf')=='5m' and p.get('last_age_sec') not in (None,) and p['last_age_sec']>1800]
    res['agg'] = {'exists': tot, 'ok30_ge95': ok30, 'ok90_ge95': ok90, 'ok365_ge95': ok365, 'future_k': fut, 'lag5m_over_30min': len(lag5m)}
except Exception as e:
    write({'error': f'agg_calc_failed: {e}'}); sys.exit(0)

write(res)
"@
Set-Content -LiteralPath $DbProbePy -Encoding UTF8 -Value $pyCode

# 运行 DB 探针（写文件，父进程稳健读取）
$dbProbe = $null
if($pyOK -and (Test-Path $DbPath)){
  try{
    if(Test-Path $DbProbeOut){ Remove-Item $DbProbeOut -Force -ErrorAction SilentlyContinue }
    $runner = "import runpy; runpy.run_path(r'$DbProbePy', run_name='__main__')"
    $null = Invoke-Py $runner $DbProbeTimeoutSec
    $dbProbe = Read-JsonRobust $DbProbeOut
    if(-not $dbProbe){ $dbProbe = @{ error = "no_output_or_parse_failed" } }
  } catch {
    $dbProbe = @{ error = "run_failed: $($_.Exception.Message)" }
  }
} else {
  $dbProbe = @{ error = "python_or_db_missing" }
}
$summary.db_probe = $dbProbe

# 使用健壮取值，避免 StrictMode 下属性缺失报错
$probeErr = Get-Prop $dbProbe 'error'
if($probeErr){
  $results.Add((New-Result "db.conn" "DB probe" "FAIL" "Probe failed" $probeErr "Check DB path and Python sqlite3"))
} else {
  $agg = Get-Prop $dbProbe 'agg'
  if($null -eq $agg){
    $results.Add((New-Result "db.conn" "DB probe" "FAIL" "Probe failed" "agg_missing" "Inspect db_probe_out.json for details"))
  } else {
    $exists = [int](Get-Prop $agg 'exists')
    $ok30   = [int](Get-Prop $agg 'ok30_ge95')
    $ok90   = [int](Get-Prop $agg 'ok90_ge95')
    $ok365  = [int](Get-Prop $agg 'ok365_ge95')
    $fut    = [int](Get-Prop $agg 'future_k')
    $lag5   = [int](Get-Prop $agg 'lag5m_over_30min')
    $results.Add((New-Result "db.cover30" "Coverage30 >=95%"    ($(if($ok30 -gt 0){"OK"}else{"WARN"})) "Existing pairs with coverage30>=95%" ("ok="+$ok30+" / exists="+$exists) "Run backfill to raise coverage30"))
    $results.Add((New-Result "db.cover90" "Coverage90 >=95%"    ($(if($ok90 -gt 0){"OK"}else{"WARN"})) "Existing pairs with coverage90>=95%" ("ok="+$ok90+" / exists="+$exists) "Backfill until >=95% for core pairs"))
    $results.Add((New-Result "db.cover365" "Coverage365 >=95%"  ($(if($ok365-gt 0){"OK"}else{"WARN"})) "Existing pairs with coverage365>=95%" ("ok="+$ok365+" / exists="+$exists) "Target 365d coverage for full backtest"))
    $results.Add((New-Result "db.futurek" "Future K guard"      ($(if($fut  -gt 0){"FAIL"}else{"OK"})) "Any table latest_ts > now+120s" ("future_k="+$fut) "Fix data source or drop future bars"))
    $results.Add((New-Result "db.fresh5m" "5m freshness"        ($(if($lag5 -gt 0){"WARN"}else{"OK"})) "5m last_age_sec > 1800 count" ("lag_count="+$lag5) "Check realtime collector and API limits"))
  }
}

Write-Host "[6/7] Flow/modules presence (static scan)..." -ForegroundColor Cyan
function Grep-Def { param([string]$name)
  $files = @()
  $files += Get-ChildItem -Path $Root1 -Recurse -Include *.py -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
  $files += Get-ChildItem -Path $Root2 -Recurse -Include *.py -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
  if($files.Count -eq 0){ return $null }
  return (Select-String -Path $files -Pattern $name -ErrorAction SilentlyContinue)
}
$fn1 = Grep-Def "save_optimized_params_to_db"
$fn2 = Grep-Def "strategy_func"
$fn1Hits = if ($fn1) { $fn1.Count } else { 0 }
$fn2Hits = if ($fn2) { $fn2.Count } else { 0 }
$results.Add((New-Result "flow.fn1" "Function save_optimized_params_to_db" ($(if($fn1){"OK"}else{"WARN"})) "Check presence in codebase" ("hits="+$fn1Hits) "Ensure utils/param_loader.py exports it"))
$results.Add((New-Result "flow.fn2" "strategy_func support"    ($(if($fn2){"OK"}else{"WARN"})) "Backtest engine should accept strategy_func" ("hits="+$fn2Hits) "Upgrade backtest engine if missing"))

Write-Host "[7/7] Consolidate & output..." -ForegroundColor Cyan
$final = @{ meta = $summary; results = $results }
$final | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 -LiteralPath $ReportJson

$ordered = @(); $ordered += $results | Where-Object {$_.status -eq 'FAIL'}; $ordered += $results | Where-Object {$_.status -eq 'WARN'}
$sb = New-Object System.Text.StringBuilder
foreach($r in $ordered){
  [void]$sb.AppendLine( ("[{0}] {1}  -- {2}" -f $r.status, $r.name, $r.details) )
  if($r.evidence){ [void]$sb.AppendLine("  evidence: "+$r.evidence) }
  if($r.suggestion){ [void]$sb.AppendLine("  fix: "+$r.suggestion) }
  [void]$sb.AppendLine("")
}
Set-Content -Encoding UTF8 -LiteralPath $IssuesTxt -Value $sb.ToString()

$metaLine = ("Started: {0} | DB: {1} | Roots: {2}, {3} | Quick: {4}" -f $summary.started_utc, $DbPath, $Root1, $Root2, [bool]$DbQuick)
$html = @"
<!doctype html>
<html><head><meta charset="utf-8"><title>Health Report</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:20px;}
h1{margin:0 0 10px 0;} .meta{color:#555;}
table{border-collapse:collapse;width:100%;}
th,td{border:1px solid #ddd;padding:8px;font-size:14px;}
th{background:#f2f2f2;}
.ok{background:#e7f7e7;} .warn{background:#fff7e0;} .fail{background:#fde7e7;}
pre{white-space:pre-wrap;font-size:12px;background:#f9f9f9;border:1px solid #eee;padding:8px;}
</style></head><body>
<h1>Health Report</h1>
<div class="meta">$metaLine</div>
<table><tr><th>Status</th><th>ID</th><th>Name</th><th>Details</th><th>Evidence</th><th>Suggestion</th></tr>
"@
foreach($r in $results){
  $cls = "ok"; if ($r.status -eq 'WARN') { $cls = "warn" } elseif ($r.status -eq 'FAIL') { $cls = "fail" }
  $html += "<tr class='$cls'><td>$($r.status)</td><td>$($r.id)</td><td>$($r.name)</td><td>$($r.details)</td><td><pre>$($r.evidence)</pre></td><td>$($r.suggestion)</td></tr>`n"
}
$html += "</table></body></html>"
Set-Content -Encoding UTF8 -LiteralPath $ReportHtml -Value $html

Write-Host ""
Write-Host "Health check completed." -ForegroundColor Green
Write-Host "Output:" -ForegroundColor Green
Write-Host "  $ReportJson"
Write-Host "  $ReportHtml"
Write-Host "  $IssuesTxt"
