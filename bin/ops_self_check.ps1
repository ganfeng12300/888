param(
  [switch]$Json,
  [switch]$FailOnError,
  [int]$MaxMinutesStale = 10,     # 守护日志“静默”阈值（分钟）
  [int]$RecentBackupHours = 36    # 备份“新鲜”阈值（小时）
)

$ErrorActionPreference = 'Stop'
$ROOT   = 'D:\SQuant_Pro'
$LOGDIR = Join-Path $ROOT 'logs'
$BKDIR  = Join-Path $ROOT 'data\backup'
$TASKS  = Join-Path $ROOT 'tasks'
$DB     = 'D:\quant_system_v2\data\market_data.db'

# --- Python 探测 ---
$PY = @(
  'C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe',
  "$env:LocalAppData\Programs\Python\Python39\python.exe",
  "$env:LocalAppData\Programs\Python\Python310\python.exe",
  "$env:LocalAppData\Programs\Python\Python311\python.exe",
  'python'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if(-not $PY){ throw "未找到 Python 解释器" }

function Invoke-PyCheck([string]$dbPath){
  $code = @"
import sqlite3, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("MISS"); sys.exit(0)
try:
    con = sqlite3.connect(p)
    v = con.execute('PRAGMA integrity_check').fetchone()[0]
    con.close()
    print(v)
except Exception as e:
    print("ERR:", e)
"@
  $tf = "$env:TEMP\sq_selfcheck.py"
  Set-Content $tf $code -Encoding ascii
  $out = & $PY $tf $dbPath 2>&1 | Where-Object { $_ -notmatch 'A6-SHIM' }
  Remove-Item $tf -Force -ErrorAction SilentlyContinue
  ($out -join "`n").Trim()
}

# --- 颜色助手（替代 ?:）---
function Color([bool]$ok,[string]$good='Green',[string]$bad='Yellow'){
  if($ok){ $good } else { $bad }
}

# --- 守护状态 ---
$guardian = @{
  pid=$null; running=$false; start=$null; stopFlag=$false; staleMinutes=$null; logPath=$null
}
$guardian.stopFlag = Test-Path (Join-Path $TASKS 'guardian.stop')
$pidFile = Join-Path $TASKS 'collector_guard.pid'
if(Test-Path $pidFile){
  $guardian.pid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if($guardian.pid){
    try{ $p = Get-Process -Id $guardian.pid -ErrorAction Stop; $guardian.running = $true; $guardian.start = $p.StartTime }catch{}
  }
}
$gl = Get-ChildItem $LOGDIR -Filter 'guardian_*.log' -File |
      Where-Object { $_.Name -notmatch '\.err\.log$' } |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
if($gl){
  $guardian.logPath = $gl.FullName
  $delta = (Get-Date).ToUniversalTime() - $gl.LastWriteTimeUtc
  $guardian.staleMinutes = [int][Math]::Floor($delta.TotalMinutes)
}

# --- 活库完整性 ---
$dbSizeMB = $null; if(Test-Path $DB){ $dbSizeMB = [math]::Round((Get-Item $DB).Length/1MB,1) }
$dbCheck  = Invoke-PyCheck $DB
$liveOk   = ($dbCheck -eq 'ok')

# --- 最新备份 ---
$latestBk = $null
if(Test-Path $BKDIR){
  $latestBk = Get-ChildItem $BKDIR -Filter *.db -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
$bk = @{ path=$null; name=$null; ageHours=$null; check=$null; ok=$false; freshOk=$false }
if($latestBk){
  $bk.path     = $latestBk.FullName
  $bk.name     = $latestBk.Name
  $bk.ageHours = [int][Math]::Round(((Get-Date) - $latestBk.LastWriteTime).TotalHours)
  $bk.check    = Invoke-PyCheck $bk.path
  $bk.ok       = ($bk.check -eq 'ok')
  $bk.freshOk  = ($bk.ageHours -le $RecentBackupHours)
}

# --- 磁盘空间 ---
$disk = @{ drive='D:'; freeGB=$null; pct=$null }
try{
  $d = Get-PSDrive -Name D -ErrorAction Stop
  $total = $d.Used + $d.Free
  $disk.freeGB = [math]::Round($d.Free/1GB,1)
  if($total -gt 0){ $disk.pct = [math]::Round(100*$d.Free/$total,1) }
}catch{}

# --- 规则判定 ---
$guardianOk = ($guardian.running -and -not $guardian.stopFlag -and ($guardian.staleMinutes -ne $null) -and ($guardian.staleMinutes -le $MaxMinutesStale))
$backupFresh = ($bk.ageHours -ne $null -and $bk.ageHours -le $RecentBackupHours)
$diskOk = ($disk.freeGB -ne $null -and $disk.freeGB -gt 5)

$result = @{
  guardian = @{ ok=$guardianOk; pid=$guardian.pid; running=$guardian.running; stopFlag=$guardian.stopFlag; staleMin=$guardian.staleMinutes; logPath=$guardian.logPath; startTime=$guardian.start }
  db       = @{ ok=$liveOk; path=$DB; sizeMB=$dbSizeMB; check=$dbCheck }
  backup   = @{ ok=($bk.ok -and $backupFresh); path=$bk.path; name=$bk.name; ageHours=$bk.ageHours; check=$bk.check; freshOk=$backupFresh }
  disk     = $disk
}
$result.overallOk = ($result.guardian.ok -and $result.db.ok -and $result.backup.ok -and $diskOk)

if($Json){
  $result | ConvertTo-Json -Depth 5
}else{
  Write-Host "== SQuant Self-Check ==" -ForegroundColor Cyan
  $g = $result.guardian
  Write-Host ("[Guardian] PID={0} Running={1} StopFlag={2} LogFresh={3}m (<= {4}m)" -f $g.pid,$g.running,$g.stopFlag,$g.staleMin,$MaxMinutesStale) -ForegroundColor (Color $result.guardian.ok 'Green' 'Yellow')
  $d = $result.db
  Write-Host ("[DB] {0}  Size={1}MB  Integrity={2}" -f $d.path,$d.sizeMB,$d.check) -ForegroundColor (Color $d.ok 'Green' 'Red')
  $b = $result.backup
  Write-Host ("[Backup] {0}  Age={1}h  Integrity={2}  Fresh={3}" -f $b.name,$b.ageHours,$b.check,$b.freshOk) -ForegroundColor (Color $b.ok 'Green' 'Yellow')
  Write-Host ("[Disk] D: Free={0}GB ({1}%)" -f $disk.freeGB,$disk.pct) -ForegroundColor (Color $diskOk 'Green' 'Yellow')
  if($result.overallOk){ $res='PASS' } else { $res='FAIL' }
  Write-Host ("[Result] {0}" -f $res) -ForegroundColor (Color $result.overallOk 'Green' 'Red')
}

if($FailOnError){
  if($result.overallOk){ exit 0 } else { exit 1 }
}
