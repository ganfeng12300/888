param(
  [string]$BackupDir = 'D:\SQuant_Pro\data\backup',
  [string]$DrillDir  = 'D:\SQuant_Pro\data\drill',
  [string]$LogDir    = 'D:\SQuant_Pro\logs'
)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Path $DrillDir -Force | Out-Null
New-Item -ItemType Directory -Path $LogDir  -Force | Out-Null

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $LogDir "drill_$ts.log"

function Write-Log($m){ $line="[$(Get-Date -Format 'u')] $m"; Add-Content -Path $log -Value $line; Write-Host $m }

$bk = Get-ChildItem $BackupDir -Filter *.db -File | Sort-Object LastWriteTime -Desc | Select-Object -First 1
if(-not $bk){ Write-Log "NO BACKUP FOUND"; exit 1 }

$dst = Join-Path $DrillDir ("market_data_drill_{0}.db" -f $ts)
Copy-Item $bk.FullName $dst -Force
Write-Log "DRILL: restored $($bk.Name) -> $dst"

# integrity_check
$py = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe'
$code = @"
import sqlite3, sys
db = r'$dst'
con = sqlite3.connect(db)
res = con.execute('PRAGMA integrity_check').fetchone()[0]
con.close()
print(res)
"@
$tf = "$env:TEMP\sq_drill_check.py"
Set-Content $tf $code -Encoding ascii
$out = & $py $tf 2>&1
Remove-Item $tf -Force -ErrorAction SilentlyContinue
Write-Log "PRAGMA integrity_check => $out"

if(($out -join '').Trim() -ne 'ok'){ Write-Log 'DRILL FAIL: integrity not ok'; exit 2 }

# 跑一次日常自检，看整机状态是否仍 PASS
$sc = (& powershell -NoProfile -ExecutionPolicy Bypass -File 'D:\SQuant_Pro\bin\ops_self_check.ps1' -Json | ConvertFrom-Json)
Write-Log ("SELF-CHECK overallOk={0}" -f $sc.overallOk)

# 清理演练文件
Remove-Item $dst -Force
Write-Log "DRILL OK and cleaned."
