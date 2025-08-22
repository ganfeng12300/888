param([switch]$NoStart)

$ErrorActionPreference='Stop'
$ROOT  = 'D:\SQuant_Pro'
$BIN   = Join-Path $ROOT 'bin'
$TASKS = Join-Path $ROOT 'tasks'
$BKDIR = Join-Path $ROOT 'data\backup'
$DB    = 'D:\quant_system_v2\data\market_data.db'
$PY    = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe'

# 1) ???
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $BIN 'Stop_Guardian.ps1')

# 2) ?????
$bk = Get-ChildItem $BKDIR -Filter *.db -File | Sort LastWriteTime -Desc | Select -First 1
if(-not $bk){ throw "????? (*.db) ? $BKDIR" }

# 3) ???????
$pre = Join-Path $BKDIR ("market_data_pre_restore_{0}.db" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
if(Test-Path $DB){ Copy-Item $DB $pre -Force }

# 4) ???????
Copy-Item $bk.FullName $DB -Force
Write-Host "[RESTORE] restored from $($bk.Name) -> $DB"
if(Test-Path $pre){ Write-Host "[RESTORE] pre-restore backup: $pre" }

# 5) ?? stop ??????
Remove-Item (Join-Path $TASKS 'guardian.stop') -Force -ErrorAction SilentlyContinue

# 6) ???????
$code = @"
import sqlite3, sys
db = r"$DB"
con = sqlite3.connect(db)
print("integrity_check:", con.execute("PRAGMA integrity_check").fetchone()[0])
con.close()
"@
$tf = "$env:TEMP\sq_restore_check.py"
Set-Content $tf $code -Encoding ascii
& $PY $tf
Remove-Item $tf -Force -ErrorAction SilentlyContinue

# 7) ?????-NoStart ????
if(-not $NoStart){
  powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $BIN 'Start_Guardian.ps1') -PythonHint $PY
}
