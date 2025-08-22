param([switch]$NoRestart)

$ErrorActionPreference = 'Stop'
$ROOT  = 'D:\SQuant_Pro'
$BIN   = Join-Path $ROOT 'bin'
$TASKS = Join-Path $ROOT 'tasks'
$DB    = 'D:\quant_system_v2\data\market_data.db'
$PY    = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe'

# 1) ???
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $BIN 'Stop_Guardian.ps1')

# 2) ?????? stop ??????????
Remove-Item (Join-Path $TASKS 'guardian.stop') -Force -ErrorAction SilentlyContinue

# 3) ???
$bkdir = Join-Path $ROOT 'data\backup'
if(-not(Test-Path $bkdir)){ New-Item -ItemType Directory -Path $bkdir -Force | Out-Null }
$bak = Join-Path $bkdir ("market_data_prevac_{0}.db" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
Copy-Item $DB $bak -Force
Write-Host "[VACUUM] Pre-backup -> $bak"

# 4) VACUUM
$code = @"
import sqlite3
db = r'$DB'
con = sqlite3.connect(db); con.execute('VACUUM'); con.close()
print('vacuum ok')
"@
$tf = "$env:TEMP\sq_vacuum_now.py"
Set-Content $tf $code -Encoding ascii
& $PY $tf
Remove-Item $tf -Force -ErrorAction SilentlyContinue

# 5) ??????? -NoRestart ???
if(-not $NoRestart){
  powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $BIN 'Start_Guardian.ps1') -PythonHint $PY
}
Write-Host "[VACUUM] finished. Pre-backup: $bak"
