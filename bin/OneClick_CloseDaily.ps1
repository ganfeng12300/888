$ErrorActionPreference = "Stop"
$db  = "D:\quant_system_v2\data\market_data.db"
$py  = (Test-Path "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe") ?
        "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe" : "python"
$patchPy = "D:\SQuant_Pro\bin\patch_d1_closeonly.py"

# ?? Python ???ASCII?
$code = @"
import sqlite3, time, os, shutil
DB = r"D:\quant_system_v2\data\market_data.db"
now = int(time.time())
last_closed_open = (now//86400)*86400 - 86400  # ?????????K???? UTC ??
bdir = r"D:\SQuant_Pro\data\backup"
os.makedirs(bdir, exist_ok=True)
bkp  = os.path.join(bdir, f"market_data_{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}.db")
if os.path.exists(DB):
    shutil.copy2(DB, bkp)
    print("Backup:", bkp)

con = sqlite3.connect(DB)
tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_1d'")]
total_removed = 0
for t in tabs:
    c = con.execute(f"SELECT COUNT(*) FROM '{t}' WHERE timestamp > ?", (last_closed_open,)).fetchone()[0]
    if c>0:
        con.execute(f"DELETE FROM '{t}' WHERE timestamp > ?", (last_closed_open,))
        print(f"{t}: removed {c} open-day rows")
        total_removed += c
con.commit(); con.close()
print("last_closed_open_utc:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(last_closed_open)))
print("total_removed:", total_removed)
"@
[IO.File]::WriteAllText($patchPy, $code, [Text.Encoding]::ASCII)

Write-Host "[1/3] ???????? _1d ?????K??????????..." -ForegroundColor Cyan
& $py -u $patchPy | Out-Host

# ??????????
$ak = "D:\Build_AuditKit_Anchored_v2.ps1"
if (!(Test-Path $ak)) { $ak = "D:\Build_AuditKit_Anchored.ps1" }
if (Test-Path $ak) {
  Write-Host "[2/3] ??????..." -ForegroundColor Cyan
  powershell -ExecutionPolicy Bypass -NoProfile -File $ak | Out-Host
  $last = Get-ChildItem -Directory "D:\AuditRepair\AuditKit\logs" | Sort-Object LastWriteTime -Desc | Select-Object -First 1
  $summary = Join-Path $last.FullName "00_SUMMARY.html"
  if (Test-Path $summary) { Start-Process $summary }
}

Write-Host "[3/3] ????????????????K??" -ForegroundColor Green