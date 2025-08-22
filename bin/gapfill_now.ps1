param([int]$Days=365,[int]$MaxReaders=16,[int]$MaxWriters=1)

$DB     = "D:\quant_system_v2\data\market_data.db"
$WL     = "D:\SQuant_Pro\configs\whitelist_all.txt"
$REPAIR = "D:\AuditRepair\RepairKit\repairkit.py"

# ?? python?PS5.1 ?????
$PY = "python"
if (Test-Path "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe") {
  $PY = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe"
}

if (!(Test-Path $WL))     { Write-Host "[gapfill] ??????" $WL -ForegroundColor Red; exit 1 }
if (!(Test-Path $REPAIR)) { Write-Host "[gapfill] ?? repairkit.py?" $REPAIR -ForegroundColor Red; exit 1 }

# ???????
$SYMS = (Get-Content $WL -Raw).Split(',') | Where-Object { $_ -and $_.Trim() -ne "" } | ForEach-Object { $_.Trim() }
$SYMS_JOIN = ($SYMS -join ",")

# ? .NET Process ????? &/??????????
$argStr = @(
  "-u", $REPAIR,
  "--db", $DB,
  "--days", $Days,
  "--canary", $SYMS_JOIN,
  "--max-readers", $MaxReaders,
  "--max-writers", $MaxWriters
) -join " "

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $PY
$psi.Arguments = $argStr
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.CreateNoWindow = $true

$p = [System.Diagnostics.Process]::Start($psi)
while (-not $p.HasExited) { Start-Sleep -Milliseconds 200 }
exit $p.ExitCode
