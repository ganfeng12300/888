param(
  [string]$DbPath    = 'D:\quant_system_v2\data\market_data.db',
  [string]$BackupDir = 'D:\SQuant_Pro\data\backup',
  [int]$KeepCount    = 0,      # 可选：按份数保留（0=不按份数）
  [int]$KeepDays     = 5,      # ✔ 新增：按天保留（N 天内的备份保留）
  [switch]$CheckpointTruncate
)
$ErrorActionPreference='Stop'
if(-not(Test-Path $BackupDir)){ New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null }

$ts  = Get-Date -Format 'yyyyMMdd_HHmmss'
$dst = Join-Path $BackupDir ("market_data_{0}.db" -f $ts)

# 在线一致性备份
$py = @"
import sqlite3
src=r'$DbPath'; dst=r'$dst'
con_src = sqlite3.connect(src, timeout=60)
con_dst = sqlite3.connect(dst)
with con_dst: con_src.backup(con_dst)
con_dst.close(); con_src.close()
print('backup ok ->', dst)
"@
$tf = "$env:TEMP\sq_backup_inline.py"
Set-Content -Path $tf -Value $py -Encoding ascii
& "$env:LocalAppData\Programs\Python\Python39\python.exe" $tf

# 可选：截断 WAL
if($CheckpointTruncate){
  $py2 = @"
import sqlite3
con = sqlite3.connect(r'$DbPath', timeout=60)
con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
con.close()
print('checkpoint truncate done')
"@
  Set-Content -Path $tf -Value $py2 -Encoding ascii
  & "$env:LocalAppData\Programs\Python\Python39\python.exe" $tf
}

# ==== 清理策略 ====
$files = Get-ChildItem $BackupDir -Filter 'market_data_*.db' | Sort LastWriteTime -Desc

# 1) 按天清理（KeepDays>0 生效）
if($KeepDays -gt 0){
  $cut = (Get-Date).AddDays(-$KeepDays)
  $old = $files | Where-Object { $_.LastWriteTime -lt $cut }
  foreach($f in $old){ try{ Remove-Item $f.FullName -Force }catch{} }
}

# 2) 再按份数兜底（KeepCount>0 生效；会在按天之后执行）
if($KeepCount -gt 0){
  $files = Get-ChildItem $BackupDir -Filter 'market_data_*.db' | Sort LastWriteTime -Desc
  $extra = $files | Select-Object -Skip $KeepCount
  foreach($f in $extra){ try{ Remove-Item $f.FullName -Force }catch{} }
}

$left = (Get-ChildItem $BackupDir -Filter 'market_data_*.db').Count
Write-Host "DONE: latest=$dst | kept=$left (KeepDays=$KeepDays, KeepCount=$KeepCount)" -ForegroundColor Green
