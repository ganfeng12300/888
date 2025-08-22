param(
  [string]$Src = 'D:\SQuant_Pro\data\backup',
  [string]$Dst = 'P:\SQuant_Mirror\backup',   # 把 P: 换成你的 100G 固态盘符
  [int]$KeepDays = 30
)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Path $Dst -Force | Out-Null

# 1) 复制最近 KeepDays 的备份
$cut = (Get-Date).AddDays(-$KeepDays)
Get-ChildItem $Src -Filter *.db -File | Where-Object { $_.LastWriteTime -ge $cut } | ForEach-Object {
  $to = Join-Path $Dst $_.Name
  Copy-Item $_.FullName $to -Force
}

# 2) 清理镜像中过期文件
Get-ChildItem $Dst -Filter *.db -File | Where-Object { $_.LastWriteTime -lt $cut } | Remove-Item -Force

# 3) 简单可用性检查：最新文件存在且尺寸>0
$latest = Get-ChildItem $Dst -Filter *.db -File | Sort-Object LastWriteTime -Desc | Select-Object -First 1
if(-not $latest -or $latest.Length -le 0){ throw "镜像盘无有效备份" }
Write-Host "Mirror OK -> $($latest.FullName)  Size=$([math]::Round($latest.Length/1MB,1))MB"
