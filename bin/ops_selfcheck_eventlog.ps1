param(
  [int]$MaxMinutesStale = 60,      # 守护日志“静默”阈值（分钟）
  [int]$RecentBackupHours = 36,    # 备份“新鲜”阈值（小时）
  [string]$LogName = 'Application',
  [string]$Source  = 'SQuantSelfCheck',
  [int]$EventIdPass = 7000,
  [int]$EventIdFail = 7001
)

$ErrorActionPreference='Stop'
$SelfCheck = 'D:\SQuant_Pro\bin\ops_self_check.ps1'

# 1) 执行自检（JSON 模式）
$json = & powershell -NoProfile -ExecutionPolicy Bypass `
          -File $SelfCheck -Json `
          -MaxMinutesStale $MaxMinutesStale -RecentBackupHours $RecentBackupHours
$sc = $json | ConvertFrom-Json

# 1.1) PID 正常化文本
$pidTxt = try{
  if($sc.guardian.pid -is [string]){ $sc.guardian.pid }
  elseif($null -ne $sc.guardian.pid -and $sc.guardian.pid.PSObject.Properties['value']){
    $sc.guardian.pid.value
  } else {
    ($sc.guardian.pid | Out-String -Width 200 -Stream | Select -First 1).Trim()
  }
}catch{ '' }

# 2) 确保事件源存在
try{
  if(-not [System.Diagnostics.EventLog]::SourceExists($Source)){
    New-EventLog -LogName $LogName -Source $Source
  }
}catch{
  Write-Warning ("无法创建事件源 {0}：{1}" -f $Source,$_.Exception.Message)
}

# 3) 组装消息
$statusText = if($sc.overallOk){'PASS'} else {'FAIL'}
$msg = @(
  "SQuant Self-Check: $statusText",
  "",
  "Guardian: ok=$($sc.guardian.ok) pid=$pidTxt staleMin=$($sc.guardian.staleMin) stopFlag=$($sc.guardian.stopFlag)",
  "DB:       ok=$($sc.db.ok) sizeMB=$($sc.db.sizeMB) integrity=$($sc.db.check)",
  "Backup:   ok=$($sc.backup.ok) ageHours=$($sc.backup.ageHours) integrity=$($sc.backup.check) freshOk=$($sc.backup.freshOk)",
  "Disk:     freeGB=$($sc.disk.freeGB) pct=$($sc.disk.pct)"
) -join [Environment]::NewLine

$etype = if($sc.overallOk){'Information'} else {'Error'}
$eid   = if($sc.overallOk){$EventIdPass} else {$EventIdFail}

# 4) 写事件日志
try{
  Write-EventLog -LogName $LogName -Source $Source -EntryType $etype -EventId $eid -Message $msg
}catch{
  Write-Warning ("写入事件日志失败：{0}" -f $_.Exception.Message)
}

# 控制台也打印
$fg = if($sc.overallOk){'Green'} else {'Red'}
Write-Host $msg -ForegroundColor $fg

# 以 PASS/FAIL 为退出码
if($sc.overallOk){ exit 0 } else { exit 1 }
