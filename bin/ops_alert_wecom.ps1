param(
  [Parameter(Mandatory=$true)][string]$Webhook,  # 你的企业微信机器人 URL
  [switch]$OnlyFail,                             # 仅在 FAIL 时推送
  [switch]$Markdown                              # 切换为 markdown 消息
)

$ErrorActionPreference='Stop'
$SelfCheck = 'D:\SQuant_Pro\bin\ops_self_check.ps1'

# 1) 读一次自检（JSON）
$json = & powershell -NoProfile -ExecutionPolicy Bypass -File $SelfCheck -Json -MaxMinutesStale 60
$sc = $json | ConvertFrom-Json

if($OnlyFail -and $sc.overallOk){ Write-Host "[WeCom] PASS 且 -OnlyFail=true，不推送"; exit 0 }

# 2) 组装文本
$status = if($sc.overallOk){'PASS'} else {'FAIL'}

# PID 规范化
$pidTxt = try{
  if($sc.guardian.pid -is [string]){ $sc.guardian.pid }
  elseif($null -ne $sc.guardian.pid -and $sc.guardian.pid.PSObject.Properties['value']){
    $sc.guardian.pid.value
  } else {
    ($sc.guardian.pid | Out-String -Width 200 -Stream | Select -First 1).Trim()
  }
}catch{ '' }

$lines = @(
  "SQuant 自检：$status",
  "Guardian: ok=$($sc.guardian.ok) pid=$pidTxt staleMin=$($sc.guardian.staleMin) stopFlag=$($sc.guardian.stopFlag)",
  "DB: ok=$($sc.db.ok) sizeMB=$($sc.db.sizeMB) integrity=$($sc.db.check)",
  "Backup: ok=$($sc.backup.ok) ageHours=$($sc.backup.ageHours) integrity=$($sc.backup.check) freshOk=$($sc.backup.freshOk)",
  "Disk: freeGB=$($sc.disk.freeGB) pct=$($sc.disk.pct)"
)
$text = $lines -join "`n"

# 3) 发送
try{
  if($Markdown){
    $body = @{ msgtype='markdown'; markdown=@{ content = ($text -replace '\n',"`n") } } | ConvertTo-Json -Depth 5
  }else{
    $body = @{ msgtype='text'; text=@{ content = $text; mentioned_list=@(); mentioned_mobile_list=@() } } | ConvertTo-Json -Depth 5
  }
  Invoke-RestMethod -Method Post -Uri $Webhook -Body $body -ContentType 'application/json; charset=utf-8'
  Write-Host "[WeCom] 已推送：$status"
}catch{
  Write-Warning "[WeCom] 推送失败：$($_.Exception.Message)"
  exit 1
}
