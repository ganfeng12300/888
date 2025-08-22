param(
  [Parameter(Mandatory=$true)][string]$SendKey,  # Server酱 Turbo 的 SCTKEY
  [switch]$OnlyFail
)

$ErrorActionPreference='Stop'
$SelfCheck = 'D:\SQuant_Pro\bin\ops_self_check.ps1'

$json = & powershell -NoProfile -ExecutionPolicy Bypass -File $SelfCheck -Json -MaxMinutesStale 60
$sc = $json | ConvertFrom-Json

if($OnlyFail -and $sc.overallOk){ Write-Host "[WxPersonal] PASS 且 -OnlyFail=true，不推送"; exit 0 }

$status = if($sc.overallOk){'PASS'} else {'FAIL'}

$pidTxt = try{
  if($sc.guardian.pid -is [string]){ $sc.guardian.pid }
  elseif($null -ne $sc.guardian.pid -and $sc.guardian.pid.PSObject.Properties['value']){
    $sc.guardian.pid.value
  } else {
    ($sc.guardian.pid | Out-String -Width 200 -Stream | Select -First 1).Trim()
  }
}catch{ '' }

$title = "[SQuant 自检] $status"
$desp  = @"
**Guardian** ok=$($sc.guardian.ok) pid=$pidTxt staleMin=$($sc.guardian.staleMin) stopFlag=$($sc.guardian.stopFlag)
**DB** ok=$($sc.db.ok) sizeMB=$($sc.db.sizeMB) integrity=$($sc.db.check)
**Backup** ok=$($sc.backup.ok) ageHours=$($sc.backup.ageHours) integrity=$($sc.backup.check) freshOk=$($sc.backup.freshOk)
**Disk** freeGB=$($sc.disk.freeGB) pct=$($sc.disk.pct)
"@

# Server酱 Turbo API
$uri = "https://sctapi.ftqq.com/$SendKey.send"
try{
  Invoke-RestMethod -Method Post -Uri $uri -Body @{ title=$title; desp=$desp } -ContentType 'application/x-www-form-urlencoded; charset=utf-8'
  Write-Host "[WxPersonal] 已推送：$status"
}catch{
  Write-Warning "[WxPersonal] 推送失败：$($_.Exception.Message)"
  exit 1
}
