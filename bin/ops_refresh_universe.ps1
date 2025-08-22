param([string]$Py="C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe")
$ErrorActionPreference='Stop'
$ROOT='D:\SQuant_Pro'
$BIN = Join-Path $ROOT 'bin'
$OUT = Join-Path $ROOT 'data\universe\symbols_latest.json'
& $Py (Join-Path $BIN 'update_universe.py') | Write-Host
if(!(Test-Path $OUT)){ throw "universe json missing: $OUT" }
$u   = Get-Content $OUT -Raw | ConvertFrom-Json
$UNI = $u.universe
$dir = Join-Path $ROOT 'data\universe'
@('T1','T2','T3','all','Top50') | ForEach-Object {
  $p = Join-Path $dir ("{0}.txt" -f $_)
  $UNI.$_ | Out-File -FilePath $p -Encoding ascii -Width 9999
  Write-Host "wrote $p ($(($UNI.$_).Count))"
}
