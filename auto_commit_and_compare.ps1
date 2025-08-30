# ?? 自动 Git 提交并比对本地与远程文件数量

# 1. 设置 Git 信息
$owner = "ganfeng12300"
$repo = "888"
$headers = @{ "User-Agent" = "PowerShell" }

# 2. 自动添加和提交
git add .
git commit -m "?? 自动更新 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" 2>$null
git push

# 3. 本地文件数量（忽略 .git 目录）
$localFileCount = (Get-ChildItem -Recurse -File -Force | Where-Object { $_.FullName -notmatch '\\.git\\' }).Count

# 4. 获取远程文件树
$response = Invoke-RestMethod -Uri "https://api.github.com/repos/$owner/$repo/git/trees/main?recursive=1" -Headers $headers
$remoteFileCount = ($response.tree | Where-Object { $_.type -eq "blob" }).Count

# 5. 输出对比结果
Write-Host "`n?? 本地文件数：" -NoNewline
Write-Host " $localFileCount" -ForegroundColor Cyan

Write-Host "??  远程文件数：" -NoNewline
Write-Host " $remoteFileCount" -ForegroundColor Yellow

if ($localFileCount -eq $remoteFileCount) {
    Write-Host "`n? 文件数一致，推送成功！" -ForegroundColor Green
} else {
    Write-Host "`n??  文件数不一致，检查是否有未添加或 .gitignore 忽略的内容。" -ForegroundColor Red
}
