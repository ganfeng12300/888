# ?? �Զ� Git �ύ���ȶԱ�����Զ���ļ�����

# 1. ���� Git ��Ϣ
$owner = "ganfeng12300"
$repo = "888"
$headers = @{ "User-Agent" = "PowerShell" }

# 2. �Զ���Ӻ��ύ
git add .
git commit -m "?? �Զ����� $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" 2>$null
git push

# 3. �����ļ����������� .git Ŀ¼��
$localFileCount = (Get-ChildItem -Recurse -File -Force | Where-Object { $_.FullName -notmatch '\\.git\\' }).Count

# 4. ��ȡԶ���ļ���
$response = Invoke-RestMethod -Uri "https://api.github.com/repos/$owner/$repo/git/trees/main?recursive=1" -Headers $headers
$remoteFileCount = ($response.tree | Where-Object { $_.type -eq "blob" }).Count

# 5. ����ԱȽ��
Write-Host "`n?? �����ļ�����" -NoNewline
Write-Host " $localFileCount" -ForegroundColor Cyan

Write-Host "??  Զ���ļ�����" -NoNewline
Write-Host " $remoteFileCount" -ForegroundColor Yellow

if ($localFileCount -eq $remoteFileCount) {
    Write-Host "`n? �ļ���һ�£����ͳɹ���" -ForegroundColor Green
} else {
    Write-Host "`n??  �ļ�����һ�£�����Ƿ���δ��ӻ� .gitignore ���Ե����ݡ�" -ForegroundColor Red
}
