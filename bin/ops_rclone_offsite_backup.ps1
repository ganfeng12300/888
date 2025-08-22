param(
  [string]$Src         = 'D:\SQuant_Pro\data\backup',        # 本地备份目录（.db 日备）
  [string]$RemoteCrypt = 'offsite_crypt:SQuant/backup',      # 已配置的 rclone 加密远端路径
  [int]   $KeepDays    = 30,                                 # 远端保留天数
  [string]$LogDir      = 'D:\SQuant_Pro\logs',
  [string]$BWLimit     = '',                                 # 例如 '8M' 限速；空=不限速
  [switch]$DryRun                                             # 试跑
)

$ErrorActionPreference='Stop'

# ===== 可选：首次一键创建加密远端（若你尚未配置），按需取消注释 =====
# 提前设置两个环境变量（一次性）：
#   setx SQ_CRYPT_PASSWORD "你的加密密码"
#   setx SQ_CRYPT_SALT     "你的盐(可空)"
# 然后取消下方注释，并把 $BaseRemote 改为你的底座远端（如 s3: / gdrive: 等）。
# $BaseRemote = 'offsite:'    # 举例：你预先在 rclone 里配置了一个 s3/gdrive 叫 offsite
# if(-not (rclone config dump | Select-String -SimpleMatch '[offsite_crypt]')){
#   $pw  = rclone obscure $env:SQ_CRYPT_PASSWORD
#   $pw2 = if($env:SQ_CRYPT_SALT){ rclone obscure $env:SQ_CRYPT_SALT } else { '' }
#   rclone config create offsite_crypt crypt remote "$BaseRemote/SQuant/backup" password "$pw" password2 "$pw2"
# }

# ===== 路径 & 日志 =====
if(-not (Test-Path $LogDir)){ New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$ts  = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $LogDir ("rclone_backup_{0}.log" -f $ts)

# ===== 组装 rclone 参数 =====
$common = @('--log-file', $log, '--log-level', 'INFO', '--fast-list', '--checkers', '8', '--transfers', '4')
if($BWLimit){ $common += @('--bwlimit', $BWLimit) }
if($DryRun) { $common += '--dry-run' }

# ===== 1) 上传/镜像（仅增量复制） =====
& rclone copy $Src $RemoteCrypt @common --size-only

# ===== 2) 保留策略：删除远端 30 天(可调)以前的历史 =====
# 注意：只会删超龄文件，保留目录结构
& rclone delete $RemoteCrypt "--min-age=${KeepDays}d" @common
& rclone rmdirs $RemoteCrypt --leave-root @common

Write-Host "[RCLONE] offsite encrypted backup done. Log at $log" -ForegroundColor Green
