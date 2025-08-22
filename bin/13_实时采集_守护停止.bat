@echo off
setlocal
chcp 65001 >nul
set PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "D:\SQuant_Pro\bin\Stop_Guardian.ps1"
echo.
echo 已执行停止脚本，按任意键关闭窗口...
pause >nul
