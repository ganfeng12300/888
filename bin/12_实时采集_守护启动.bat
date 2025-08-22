@echo off
setlocal
chcp 65001 >nul
set PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe

REM 可选：给 collector 传参（示例：排除 1m，显式指定区间）
set COLLECTOR_ARGS=--intervals 5m,15m,30m,1h,2h,4h,1d

REM 如需强制用特定 Python，可在此指定；留空则自动探测
set PY_HINT=

REM 用新窗口运行，便于查看守护输出；守护自身有日志（logs\guardian_*.log）
start "SQuant Guardian" "%PS_EXE%" -NoProfile -ExecutionPolicy Bypass ^
  -File "D:\SQuant_Pro\bin\Start_Guardian.ps1" ^
  -PythonHint "%PY_HINT%" -CollectorArgs "%COLLECTOR_ARGS%"

echo.
echo 已发送启动指令。守护会持续运行并写入 logs\guardian_*.log
echo 如需停止，运行 13_实时采集_守护停止.bat
echo.
pause
