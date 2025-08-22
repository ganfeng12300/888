@echo off
setlocal
python -u "%~dp0..\modules\risk\daily_report.py" --config "%~dp0..\configs\system.ini" --risk "%~dp0..\configs\risk.yml"
pause
