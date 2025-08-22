@echo off
setlocal
python -u "%~dp0..\modules\exec\run_paper.py" --config "%~dp0..\configs\system.ini" --risk "%~dp0..\configs\risk.yml"
pause
