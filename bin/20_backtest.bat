@echo off
setlocal
python -u "%~dp0..\modules\backtest\backtest.py" --config "%~dp0..\configs\system.ini" --risk "%~dp0..\configs\risk.yml" --wl "%~dp0..\configs\whitelist.txt"
pause
