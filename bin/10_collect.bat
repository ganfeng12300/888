@echo off
setlocal
pushd D:\SQuant_Pro
set "PYTHONIOENCODING=utf-8"
python -u -m modules.data.collector --config ".\configs\system.ini" --risk ".\configs\risk.yml" --wl ".\configs\whitelist.txt" --max-readers 24 --bootstrap-days 0
popd
pause
