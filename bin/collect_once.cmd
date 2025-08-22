@echo off
setlocal
pushd D:\SQuant_Pro
set "PYTHONIOENCODING=utf-8"
python -u -m modules.data.collector --config ".\configs\system.ini" --risk ".\configs\risk.yml" --wl ".\configs\whitelist.txt" --max-readers 24 --bootstrap-days 0 >> D:\SQuant_Pro\logs\collector_task.log 2>&1
popd
