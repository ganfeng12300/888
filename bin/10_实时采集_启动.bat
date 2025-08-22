@echo off
start "" /min cmd /c "python -u D:\SQuant_Pro\bin\collector_rt.py --db D:\quant_system_v2\data\market_data.db --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,APTUSDT --intervals 5m,15m,30m,1h,2h,4h,1d --max-readers 24 --http-parallel 8 --bootstrap-days 0 --loop 300 >> D:\SQuant_Pro\logs\collector_task.log 2>&1"
echo started in background (see D:\SQuant_Pro\logs\collector_task.log)
pause
