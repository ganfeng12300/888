@echo off
set DB=D:\quant_system_v2\data\market_data.db
set OUT=D:\SQuant_Pro\reports\backtest_quick.html
python -u D:\SQuant_Pro\modules\backtest\backtester.py %DB% %OUT% BTCUSDT,ETHUSDT,SOLUSDT
if exist "%OUT%" start "" "%OUT%"
echo.
echo [OK] ????????%OUT%
pause
