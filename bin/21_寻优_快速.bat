@echo off
python -u D:\SQuant_Pro\modules\optimize\optimizer.py D:\quant_system_v2\data\market_data.db D:\SQuant_Pro\reports\optimize_quick.csv BTCUSDT 1h
type D:\SQuant_Pro\reports\optimize_quick.csv
pause
