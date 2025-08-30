@echo off
chcp 65001 >nul
title 2币回测 · 彩色总进度 + 实导出门槛 · 终极版
cd /d "D:\quant_system_pro (3)\quant_system_pro"

:: 资金/风控/费用
set QS_EQUITY=100
set QS_LEVERAGE=5
set QS_POS_PCT=0.05
set QS_MAX_CONCURRENCY=5
set QS_EOD_CUTOFF=0.05
set QS_TAKER_FEE=0.0005
set QS_SLIPPAGE=0.0003
set QS_FUNDING_ON=1
set QS_AGG_BACKFILL=1

:: 数据库与输出
set QS_DB=D:\quant_system_v2\data\market_data.db
set OUTDIR=results

:: 2个币（可改）
set SYMBOLS=BTCUSDT ETHUSDT

python -u tools\run_2symbols_with_progress_enterprise.py ^
  --db "%QS_DB%" ^
  --symbols %SYMBOLS% ^
  --days 365 ^
  --topk 40 ^
  --outdir "%OUTDIR%" ^
  --strategies 8 ^
  --spa on --spa-alpha 0.05 ^
  --pbo on --pbo-bins 10 ^
  --impact-recheck on ^
  --wfo off --wfo-train 180 --wfo-test 30 --wfo-step 30 ^
  --tf-consistency on --tf-consistency-w 0.2

echo.
echo 任务已结束。按任意键关闭窗口…
pause >nul
