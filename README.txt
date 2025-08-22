SQuant_Pro scaffold generated.

Key folders:
- bin       : one-click launchers
- configs   : system.ini, risk.yml, exchanges.yml, whitelist.txt
- modules   : data, backtest, exec, risk (stubs)
- logs      : runtime logs
- reports   : HTML/CSV reports

First steps:
1) Run bin\00_audit.bat then bin\01_repair.bat
2) Edit configs\whitelist.txt and configs\exchanges.yml if needed
3) Try bin\10_collect.bat and bin\20_backtest.bat (stubs)

CAUTION: live trading is disabled by default.
