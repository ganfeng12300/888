@echo off
taskkill /FI "IMAGENAME eq python.exe" /F >NUL 2>&1
schtasks /end /tn SQuant_Collector_OnStart 2>NUL
schtasks /end /tn SQuant_Collector_Loop 2>NUL
echo tried to stop all collectors/tasks
pause
