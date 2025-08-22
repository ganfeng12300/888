@echo off
taskkill /FI "IMAGENAME eq python.exe" /F >NUL 2>&1
echo if you used Task Scheduler, you can also run: schtasks /end
pause
