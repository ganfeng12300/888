@echo off
cd /d D:\SQuant_Pro
start "" "C:\Windows\System32\cmd.exe" /c ""C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe" "D:\SQuant_Pro\dashboard\server.py""
"C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe" "D:\SQuant_Pro\trading\engine_paper.py"
