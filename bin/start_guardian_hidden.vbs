Set W = CreateObject("WScript.Shell")
cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File ""D:\SQuant_Pro\bin\Start_Guardian.ps1"" -PythonHint ""C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe"""
W.Run cmd, 0, False
