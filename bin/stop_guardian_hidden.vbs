Set W = CreateObject("WScript.Shell")
cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File ""D:\SQuant_Pro\bin\Stop_Guardian.ps1"""
W.Run cmd, 0, False
