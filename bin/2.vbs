' D:\SQuant_Pro\bin\stop_guardian_hidden.vbs
Set W=CreateObject("WScript.Shell")
W.Run "powershell -NoProfile -ExecutionPolicy Bypass -File ""D:\SQuant_Pro\bin\Stop_Guardian.ps1""", 0, False
