' D:\SQuant_Pro\bin\start_guardian_hidden.vbs
Set W=CreateObject("WScript.Shell")
W.Run "powershell -NoProfile -ExecutionPolicy Bypass -File ""D:\SQuant_Pro\bin\Start_Guardian.ps1"" -PythonHint ""C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe""", 0, False
