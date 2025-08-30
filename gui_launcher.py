import tkinter as tk
import subprocess
import os

# 配置路径
PROJECT_DIR = r"D:\quant_system_pro (4)"
COMMANDS = {
    "启动实盘交易": f"python \"{PROJECT_DIR}\\launcher.py\"",
    "策略评分中心": f"python \"{PROJECT_DIR}\\scoring\\strategy_scorer.py\"",
    "订单状态监控": f"python \"{PROJECT_DIR}\\executor\\order_manager.py\"",
    "定时调度任务": f"python \"{PROJECT_DIR}\\scheduler\\run_scheduler.py\"",
    "一键启动系统": f"powershell -ExecutionPolicy Bypass -File \"{PROJECT_DIR}\\run_ai_all.ps1\""
}

def run_command(cmd):
    try:
        subprocess.Popen(cmd, shell=True)
    except Exception as e:
        print(f"❌ 启动失败: {e}")

root = tk.Tk()
root.title("千面猎手™ GUI 控制台")
root.geometry("380x350")
root.configure(bg="#111")

title = tk.Label(root, text="千面猎手™ 控制面板", font=("微软雅黑", 18, "bold"), bg="#111", fg="cyan")
title.pack(pady=20)

for name, cmd in COMMANDS.items():
    tk.Button(root, text=name, font=("微软雅黑", 12), width=25, height=2,
              bg="#222", fg="lime", activebackground="gray20",
              command=lambda c=cmd: run_command(c)).pack(pady=5)

root.mainloop()
