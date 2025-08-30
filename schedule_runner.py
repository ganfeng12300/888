import schedule
import time
import subprocess
import datetime

def run_daily_jobs():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[⏰ {now}] 🚀 启动每日自动调度任务...")

    try:
        print("📦 启动数据采集器...")
        subprocess.run(["python", "data/data_collector.py"], check=True)

        print("🔄 聚合周期数据...")
        subprocess.run(["python", "data/aggregator.py"], check=True)

        print("📊 执行回测与评分...")
        subprocess.run(["python", "backtest/run_all_backtests.py"], check=True)

        print("📈 更新策略评分...")
        subprocess.run(["python", "strategy_score_center.py"], check=True)

        print("✅ 全部任务完成。")

    except subprocess.CalledProcessError as e:
        print(f"❌ 错误：{e}")
        with open("D:/quant_system_pro (4)/logs/schedule_errors.txt", "a") as f:
            f.write(f"{now} 失败: {e}\n")

schedule.every().day.at("02:30").do(run_daily_jobs)

print("🕒 调度中心已启动，等待每日 02:30 自动运行...")
while True:
    schedule.run_pending()
    time.sleep(60)
