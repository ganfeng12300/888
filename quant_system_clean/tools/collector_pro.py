import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm
from rich.console import Console

# 自动添加 tools 目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(current_dir)
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

try:
    from collector_bin import Collector
except ImportError as e:
    print("无法导入 collector_bin 模块，请检查路径是否正确")
    raise e

console = Console()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="数据回补脚本")
    parser.add_argument("--start", type=str, required=True, help="起始时间 (例如 2023-08-01)")
    return parser.parse_args()

def main():
    args = parse_args()
    start_str = args.start
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    except ValueError:
        console.print(f"[red]起始时间格式错误：{start_str}，应为 YYYY-MM-DD 格式[/red]")
        return

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    timeframes = ["1m", "5m", "15m", "1h"]

    for s in symbols:
        for tf in timeframes:
            try:
                console.print(f"[cyan][Backfill][/cyan] {s} {tf} 从 {start_dt.strftime('%Y-%m-%d')} 开始回补")
                collector = Collector(symbol=s, timeframe=tf, start_time=start_dt)
                collector.run()
                time.sleep(0.3)
            except Exception as e:
                console.print(f"[red]错误处理 {s} {tf}：{e}[/red]")
                traceback.print_exc()

if __name__ == "__main__":
    main()
