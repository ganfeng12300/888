import sys
import os

# 加入项目根路径，确保能识别 executor 包
sys.path.append(r"D:\quant_system_pro (4)")

from executor.real_executor_pro import execute_trade

# 测试信号（后续可由调度中心或策略模块自动生成）
test_signal = {
    "symbol": "BTCUSDT",
    "side": "Buy",
    "qty": 0.05,
    "strategy": "A1_BBandsBreak"
}

# 执行交易
execute_trade(test_signal)
