import os
import json
from datetime import datetime

SCORE_SAVE_PATH = "D:/quant_system_pro (4)/strategy_scores/"
os.makedirs(SCORE_SAVE_PATH, exist_ok=True)

def score_strategy(results):
    """
    根据回测结果评估策略得分：
    - 年化收益、胜率、最大回撤三因子评分
    """
    profit = results.get("annual_return", 0)
    winrate = results.get("win_rate", 0)
    drawdown = results.get("max_drawdown", 1)

    score = (
        profit * 100 * 0.5 +
        winrate * 100 * 0.3 -
        drawdown * 100 * 0.2
    )
    return round(score, 2)

def save_score(strategy_name, results, symbol):
    score = score_strategy(results)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategy": strategy_name,
        "symbol": symbol,
        "score": score,
        "metrics": results
    }
    filename = f"{SCORE_SAVE_PATH}{symbol}_{strategy_name}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"✅ 策略评分已保存: {filename}")
    return score
