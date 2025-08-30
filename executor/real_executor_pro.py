from executor.order_logger import log_order
from executor.result_checker import check_result
from executor.risk_guard_adv import should_dry_run, get_strategy_score
from executor.bybit_api import place_order

def execute_trade(signal):
    score = get_strategy_score(signal['strategy'])
    if should_dry_run(score):
        log_order(signal, score, dry_run=True, reason="score_low")
        print(f"🧪 Dry-Run 模拟下单: {signal}")
        return

    try:
        result = place_order(**signal)
        check_result(result)
        log_order(signal, score, dry_run=False, result=result)
        print(f"✅ 实盘下单成功: {result}")
    except Exception as e:
        print(f"❌ 实盘下单失败: {e}")
        log_order(signal, score, dry_run=False, error=str(e))
