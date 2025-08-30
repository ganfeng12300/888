def get_strategy_score(strategy_id):
    # 假设调用评分系统或数据库查询
    return 58.6 if strategy_id == "A1_BBandsBreak" else 77.5

def should_dry_run(score):
    return score < 60
