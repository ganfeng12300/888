# -*- coding: utf-8 -*-
# File: tools/patch_only_strategy.py
"""
�?backtest/backtest_pro.py 注入 --only-strategy 支持（A1..A8 映射�?MA/BOLL/ATR/REVERSAL/LGBM/XGB/LSTM/ENSEMBLE�?
用法�?
    python -u tools/patch_only_strategy.py
"""
import io, re
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
TARGET = ROOT / "backtest" / "backtest_pro.py"
assert TARGET.exists(), f"未找到：{TARGET}"

code = io.open(TARGET, 'r', encoding='utf-8', errors='ignore').read()

if "--only-strategy" in code and "STRATEGIES_TO_RUN" in code and "_STRAT_ALIASES" in code:
    print("�?检测到补丁已存在，无需重复注入�?)
    raise SystemExit(0)

backup = TARGET.with_suffix(".py.bak")
io.open(backup, 'w', encoding='utf-8').write(code)
print(f"已备份：{backup}")

aliases_block = r'''
# --- [PATCH only-strategy] begin ---
# A1..A8 精准映射至真实策略名
_STRAT_ALIASES = {
    "A1": "MA", "A2": "BOLL", "A3": "ATR", "A4": "REVERSAL",
    "A5": "LGBM", "A6": "XGB", "A7": "LSTM", "A8": "ENSEMBLE",
    "MA": "MA", "BOLL": "BOLL", "ATR": "ATR", "REVERSAL": "REVERSAL",
    "LGBM": "LGBM", "XGB": "XGB", "LSTM": "LSTM", "ENSEMBLE": "ENSEMBLE",
}
_ALL_STRATS = ["MA", "BOLL", "ATR", "REVERSAL", "LGBM", "XGB", "LSTM", "ENSEMBLE"]

def _normalize_strategy(tag: str) -> str:
    if not tag:
        return ""
    tag = tag.strip().upper()
    return _STRAT_ALIASES.get(tag, "")
# --- [PATCH only-strategy] end ---
'''

# 1) 尝试�?import 块后注入映射
m = re.search(r"(?ms)^(?:from\s+\S+?\s+import\s+.*\n|import\s+.*\n)+", code)
if m:
    code = code[:m.end()] + aliases_block + code[m.end():]
else:
    code = aliases_block + code

# 2) 注入 argparse 参数
add_arg = r'''
# --- [PATCH only-strategy arg] begin ---
try:
    parser.add_argument(
        "--only-strategy", dest="only_strategy", default="",
        help="仅运行指定策略：A1..A8 �?MA/BOLL/ATR/REVERSAL/LGBM/XGB/LSTM/ENSEMBLE"
    )
except Exception:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-strategy", dest="only_strategy", default="",
        help="仅运行指定策略：A1..A8 �?MA/BOLL/ATR/REVERSAL/LGBM/XGB/LSTM/ENSEMBLE"
    )
# --- [PATCH only-strategy arg] end ---
'''
if "argparse" in code:
    code = re.sub(r"(?ms)(import\s+argparse[^\n]*\n)", r"\1"+add_arg+"\n", code, count=1)
else:
    code += "\n" + add_arg

# 3) 注入选择逻辑（生�?STRATEGIES_TO_RUN�?
selector = r'''
# --- [PATCH only-strategy select] begin ---
try:
    args  # 若上游已 parse
except NameError:
    try:
        args = parser.parse_args()
    except Exception:
        class _A: only_strategy=""
        args = _A()

if getattr(args, "only_strategy", ""):
    _sel = _normalize_strategy(args.only_strategy)
    if not _sel:
        raise SystemExit(f"[ERROR] unknown --only-strategy: {args.only_strategy}")
    STRATEGIES_TO_RUN = [_sel]
else:
    STRATEGIES_TO_RUN = list(_ALL_STRATS)
# --- [PATCH only-strategy select] end ---
'''
code += "\n" + selector + "\n"

# 4) for 循环替换为使�?STRATEGIES_TO_RUN（两种常见写法）
code = re.sub(r"for\s+strat\s+in\s+ALL_STRATEGIES\s*:", "for strat in STRATEGIES_TO_RUN:", code)
code = re.sub(r"for\s+strat\s+in\s+strategies\s*:", "for strat in STRATEGIES_TO_RUN:", code)

io.open(TARGET, 'w', encoding='utf-8').write(code)
print(f"�?已注�?--only-strategy 至：{TARGET}")
