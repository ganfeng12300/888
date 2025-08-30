# -*- coding: utf-8 -*-
"""
patch_resolver_aliases.py —�?重写 backtest_pro._resolve_fn 以支�?A1~A8 等别名（机构级）
运行：cd /d D:\quant_system_pro && python patch_resolver_aliases.py
"""
import io, os, re, datetime

BASE = r"D:\quant_system_pro"
TARGET = os.path.join(BASE, "backtest", "backtest_pro.py")

NEW_RESOLVER = r'''
# === auto override: smarter resolver for A1-A8 aliases ===
from importlib import import_module as _imp
import re as _re

def _resolve_fn(strat_key):
    S = _imp("strategy.strategies_a1a8")
    key = str(strat_key)

    # 1) 直接模块属�?
    if hasattr(S, key):
        return getattr(S, key)

    # 2) 常见注册�?字典
    for k in ("STRATEGIES","STRATEGY_FUNCS","STRAT_TABLE","REGISTRY","ALIASES","ALIAS"):
        if hasattr(S, k):
            M = getattr(S, k)
            try:
                if key in M:
                    return M[key]
            except Exception:
                pass

    # 3) 标准 A1~A8 映射（含常见函数名）
    alias = {
        "A1": ["strat_bbands","strat_a1","A1","a1"],
        "A2": ["strat_breakout","strat_a2","A2","a2"],
        "A3": ["strat_rsi","strat_a3","A3","a3"],
        "A4": ["strat_macd","strat_a4","A4","a4"],
        "A5": ["strat_kelly","strat_a5","A5","a5"],
        "A6": ["strat_meanrev","strat_a6","A6","a6"],
        "A7": ["strat_trend","strat_a7","A7","a7"],
        "A8": ["strat_mix","strat_a8","A8","a8"],
        # 便捷别名（GPU�?
        "XGB": ["strat_xgb_gpu"],
        "LGBM": ["strat_lgbm_gpu"],
        "LSTM": ["strat_lstm_gpu"],
    }
    up = key.upper()
    if up in alias:
        for name in alias[up]:
            if hasattr(S, name):
                return getattr(S, name)
        # 兜底：A1 找到包含 bbands 的函数名即用
        if up == "A1":
            for name in dir(S):
                if "bbands" in name.lower():
                    return getattr(S, name)

    raise KeyError(f"Unknown strategy: {strat_key}")
'''

def main():
    if not os.path.exists(TARGET):
        print(f"[ERR] 未找�?{TARGET}")
        return
    s = io.open(TARGET, "r", encoding="utf-8").read()

    # 备份
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = TARGET + f".bak.{ts}"
    try:
        io.open(bak, "w", encoding="utf-8").write(s)
        print(f"[BACKUP] {TARGET} -> {bak}")
    except Exception as e:
        print(f"[WARN] 备份失败: {e}")

    # 用正则替换整�?def _resolve_fn(...) 函数体（若已有）
    pat = re.compile(r"(?ms)^def\s+_resolve_fn\s*\([^)]*\)\s*:\s*.*?(?=^\w|^#|\Z)")
    if pat.search(s):
        s2 = pat.sub(NEW_RESOLVER.strip()+"\n", s)
        changed = True
    else:
        # 没找到就插到文件顶部 import 之后
        # 尝试在第一次出�?'import numpy as np' 后插�?
        idx = s.find('import numpy as np')
        if idx != -1:
            insert_at = idx + len('import numpy as np')
            s2 = s[:insert_at] + "\n" + NEW_RESOLVER.strip() + "\n" + s[insert_at:]
        else:
            s2 = NEW_RESOLVER.strip() + "\n" + s
        changed = True

    if changed:
        io.open(TARGET, "w", encoding="utf-8").write(s2)
        print("[PATCH] 已写入新�?_resolve_fn（支�?A1~A8/GPU 别名�?)
    else:
        print("[SKIP] 无变�?)

if __name__ == "__main__":
    main()
