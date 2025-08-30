# -*- coding: utf-8 -*-
"""
patch_resolver_aliases_v2.py —�?更健壮的 A1~A8 策略解析（含模糊匹配关键词）
运行：cd /d D:\quant_system_pro && python patch_resolver_aliases_v2.py
"""
import io, os, re, datetime

BASE   = r"D:\quant_system_pro"
TARGET = os.path.join(BASE, "backtest", "backtest_pro.py")

NEW_RESOLVER = r'''
# === auto override: robust resolver for A1-A8 (with fuzzy matching) ===
from importlib import import_module as _imp
import re as _re
from inspect import isfunction as _isfunc

def _resolve_fn(strat_key):
    S = _imp("strategy.strategies_a1a8")
    key = str(strat_key)

    # 1) 直接模块属�?
    if hasattr(S, key):
        fn = getattr(S, key)
        if callable(fn): return fn

    # 2) 常见注册�?字典
    for k in ("STRATEGIES","STRATEGY_FUNCS","STRAT_TABLE","REGISTRY","ALIASES","ALIAS"):
        if hasattr(S, k):
            M = getattr(S, k)
            try:
                if key in M:
                    fn = M[key]
                    if callable(fn): return fn
            except Exception:
                pass

    # 收集候选函数名
    names = [n for n in dir(S) if n.startswith("strat_") and callable(getattr(S,n, None))]
    lowmap = {n.lower(): n for n in names}

    # 3) 标准映射关键�?
    alias_keywords = {
        "A1": ["bbands","band"],
        "A2": ["break","don","channel","bo"],  # breakout/donchian/channel/bo
        "A3": ["rsi"],
        "A4": ["macd"],
        "A5": ["kelly"],
        "A6": ["meanrev","mean_re","revert","mr"],
        "A7": ["trend","ma_cross","sma","ema","adx"],
        "A8": ["mix","combo","blend","stack"],
        "XGB": ["xgb"],
        "LGBM": ["lgb","lightgbm"],
        "LSTM": ["lstm","rnn"],
    }

    up = key.upper()
    num = None
    m = _re.fullmatch(r"A(\d+)", up)
    if m:
        num = m.group(1)

    # 4) 先尝�?A\d 命名/后缀/下划线变�?
    if num:
        patt = [
            rf"^strat_.*(?:^|_)a{num}$",   # strat_xxx_a2
            rf"^strat_.*a{num}$",         # strat_xxxa2
            rf"^strat_a{num}$",           # strat_a2
            rf"^a{num}$"                  # a2
        ]
        for p in patt:
            for ln in list(lowmap.keys()):
                if _re.search(p, ln):
                    return getattr(S, lowmap[ln])

    # 5) 再按关键词模糊匹配（按优先级顺序�?
    if up in alias_keywords:
        kws = alias_keywords[up]
        for kw in kws:
            for ln in list(lowmap.keys()):
                if kw in ln:
                    return getattr(S, lowmap[ln])

    # 6) A1 特殊兜底：找�?bbands 的任意策�?
    if up == "A1":
        for ln in list(lowmap.keys()):
            if "bbands" in ln:
                return getattr(S, lowmap[ln])

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

    # 用正则替换整�?def _resolve_fn(...)
    pat = re.compile(r"(?ms)^def\s+_resolve_fn\s*\([^)]*\)\s*:\s*.*?(?=^\w|^#|\Z)")
    if pat.search(s):
        s2 = pat.sub(NEW_RESOLVER.strip()+"\n", s)
    else:
        # 插入到首�?import numpy as np 之后
        idx = s.find('import numpy as np')
        if idx != -1:
            insert_at = idx + len('import numpy as np')
            s2 = s[:insert_at] + "\n" + NEW_RESOLVER.strip() + "\n" + s[insert_at:]
        else:
            s2 = NEW_RESOLVER.strip() + "\n" + s

    io.open(TARGET, "w", encoding="utf-8").write(s2)
    print("[PATCH] _resolve_fn 已升级为 A1~A8 模糊解析�?)

if __name__ == "__main__":
    main()
