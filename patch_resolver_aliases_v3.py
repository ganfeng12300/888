# -*- coding: utf-8 -*-
"""
patch_resolver_aliases_v3.py —�?以“追加覆盖”的方式重写 _resolve_fn（支�?A1~A8 + 模糊匹配关键词）�?
运行：cd /d D:\quant_system_pro && python patch_resolver_aliases_v3.py
"""
import io, os, datetime

BASE   = r"D:\quant_system_pro"
TARGET = os.path.join(BASE, "backtest", "backtest_pro.py")

NEW_RESOLVER = r'''
# === auto override (append): robust resolver for A1-A8 with fuzzy matching ===
from importlib import import_module as _imp
import re as _re

def _resolve_fn(strat_key):
    S = _imp("strategy.strategies_a1a8")
    key = str(strat_key)

    # 1) 直接模块属�?
    if hasattr(S, key) and callable(getattr(S, key)):
        return getattr(S, key)

    # 2) 常见注册�?字典
    for k in ("STRATEGIES","STRATEGY_FUNCS","STRAT_TABLE","REGISTRY","ALIASES","ALIAS"):
        if hasattr(S, k):
            M = getattr(S, k)
            try:
                fn = M.get(key) if hasattr(M, "get") else (M[key] if key in M else None)
            except Exception:
                fn = None
            if callable(fn):
                return fn

    # 3) 收集 strat_* 候�?
    names = [n for n in dir(S) if n.startswith("strat_") and callable(getattr(S, n, None))]
    low   = {n.lower(): n for n in names}

    # 4) A1~A8 专属映射 + 关键�?
    alias_kw = {
        "A1": ["bbands","band"],
        "A2": ["break","don","channel","bo","brk","donch"],
        "A3": ["rsi"],
        "A4": ["macd"],
        "A5": ["kelly"],
        "A6": ["meanrev","mean_rev","revert","mr"],
        "A7": ["trend","ma_cross","cross","sma","ema","adx"],
        "A8": ["mix","combo","blend","stack"],
        "XGB": ["xgb"],
        "LGBM": ["lgb","lightgbm"],
        "LSTM": ["lstm","rnn"],
    }
    up = key.upper()

    # 4.1 直接�?A\d 形态匹配尾缀/变体
    m = _re.fullmatch(r"A(\d+)", up)
    if m:
        num = m.group(1)
        patterns = [
            f"strat_a{num}",
            f"strat_*_a{num}",  # 宽松：名字里�?a{num}
        ]
        for ln, orig in low.items():
            if ln == f"strat_a{num}" or ln.endswith(f"_a{num}") or ln.find(f"a{num}")>=0:
                return getattr(S, orig)

    # 4.2 关键词模糊匹�?
    if up in alias_kw:
        kws = alias_kw[up]
        for kw in kws:
            for ln, orig in low.items():
                if kw in ln:
                    return getattr(S, orig)

    # 5) A1 特殊兜底：任何包�?bbands 的策�?
    if up == "A1":
        for ln, orig in low.items():
            if "bbands" in ln:
                return getattr(S, orig)

    raise KeyError(f"Unknown strategy: {strat_key}")
'''

def main():
    if not os.path.exists(TARGET):
        print(f"[ERR] 未找�?{TARGET}")
        return
    s = io.open(TARGET, "r", encoding="utf-8").read()

    # 备份
    ts  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = TARGET + f".bak.{ts}"
    try:
        io.open(bak, "w", encoding="utf-8").write(s)
        print(f"[BACKUP] {TARGET} -> {bak}")
    except Exception as e:
        print(f"[WARN] 备份失败: {e}")

    # 直接“追加”新版本解析器（覆盖同名函数定义�?
    s2 = s.rstrip() + "\n\n" + NEW_RESOLVER.strip() + "\n"
    io.open(TARGET, "w", encoding="utf-8").write(s2)
    print("[PATCH] 追加覆盖�?_resolve_fn 已写入（支持 A1~A8 + 模糊匹配�?)

if __name__ == "__main__":
    main()
