# -*- coding: utf-8 -*-
import os, sys, json, time, hashlib, math, csv
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "bin" else Path.cwd()
DATA = ROOT / "data"
SCORES_DIR = DATA / "scores"
ENSEMBLE_DIR = DATA / "ensemble"
HIST_DIR = ENSEMBLE_DIR / "weights_history"
HIST_DIR.mkdir(parents=True, exist_ok=True)

def _now_utc_str():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _load_yaml_like(path):
    # Try PyYAML; fallback to naive "key: value" parser
    cfg = {}
    if not path.exists():
        return cfg
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        section = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): 
                    continue
                if ":" in s:
                    k, v = s.split(":", 1)
                    cfg[k.strip()] = v.strip()
        return cfg

def _read_cfg():
    # Defaults for A plan
    defaults = {
        "A1": 0.35, "B1": 0.60, "C1": 0.50, "N1": 20,   # strict
        "A2": 0.20, "B2": 0.80, "C2": 0.30, "N2": 5,    # loose
        "KMIN": 3,
        "W_FLOOR": 0.05,
        "W_CAP": 0.60,
        "SMOOTH_BETA": 0.5,
        "INERTIA_DELTA": 0.15,   # L1 threshold; keep old if below
        "TTL_HOURS": 24,
        "FALLBACK_TOPK": 5
    }
    # governance.yaml may live in root or configs/
    cand = [ROOT / "governance.yaml", ROOT / "configs" / "governance.yaml"]
    cfg_raw = {}
    for p in cand:
        cfg_raw = _load_yaml_like(p)
        if cfg_raw: break

    # Extract with fallbacks
    def pick(*keys, default=None):
        for k in keys:
            if isinstance(cfg_raw, dict) and k in cfg_raw:
                try:
                    return float(cfg_raw[k])
                except Exception:
                    try:
                        return int(cfg_raw[k])
                    except Exception:
                        pass
        return default

    out = dict(defaults)
    out["A1"] = pick("spa_alpha_strict","A1", default=defaults["A1"])
    out["B1"] = pick("pbo_max_strict","B1", default=defaults["B1"])
    out["C1"] = pick("dsr_min_strict","C1", default=defaults["C1"])
    out["N1"] = int(pick("trades_min_strict","N1", default=defaults["N1"]))
    out["A2"] = pick("spa_alpha_loose","A2", default=defaults["A2"])
    out["B2"] = pick("pbo_max_loose","B2", default=defaults["B2"])
    out["C2"] = pick("dsr_min_loose","C2", default=defaults["C2"])
    out["N2"] = int(pick("trades_min_loose","N2", default=defaults["N2"]))
    out["KMIN"] = int(pick("kmin","KMIN", default=defaults["KMIN"]))
    out["W_FLOOR"] = float(pick("w_floor","W_FLOOR", default=defaults["W_FLOOR"]))
    out["W_CAP"] = float(pick("w_cap","W_CAP", default=defaults["W_CAP"]))
    out["SMOOTH_BETA"] = float(pick("smooth_beta","SMOOTH_BETA", default=defaults["SMOOTH_BETA"]))
    out["INERTIA_DELTA"] = float(pick("inertia_delta","INERTIA_DELTA", default=defaults["INERTIA_DELTA"]))
    out["TTL_HOURS"] = int(pick("ttl_hours","TTL_HOURS", default=defaults["TTL_HOURS"]))
    out["FALLBACK_TOPK"] = int(pick("fallback_topk","FALLBACK_TOPK", default=defaults["FALLBACK_TOPK"]))
    return out

REQ_COLS = [
    "symbol","tf","strategy","trades","ret","sharpe","sortino","mdd",
    "calmar","ulcer","upi","dsr","spa_alpha","pbo","cagr","vol","turnover","asof_utc","score"
]

def _load_scores_df():
    latest = SCORES_DIR / "latest.csv"
    cand = []
    if latest.exists():
        cand.append(latest)
    for p in sorted(SCORES_DIR.glob("strategy_scores_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
        cand.append(p)
    if not cand:
        raise FileNotFoundError("scores csv not found in data/scores")
    for p in cand:
        try:
            df = pd.read_csv(p)
            # normalize columns
            for c in REQ_COLS:
                if c not in df.columns:
                    df[c] = pd.NA
            num_cols = ["trades","ret","sharpe","sortino","mdd","calmar","ulcer","upi","dsr","spa_alpha","pbo","cagr","vol","turnover","score"]
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            # enforce types
            if "tf" not in df.columns:
                df["tf"] = ""
            df["symbol"] = df["symbol"].astype(str)
            df["strategy"] = df["strategy"].astype(str)
            # key for alignment
            df["key"] = df["symbol"].astype(str)+"|"+df["tf"].astype(str)+"|"+df["strategy"].astype(str)
            return df, p
        except Exception:
            continue
    raise RuntimeError("no readable scores csv")

def _filter_adaptive(df, cfg):
    def mask(df, A, B, C, N):
        m = pd.Series(True, index=df.index)
        if "spa_alpha" in df: m &= (df["spa_alpha"].fillna(0) >= A)
        if "pbo" in df:       m &= (df["pbo"].fillna(1.0) <= B)
        if "dsr" in df:       m &= (df["dsr"].fillna(0) >= C)
        if "trades" in df:    m &= (df["trades"].fillna(0) >= N)
        return df[m]

    df1 = mask(df, cfg["A1"], cfg["B1"], cfg["C1"], cfg["N1"])
    if len(df1) >= cfg["KMIN"]:
        return df1, "strict"
    df2 = mask(df, cfg["A2"], cfg["B2"], cfg["C2"], cfg["N2"])
    if len(df2) >= cfg["KMIN"]:
        return df2, "loose"

    # fallback: Top-K by sortino->calmar->sharpe->score
    dfx = df.copy()
    metric_order = [("sortino", True), ("calmar", True), ("sharpe", True), ("score", True)]
    for col, desc in metric_order:
        if col not in dfx.columns:
            dfx[col] = pd.NA
    dfx["__rank"] = (
        dfx["sortino"].fillna(-1e9) * 1e9 +
        dfx["calmar"].fillna(-1e8) * 1e6 +
        dfx["sharpe"].fillna(-1e7) * 1e3 +
        dfx["score"].fillna(-1e6)
    )
    dft = dfx.sort_values("__rank", ascending=False).head(max(cfg["FALLBACK_TOPK"], cfg["KMIN"]))
    return dft.drop(columns=["__rank"]), "fallback"

def _load_last_weights():
    cur = ENSEMBLE_DIR / "ensemble_weights.csv"
    if cur.exists():
        try:
            df = pd.read_csv(cur)
            if "key" not in df.columns and {"symbol","tf","strategy"}.issubset(df.columns):
                df["key"] = df["symbol"].astype(str)+"|"+df["tf"].astype(str)+"|"+df["strategy"].astype(str)
            return df
        except Exception:
            return None
    return None

def _l1_diff(a, b):
    cols = ["key","weight"]
    ta = a[cols].set_index("key")["weight"]
    tb = b[cols].set_index("key")["weight"]
    keys = set(ta.index) | set(tb.index)
    s = 0.0
    for k in keys:
        s += abs(float(ta.get(k,0.0)) - float(tb.get(k,0.0)))
    return s

def _ema_smooth(new_df, prev_df, beta):
    if prev_df is None or prev_df.empty:
        return new_df
    ta = new_df[["key","weight"]].set_index("key")["weight"]
    tb = prev_df[["key","weight"]].set_index("key")["weight"]
    keys = sorted(set(ta.index)|set(tb.index))
    out = []
    for k in keys:
        na = float(ta.get(k,0.0))
        nb = float(tb.get(k,0.0))
        out.append((k, beta*na + (1.0-beta)*nb))
    sm = pd.DataFrame(out, columns=["key","weight"])
    # split back to cols if possible
    parts = sm["key"].str.split("|", expand=True)
    sm["symbol"] = parts[0]; sm["tf"] = parts[1] if parts.shape[1]>1 else ""; sm["strategy"]=parts[2] if parts.shape[1]>2 else ""
    # renormalize
    s = sm["weight"].sum()
    if s <= 0: 
        sm["weight"] = 1.0/len(sm)
    else:
        sm["weight"] = sm["weight"]/s
    return sm

def _apply_floor_cap(df, floor_v, cap_v):
    n = len(df)
    if n == 0: return df
    w = df["weight"].values.astype(float)
    if w.sum() <= 0: 
        w[:] = 1.0/n
    # floor
    w = [max(floor_v, float(x)) for x in w]
    s = sum(w)
    w = [x/s for x in w]
    # cap then renorm
    w = [min(cap_v, float(x)) for x in w]
    s = sum(w)
    if s <= 0: 
        w = [1.0/n]*n
    else:
        w = [x/s for x in w]
    df = df.copy()
    df["weight"] = w
    return df

def _save_weights(df, reason=""):
    ts = _now_utc_str()
    out = df.copy()
    out["ts_utc"] = ts
    out = out[["symbol","tf","strategy","key","weight","ts_utc"]]
    cur = ENSEMBLE_DIR / "ensemble_weights.csv"
    hist = HIST_DIR / f"weights_{ts}.csv"
    out.to_csv(cur, index=False)
    out.to_csv(hist, index=False)
    print(f"[OK] ensemble_weights saved ({reason}). items={len(out)}")

def _within_ttl(path, hours):
    if not path.exists(): return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age.total_seconds() <= hours*3600

def main():
    cfg = _read_cfg()
    try:
        df, src = _load_scores_df()
    except Exception as e:
        # No scores at all -> rollback to last valid weights if within TTL
        last = ENSEMBLE_DIR / "ensemble_weights.csv"
        if _within_ttl(last, cfg["TTL_HOURS"]):
            print("[WARN] scores not found; keep existing ensemble within TTL")
            sys.exit(0)
        else:
            print("[FATAL] scores not found and no valid ensemble to rollback")
            sys.exit(1)

    # raw -> nonnegative weight seed
    if "score" in df.columns and df["score"].notna().any():
        seed = df["score"].clip(lower=0.0).fillna(0.0)
        if seed.sum() <= 0:
            seed = pd.Series([1.0]*len(df), index=df.index)
    else:
        # fallback seed: exp of sortino (or sharpe)
        base = df["sortino"].fillna(df["sharpe"]).fillna(0.0)
        seed = (base - base.min()).clip(lower=0.0) + 1e-9

    df["__seed"] = seed
    # filtering
    sel, stage = _filter_adaptive(df, cfg)
    sel = sel.copy()
    sel["key"] = sel["symbol"].astype(str)+"|"+sel["tf"].astype(str)+"|"+sel["strategy"].astype(str)
    # weights from seed
    base = sel["__seed"].astype(float).values
    if base.sum() <= 0:
        base[:] = 1.0/len(base)
    w = base / base.sum()
    out = sel[["symbol","tf","strategy","key"]].copy()
    out["weight"] = w

    # floor/cap + smooth + inertia
    out = _apply_floor_cap(out, cfg["W_FLOOR"], cfg["W_CAP"])
    prev = _load_last_weights()
    out_sm = _ema_smooth(out, prev, cfg["SMOOTH_BETA"])
    out_sm = _apply_floor_cap(out_sm, cfg["W_FLOOR"], cfg["W_CAP"])

    if prev is not None and not prev.empty:
        diff = _l1_diff(out_sm, prev)
        if diff < cfg["INERTIA_DELTA"]:
            print(f"[INFO] inertia keep: L1={diff:.4f} < {cfg['INERTIA_DELTA']}")
            _save_weights(prev, reason="inertia")
            return

    if len(out_sm) == 0:
        # rollback if within TTL
        last = ENSEMBLE_DIR / "ensemble_weights.csv"
        if _within_ttl(last, cfg["TTL_HOURS"]):
            print("[WARN] empty selection; rollback to previous within TTL")
            return
        else:
            print("[FATAL] empty selection and no rollback available")
            sys.exit(2)

    _save_weights(out_sm, reason=stage)

if __name__ == "__main__":
    main()
