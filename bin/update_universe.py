# -*- coding: utf-8 -*-
# Binance USDT-M PERP Top50（带黏性 45/60）+ 分层 T1/T2/T3
import os, json, time, ssl, urllib.request

OUTDIR = r"D:\SQuant_Pro\data\universe"
os.makedirs(OUTDIR, exist_ok=True)

EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
TICKER_24HR   = "https://fapi.binance.com/fapi/v1/ticker/24hr"
STICKY_IN  = 45
STICKY_OUT = 60

def get(url, timeout=20):
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))

def fetch_syms():
    info = get(EXCHANGE_INFO)
    syms = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":         continue
        if s.get("contractType") != "PERPETUAL": continue
        if s.get("quoteAsset")   != "USDT":      continue
        syms.append(s["symbol"])
    return syms

def fetch_24h():
    data = get(TICKER_24HR)
    m = {}
    for x in data:
        s = x.get("symbol")
        if not s: continue
        try:
            qv = float(x.get("quoteVolume", "0"))
        except:
            qv = 0.0
        m[s] = {"qv": qv, "count": x.get("count")}
    return m

def tiering(syms, m24):
    rows = [(s, m24.get(s, {}).get("qv", 0.0)) for s in syms]
    rows.sort(key=lambda z: z[1], reverse=True)
    T1, T2, T3 = [], [], []
    for i, (s, _) in enumerate(rows):
        if   i < 30:  T1.append(s)
        elif i < 100: T2.append(s)
        elif i < 300: T3.append(s)
    top50 = [x[0] for x in rows[:50]]
    return T1, T2, T3, top50, rows

def sticky_select(latest50, prev50):
    if not prev50: return latest50
    rank = {s:i+1 for i,s in enumerate(latest50)}
    out = []
    # 保留旧 Top50 中排名 <= STICKY_OUT 的
    for s in prev50:
        if rank.get(s, 9999) <= STICKY_OUT:
            out.append(s)
    # 加入新入榜且排名 <= STICKY_IN 的
    for s in latest50:
        if s not in out and rank.get(s, 9999) <= STICKY_IN:
            out.append(s)
    # 不足 50 个则按最新排名补齐
    for s in latest50:
        if len(out) >= 50: break
        if s not in out:
            out.append(s)
    return out[:50]

def main():
    syms = fetch_syms()
    m24  = fetch_24h()
    T1, T2, T3, top50_raw, rows = tiering(syms, m24)
    latest = os.path.join(OUTDIR, "symbols_latest.json")
    prev50 = []
    if os.path.exists(latest):
        try:
            prev = json.load(open(latest, "r", encoding="utf-8"))
            prev50 = prev.get("universe", {}).get("Top50", [])
        except:
            prev50 = []
    top50 = sticky_select(top50_raw, prev50)
    out = {
        "lastUpdated": int(time.time()),
        "universe": {"T1": T1, "T2": T2, "T3": T3, "Top50": top50, "all": T1 + T2 + T3},
        "meta": {"totalUSDT_PERP": len(syms), "ranked": len(rows)}
    }
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    snap = os.path.join(OUTDIR, f"symbols_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(snap, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("universe ok:", latest, "| T1/T2/T3", len(T1), len(T2), len(T3), "| Top50", len(top50))

if __name__ == "__main__":
    main()
