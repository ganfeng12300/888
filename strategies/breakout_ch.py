# -*- coding: utf-8 -*-
import sqlite3, pandas as pd, numpy as np

def _load(con, symbol, tf):
    tbl=f"{symbol}_{tf}"
    try: df=pd.read_sql(f'SELECT * FROM "{tbl}" ORDER BY 1', con)
    except: return pd.DataFrame()
    if df.empty: return df
    if "timestamp" in df.columns: ts=pd.to_datetime(df["timestamp"], unit="s", utc=True)
    elif "ts" in df.columns:
        try: ts=pd.to_datetime(df["ts"], utc=True)
        except: ts=pd.to_datetime(df["ts"], unit="s", utc=True)
    else: return pd.DataFrame()
    return pd.DataFrame({"ts":ts, "close":pd.to_numeric(df["close"], errors='coerce')})

def generate_signals(db_path, symbols, params):
    n=int(params.get("n",55)); tf=params.get("tf","1h")
    con=sqlite3.connect(db_path); rows=[]
    for sym in symbols:
        df=_load(con,sym,tf)
        if df.empty: continue
        high=df["close"].rolling(n, min_periods=1).max()
        low =df["close"].rolling(n, min_periods=1).min()
        long =(df["close"]>high.shift(1)).astype(int)
        short=(df["close"]<low.shift(1)).astype(int)*-1
        # 只在已收盘K线上产生信号；空档用上一根的信号前推，避免“回填后重绘”
        sig=(long+short).replace(0, np.nan).ffill().fillna(0).clip(-1,1).astype(int)
        rows.append(pd.DataFrame({"ts":df["ts"],"symbol":sym,"signal":sig}))
    con.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ts","symbol","signal"])
