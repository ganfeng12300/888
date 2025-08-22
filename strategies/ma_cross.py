# -*- coding: utf-8 -*-
import sqlite3, pandas as pd
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
def _gen(con, sym, tf, fast, slow):
    df=_load(con, sym, tf)
    if df.empty: return pd.DataFrame(columns=["ts","symbol","signal"])
    s=df["close"].rolling(int(fast), min_periods=1).mean()
    l=df["close"].rolling(int(slow), min_periods=1).mean()
    sig=(s>l).astype(int)*2-1
    return pd.DataFrame({"ts":df["ts"],"symbol":sym,"signal":sig})
def generate_signals(db_path, symbols, params):
    fast=int(params.get("fast",9)); slow=int(params.get("slow",21)); tf=params.get("tf","1h")
    con=sqlite3.connect(db_path)
    parts=[_gen(con,s,tf,fast,slow) for s in symbols]
    con.close()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts","symbol","signal"])
