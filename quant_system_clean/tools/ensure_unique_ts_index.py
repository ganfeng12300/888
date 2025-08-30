# -*- coding: utf-8 -*-
"""
tools/ensure_unique_ts_index.py �?为K线表补齐 UNIQUE(ts) 索引（可重复执行，幂等）
"""
import sqlite3
import re

DB = r"D:\quant_system_v2\data\market_data.db"

def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    pat = re.compile(r'^[A-Z0-9_]+_(?:1m|5m|15m|30m|1h|2h|4h|1d)$')
    created = 0
    for t in tables:
        if pat.match(t):
            idx = f'uidx_{t}_ts'
            cur.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{idx}" ON "{t}"(ts)')
            created += 1
    con.commit(); con.close()
    print(f"�?确保 UNIQUE(ts) 完成，处理表数：{created}")

if __name__ == "__main__":
    main()
