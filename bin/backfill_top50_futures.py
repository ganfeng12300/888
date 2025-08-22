# -*- coding: utf-8 -*-
# 回填 Top50（收盘时间做主键）
import os, ssl, json, time, sqlite3, urllib.request
ROOT=r"D:\SQuant_Pro"; DB=r"D:\quant_system_v2\data\market_data.db"
UNI=os.path.join(ROOT,'data','universe','Top50.txt')
BASE="https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1h&limit=1000"
def get(url):
    ctx=ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=20, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))
def rebuild_1h(con, sym, rows):
    tbl=f"{sym}_1h"
    con.execute(f'DROP TABLE IF EXISTS "{tbl}"')
    con.execute(f'CREATE TABLE "{tbl}" (timestamp INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL)')
    data=[(int(r[6]//1000), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]  # 用 closeTime
    con.executemany(f'INSERT OR REPLACE INTO "{tbl}" (timestamp,open,high,low,close,volume) VALUES (?,?,?,?,?,?)', data)
    con.commit()
def main():
    syms=[x.strip() for x in open(UNI,'r',encoding='ascii').read().splitlines() if x.strip()]
    con=sqlite3.connect(DB)
    for s in syms:
        try:
            rows=get(BASE.format(sym=s))
            if isinstance(rows,list) and rows:
                rebuild_1h(con, s, rows)
                print("ok", s, "1h", len(rows))
        except Exception as e:
            print("err", s, e)
        time.sleep(0.15)
    con.close()
if __name__=="__main__": main()
