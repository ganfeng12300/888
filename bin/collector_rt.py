# -*- coding: utf-8 -*-
"""
collector_rt.py  —  Standalone incremental collector for Binance USDT-M (closed candles only)
- One-shot by default (good for Windows Task Scheduler). Use --loop to keep running.
- Creates tables like SYMBOL_tf with schema:
  (timestamp INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL)
"""

import argparse, json, os, sqlite3, sys, threading, queue, time, ssl
from pathlib import Path
from urllib import request, parse, error

TF_SECONDS = {"5m":300,"15m":900,"30m":1800,"1h":3600,"2h":7200,"4h":14400,"1d":86400}
VALID_TF = set(TF_SECONDS.keys())
BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"  # USDT-M futures

# ---------- DB helpers ----------
def open_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=120, check_same_thread=False)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA temp_store=MEMORY")
    try: cur.execute("PRAGMA page_size=32768")
    except sqlite3.Error: pass
    cur.execute("PRAGMA mmap_size=134217728")
    return con

def ensure_table(cur: sqlite3.Cursor, tname: str):
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS '{tname}' ("
        "timestamp INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL)"
    )

def max_ts(con: sqlite3.Connection, tname: str):
    try:
        r = con.execute(f"SELECT MAX(timestamp) FROM '{tname}'").fetchone()
        return int(r[0]) if r and r[0] is not None else None
    except sqlite3.Error:
        return None

# ---------- HTTP ----------
ssl._create_default_https_context = ssl._create_unverified_context

def http_get(url: str, params: dict, timeout=20):
    ua = "SQuantCollector/1.0 (+win64)"
    q = parse.urlencode(params)
    req = request.Request(url + "?" + q, headers={"User-Agent": ua})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

def parse_klines(data_bytes: bytes):
    arr = json.loads(data_bytes.decode("utf-8"))
    out = []
    for k in arr:
        # [openTime, open, high, low, close, volume, closeTime, ...]
        out.append((int(k[0])//1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
    return out

def fetch_klines(symbol: str, tf: str, start_ms: int, end_ms: int, limit=1500):
    return parse_klines(http_get(BINANCE_KLINES, {
        "symbol": symbol,
        "interval": tf,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit
    }))

# ---------- Collector core ----------
def run_pair(symbol: str, tf: str, db_path: str, bootstrap_days: int,
             writer_q: "queue.Queue[tuple]", rate_sem: threading.Semaphore) -> int:
    s = TF_SECONDS[tf]
    now = int(time.time())
    last_closed = (now // s) * s - s  # only CLOSED bar
    con = open_db(db_path)
    tname = f"{symbol}_{tf}"
    cur = con.cursor()
    ensure_table(cur, tname)
    last = max_ts(con, tname)

    if last is None:
        # Empty table: backfill window
        if bootstrap_days > 0:
            start = last_closed - bootstrap_days * 86400
        else:
            # grab the latest 30 closed bars by default
            start = last_closed - (30 - 1) * s
    else:
        # Normal incremental
        start = last + s
    end = last_closed

    if start > end:
        con.close()
        return 0

    step_ms = s * 1000 * 1500
    cur_ms = start * 1000
    end_ms = end * 1000
    total = 0

    while cur_ms <= end_ms:
        seg_end = min(cur_ms + step_ms - 1, end_ms)
        try:
            with rate_sem:
                rows = fetch_klines(symbol, tf, int(cur_ms), int(seg_end))
        except error.HTTPError as e:
            # simple backoff on 429/5xx
            time.sleep(1.0)
            continue
        except Exception:
            time.sleep(0.5)
            continue

        if not rows:
            cur_ms = seg_end + 1
            continue

        for ts, o, h, l, c, v in rows:
            writer_q.put((tname, ts, o, h, l, c, v))
            total += 1

        # next slice: jump to next bar after the latest returned
        cur_ms = (rows[-1][0] + s) * 1000

    con.close()
    return total

def collect_once(db_path: str, symbols: list, intervals: list, max_readers: int,
                 bootstrap_days: int, http_parallel: int):
    print(f">>> collector start | symbols={len(symbols)} | intervals={intervals}")
    print("db:", db_path)

    writer_q: "queue.Queue[tuple]" = queue.Queue(maxsize=100000)
    stop_evt = threading.Event()

    def writer():
        con = open_db(db_path)
        cur = con.cursor()
        pending = 0
        while not (stop_evt.is_set() and writer_q.empty()):
            try:
                tname, ts, o, h, l, c, v = writer_q.get(timeout=0.5)
            except queue.Empty:
                if pending:
                    con.commit(); pending = 0
                continue
            ensure_table(cur, tname)
            cur.execute(f"INSERT OR REPLACE INTO '{tname}' VALUES (?,?,?,?,?,?)", (ts, o, h, l, c, v))
            pending += 1
            if pending >= 1000:
                con.commit(); pending = 0
        if pending: con.commit()
        con.close()
        print("[writer] stopped")

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    rate_sem = threading.BoundedSemaphore(max(1, http_parallel))
    tasks = [(sym, tf) for sym in symbols for tf in intervals]
    total = len(tasks)
    done = 0
    lock = threading.Lock()

    def worker(sym, tf):
        nonlocal done
        rows = run_pair(sym, tf, db_path, bootstrap_days, writer_q, rate_sem)
        with lock:
            done += 1
            pct = done * 100.0 / total if total else 100.0
            print(f"[progress] {done}/{total} {pct:5.1f}%  {sym}_{tf} +{rows}")

    threads: list[threading.Thread] = []
    for (sym, tf) in tasks:
        th = threading.Thread(target=worker, args=(sym, tf), daemon=True)
        th.start(); threads.append(th)
        if len(threads) >= max_readers:
            for t in threads: t.join()
            threads.clear()
    for t in threads: t.join()

    stop_evt.set(); wt.join()
    print("collector done")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Standalone incremental collector for Binance USDT-M")
    p.add_argument("--db", required=True, help="SQLite database path, e.g. D:\\quant_system_v2\\data\\market_data.db")
    p.add_argument("--symbols", required=False, default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,APTUSDT",
                   help="Comma-separated symbols")
    p.add_argument("--intervals", default="5m,15m,30m,1h,2h,4h,1d",
                   help="Comma-separated intervals (excludes 1m by design)")
    p.add_argument("--max-readers", type=int, default=24, help="Max parallel read tasks")
    p.add_argument("--http-parallel", type=int, default=8, help="Max concurrent HTTP requests")
    p.add_argument("--bootstrap-days", type=int, default=0,
                   help="If a table is empty, backfill this many days; 0 means last 30 bars")
    p.add_argument("--loop", type=int, default=0,
                   help="If >0, keep running and sleep N seconds between passes (e.g. 300)")
    return p.parse_args()

def main():
    args = parse_args()
    db_path = args.db
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip() in VALID_TF]

    if not intervals:
        print("No valid intervals. Allowed:", ",".join(sorted(VALID_TF))); sys.exit(2)

    try:
        while True:
            collect_once(db_path, symbols, intervals, args.max_readers, args.bootstrap_days, args.http_parallel)
            if args.loop and args.loop > 0:
                time.sleep(args.loop)
            else:
                break
    except KeyboardInterrupt:
        print("Interrupted.")

if __name__ == "__main__":
    main()
