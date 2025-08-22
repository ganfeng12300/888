# -*- coding: utf-8 -*-
# Clean incremental collector (Binance USDT-M, closed candles only)
import sys, os, time, json, sqlite3, argparse, threading, queue, urllib.request, urllib.parse, ssl, configparser
from pathlib import Path

# add project root to sys.path so "modules.*" works even if launched by path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from modules.common.minimal_ansi import enable_virtual_terminal, colorize, CYAN, GREEN, YELLOW, RED
except Exception:
    CYAN=GREEN=YELLOW=RED=""
    def colorize(x,_): return x
    def enable_virtual_terminal(): pass

enable_virtual_terminal()
ssl._create_default_https_context = ssl._create_unverified_context

TF_SECONDS = {"5m":300,"15m":900,"30m":1800,"1h":3600,"2h":7200,"4h":14400,"1d":86400}
VALID_TF = list(TF_SECONDS.keys())

def http_get(url, params):
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(url + "?" + q, timeout=20) as r:
        return r.read()

def parse_klines(data_bytes):
    arr = json.loads(data_bytes.decode("utf-8"))
    out = []
    for k in arr:
        # Binance: k[0]=openTime(ms), k[6]=closeTime(ms). We only write CLOSED bars.
        out.append((int(k[0])//1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
    return out

def fetch_klines(symbol, tf, start_ms, end_ms, limit=1500):
    url = "https://fapi.binance.com/fapi/v1/klines"
    return parse_klines(http_get(url, {
        "symbol": symbol, "interval": tf, "limit": limit,
        "startTime": start_ms, "endTime": end_ms
    }))

def open_rw(db_path):
    con = sqlite3.connect(db_path, timeout=120, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    try: con.execute("PRAGMA page_size=32768")
    except: pass
    con.execute("PRAGMA mmap_size=134217728")
    return con

def ensure_table(cur, tname):
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{tname}' (timestamp INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL)")

def max_ts(con, tname):
    try:
        r = con.execute(f"SELECT MAX(timestamp) FROM '{tname}'").fetchone()
        return int(r[0]) if r and r[0] is not None else None
    except sqlite3.Error:
        return None

def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def main():
    ap = argparse.ArgumentParser(description="incremental collector for Binance USDT-M")
    ap.add_argument("--config", required=True)
    ap.add_argument("--risk", required=True)
    ap.add_argument("--wl", required=True)
    ap.add_argument("--max-readers", type=int, default=24)
    ap.add_argument("--bootstrap-days", type=int, default=0, help="if table empty, backfill this many days")
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding="utf-8")
    db_path   = cfg.get("paths","db")
    intervals = [x.strip() for x in cfg.get("periods","intervals").split(",") if x.strip() in VALID_TF]

    wl = read_list(args.wl)
    print(colorize(f">>> collector start | symbols={len(wl)} | intervals={intervals}", CYAN))
    print("db:", db_path)

    writer_q = queue.Queue(maxsize=50000)
    stop_evt = threading.Event()

    def writer():
        con = open_rw(db_path)
        cur = con.cursor()
        pending = 0
        while not (stop_evt.is_set() and writer_q.empty()):
            try:
                tname, ts, o,h,l,c,v = writer_q.get(timeout=0.5)
            except queue.Empty:
                if pending:
                    con.commit(); pending = 0
                continue
            ensure_table(cur, tname)
            cur.execute(f"INSERT OR REPLACE INTO '{tname}' VALUES (?,?,?,?,?,?)", (ts,o,h,l,c,v))
            pending += 1
            if pending >= 1000:
                con.commit(); pending = 0
        if pending: con.commit()
        con.close()
        print(colorize("[writer] stopped", GREEN))

    wt = threading.Thread(target=writer, daemon=True); wt.start()

    rate_guard = threading.BoundedSemaphore(8)

    def run_pair(symbol, tf):
        con = open_rw(db_path)
        tname = f"{symbol}_{tf}"
        last  = max_ts(con, tname)
        s = TF_SECONDS[tf]
        now_ms = int(time.time()*1000)
        if last is None:
            start_ms = now_ms - max(1, args.bootstrap_days)*86400*1000
        else:
            start_ms = (last + s)*1000
        end_cap = now_ms - s*1000  # only closed candle
        step_ms = s*1000*1500
        cur_ms = start_ms
        total = 0
        while cur_ms <= end_cap:
            end_ms = min(cur_ms + step_ms - 1, end_cap)
            try:
                with rate_guard: ks = fetch_klines(symbol, tf, cur_ms, end_ms)
            except Exception:
                time.sleep(0.8); continue
            if not ks:
                cur_ms = end_ms + 1; continue
            for k in ks:
                writer_q.put((tname, k[0], k[1],k[2],k[3],k[4],k[5]))
                total += 1
            cur_ms = (ks[-1][0]*1000) + 1
        con.close()
        return total

    tasks = [(sym, tf) for sym in wl for tf in intervals]
    done = 0; total = len(tasks)
    lock = threading.Lock()

    def worker(sym, tf):
        nonlocal done
        rows = run_pair(sym, tf)
        with lock:
            done += 1
            pct = done*100.0/total if total else 100.0
            print(f"[progress] {done}/{total} {pct:5.1f}%  {sym}_{tf} +{rows}")

    threads = []
    for (sym, tf) in tasks:
        th = threading.Thread(target=worker, args=(sym,tf), daemon=True)
        th.start(); threads.append(th)
        if len(threads) >= args.max_readers:
            for x in threads: x.join()
            threads.clear()
    for x in threads: x.join()

    stop_evt.set(); wt.join()
    print(colorize("collector done", GREEN))

if __name__ == "__main__":
    main()
