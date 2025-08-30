# -*- coding: utf-8 -*-
"""
execution_engine_binance_ws.py �?机构�?WebSocket 实时执行引擎（彩色看板）
- 仅使�?DB 内已有合约币：扫�?*_5m 表或 --symbols-file
- 分片订阅 bookTicker（每�?=50）→ 毫秒级最新价�?s无心跳回退REST
- 真盘 reduceOnly 市价平仓（自动识�?单向/对冲；数量按 stepSize/minQty/precision 对齐�?
- Kill-Switch：close SYMBOL / closeall / panic（并发）/ q 退�?
- 彩色实时看板（Rich Live）：全局状态、心跳延迟、仓�?PnL热力色、指令提�?

依赖：pip install websocket-client rich requests
"""

import os, json, hmac, time, sqlite3, argparse, threading, queue, hashlib, requests, random, string
from urllib.parse import urlencode
from datetime import datetime
from typing import List, Dict

# ---------- 终端 UI ----------
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box

console = Console()

try:
    import websocket  # pip install websocket-client
except Exception:
    console.print("[red]缺少 websocket-client，请执行�?pip install websocket-client[/red]")
    raise

BINANCE_FAPI = "https://fapi.binance.com"
WS_BASE      = "wss://fstream.binance.com/stream?streams="
STREAM_FMT   = "{sym}@bookTicker"
SHARD_SIZE   = 50
HEARTBEAT_MS = 5_000

# ------------------ 小工�?------------------
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ts_ms(): return int(time.time() * 1000)
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def rand_id(n=16): return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def load_keys():
    ak = os.getenv("BINANCE_API_KEY")
    sk = os.getenv("BINANCE_API_SECRET")
    if ak and sk: return ak.strip(), sk.strip()
    p = os.path.join("configs","keys.yaml")
    if os.path.exists(p):
        try:
            import yaml
            d = yaml.safe_load(open(p,"r",encoding="utf-8"))
            b = d.get("binance") or {}
            ak, sk = b.get("api_key"), b.get("api_secret")
            if ak and sk: return ak.strip(), sk.strip()
        except Exception as e:
            console.print(f"[yellow]读取 configs/keys.yaml 失败：{e}[/yellow]")
    return None, None

def list_symbols_from_db(db):
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_5m'").fetchall()
    return sorted(set(r["name"][:-3] for r in rows))

# ------------------ REST 客户�?------------------
class FuturesREST:
    def __init__(self, api_key=None, api_secret=None, timeout=10, recv_window=5000):
        self.ak = (api_key or "").strip()
        self.sk = (api_secret or "").strip()
        self.timeout = timeout
        self.recv_window = int(recv_window)

    def _sign(self, d: dict):
        q = urlencode(d, doseq=True)
        sig = hmac.new(self.sk.encode(), q.encode(), hashlib.sha256).hexdigest()
        return q + "&signature=" + sig

    def _hdr(self): return {"X-MBX-APIKEY": self.ak}

    def _get(self, path, params=None):
        r = requests.get(BINANCE_FAPI + path, params=params or {}, headers=self._hdr(), timeout=self.timeout)
        if r.status_code >= 400: raise RuntimeError(f"GET {path} {r.status_code}: {r.text}")
        return r.json()

    def _signed(self, method, path, params: dict):
        if not self.ak or not self.sk:
            raise RuntimeError("缺少 API Key/Secret（BINANCE_API_KEY / BINANCE_API_SECRET �?configs/keys.yaml�?)
        d = dict(params)
        d["timestamp"] = ts_ms()
        d["recvWindow"] = self.recv_window
        url = BINANCE_FAPI + path + "?" + self._sign(d)
        r = requests.request(method, url, headers=self._hdr(), timeout=self.timeout)
        if r.status_code >= 400: raise RuntimeError(f"{method} {path} {r.status_code}: {r.text}")
        return r.json()

    def position_mode_dual(self):
        d = self._signed("GET", "/fapi/v1/positionSide/dual", {})
        v = d.get("dualSidePosition")
        return bool(v) if isinstance(v, bool) else str(v).lower() == "true"

    def positions(self, symbols=None):
        data = self._signed("GET", "/fapi/v2/positionRisk", {})
        out = {}
        for p in data:
            sym = p["symbol"]
            if symbols and sym not in symbols: continue
            amt = float(p["positionAmt"])
            entry = float(p["entryPrice"])
            side = "LONG" if amt > 0 else ("SHORT" if amt < 0 else "NONE")
            out[sym] = dict(amt=amt, entry=entry, side=side)
        return out

    def symbol_filters(self, symbol):
        info = self._get("/fapi/v1/exchangeInfo", {"symbol": symbol})
        s = info["symbols"][0]
        step = 0.0; min_qty = 0.0
        for f in s["filters"]:
            if f["filterType"] in ("MARKET_LOT_SIZE", "LOT_SIZE"):
                step = float(f["stepSize"]); min_qty = float(f["minQty"])
        return step, min_qty, int(s.get("quantityPrecision", 8))

    def market_reduce_only(self, symbol, qty, side, dual=False, max_retry=3):
        q = abs(float(qty))
        if q <= 0: return {"status":"SKIP_QTY0"}
        body = {
            "symbol": symbol,
            "side": side,             # SELL=平多  BUY=平空
            "type": "MARKET",
            "reduceOnly": "true",
            "newClientOrderId": f"CLOSE-{rand_id()}",
            "newOrderRespType": "RESULT",
            "quantity": None,
        }
        if dual:
            body["positionSide"] = "LONG" if side=="SELL" else "SHORT"
        step, min_qty, prec = self.symbol_filters(symbol)
        def round_step(x, step):
            if step <= 0: return x
            return (int(x / step)) * step
        qty_adj = max(round_step(q, step), min_qty)
        body["quantity"] = f"{qty_adj:.{max(0,prec)}f}".rstrip("0").rstrip(".")
        for i in range(max_retry):
            try:
                r = self._signed("POST", "/fapi/v1/order", body)
                return r
            except Exception:
                if i == max_retry-1: raise
                time.sleep(0.2*(i+1))

    def rest_book_ticker(self, symbol):
        r = requests.get(BINANCE_FAPI+"/fapi/v1/ticker/bookTicker", params={"symbol":symbol}, timeout=3)
        if r.status_code==200:
            d = r.json(); bid = float(d["bidPrice"]); ask = float(d["askPrice"])
            return (bid+ask)/2.0
        return 0.0

# ------------------ WS 分片 ------------------
class WSShard(threading.Thread):
    def __init__(self, symbols, price_dict, stamp_dict, err_q: queue.Queue):
        super().__init__(daemon=True)
        self.symbols = symbols
        self.price = price_dict
        self.stamp = stamp_dict
        self.err_q = err_q
        self._stop = threading.Event()
        streams = "/".join(STREAM_FMT.format(sym=s.lower()) for s in symbols)
        self.url = WS_BASE + streams

    def run(self):
        while not self._stop.is_set():
            ws = websocket.WebSocketApp(
                self.url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open,
            )
            t = threading.Thread(target=ws.run_forever, kwargs={"ping_interval":20,"ping_timeout":10}, daemon=True)
            t.start()
            while t.is_alive() and not self._stop.is_set():
                time.sleep(0.2)
            time.sleep(1.0)  # 退避重�?

    def stop(self): self._stop.set()
    def on_open(self, ws): self.err_q.put(("INFO", f"WS shard up {len(self.symbols)} symbols"))
    def on_close(self, ws, code, msg): self.err_q.put(("WARN", f"WS shard closed {code} {msg}"))
    def on_error(self, ws, err): self.err_q.put(("ERROR", f"WS shard error {err}"))

    def on_message(self, ws, message):
        try:
            d = json.loads(message)
            x = d.get("data") or {}
            sym = x.get("s")
            if not sym: return
            bid = float(x.get("b",0)); ask = float(x.get("a",0))
            if bid>0 and ask>0:
                mid = (bid+ask)/2.0
                self.price[sym] = mid
                self.stamp[sym] = ts_ms()
        except Exception as e:
            self.err_q.put(("ERROR", f"parse message fail: {e}"))

# ------------------ 执行引擎（带彩色看板�?------------------
class ExecEngine:
    def __init__(self, db, symbols, mode="paper", sl_pct=0.0, tp_pct=0.0, trailing_pct=0.0,
                 ui_rows=25):
        self.db = db
        self.symbols: List[str] = symbols
        self.mode = mode.lower()
        self.ui_rows = max(10, int(ui_rows))
        ak, sk = load_keys()
        self.rest = FuturesREST(ak, sk)
        if self.mode=="real" and (not ak or not sk):
            console.print("[yellow]未检测到 API Key，自动降级为 paper[/yellow]")
            self.mode = "paper"
        try:
            self.dual = self.rest.position_mode_dual()
        except Exception:
            self.dual = False

        # 行情与状�?
        self.price: Dict[str, float] = {s:0.0 for s in symbols}
        self.stamp: Dict[str, int]   = {s:0 for s in symbols}
        self.positions: Dict[str, dict] = {}
        self.trailing_ref = {}
        self.cooldown = {}
        self.cmd_q: queue.Queue = queue.Queue()
        self.log_q: queue.Queue = queue.Queue()  # 记录WS/系统信息
        self.shards: List[WSShard] = []

        ensure_dir(os.path.join("results","live_exec"))

    # ====== 交易逻辑 ======
    def refresh_positions(self):
        try:
            self.positions = self.rest.positions(set(self.symbols))
        except Exception as e:
            self.log_q.put(("ERROR", f"读取持仓失败：{e}"))

    def pnl_ratio(self, sym, px):
        p = self.positions.get(sym)
        if not p or p["side"]=="NONE" or p["entry"]<=0: return 0.0
        e = p["entry"]; amt = p["amt"]
        if amt>0: return (px-e)/e
        if amt<0: return (e-px)/e
        return 0.0

    def need_close(self, sym, px):
        p = self.positions.get(sym)
        if not p or p["side"]=="NONE": return False, "NO_POS"
        r = self.pnl_ratio(sym, px)
        if self.sl>0 and r <= -abs(self.sl): return True, f"SL {r:.4f}"
        if self.tp>0 and r >= abs(self.tp):
            if self.tr>0:
                base = self.trailing_ref.get(sym, {"peak": px})
                if p["amt"]>0: base["peak"] = max(base["peak"], px)
                else:          base["peak"] = min(base["peak"], px)
                self.trailing_ref[sym] = base
                return False, "TR_ARMED"
            return True, f"TP {r:.4f}"
        t = self.trailing_ref.get(sym)
        if t and self.tr>0:
            peak = t["peak"]
            if p["amt"]>0 and (peak-px)/peak >= self.tr:  return True, f"TR_STOP {r:.4f}"
            if p["amt"]<0 and (px-peak)/peak >= self.tr:  return True, f"TR_STOP {r:.4f}"
        return False, "HOLD"

    def _cool_ok(self, sym, ms=1200):
        now = ts_ms()
        if now < self.cooldown.get(sym, 0): return False
        self.cooldown[sym] = now + ms
        return True

    def _side_for_close(self, amt): return "SELL" if amt>0 else "BUY"

    def close_symbol(self, sym, reason="manual"):
        try:
            self.refresh_positions()
            p = self.positions.get(sym)
            if not p or p["side"]=="NONE" or abs(p["amt"])<=0:
                self.log_q.put(("INFO", f"{sym} 无持�?))
                return True
            if not self._cool_ok(sym): 
                return False

            qty = abs(p["amt"])
            side = self._side_for_close(p["amt"])
            if self.mode=="paper":
                self.positions[sym]["amt"] = 0.0
                self.log_q.put(("CLOSE", f"[PAPER] {sym} qty={qty} reason={reason}"))
                return True

            # 真盘：重试直到仓位为 0 或超�?
            for attempt in range(4):
                resp = self.rest.market_reduce_only(sym, qty, side, dual=self.dual)
                oid = resp.get("orderId", "?")
                time.sleep(0.3)
                self.refresh_positions()
                p = self.positions.get(sym)
                if not p or abs(p["amt"])<=1e-12:
                    self.log_q.put(("CLOSE", f"[REAL] {sym} oid={oid} reason={reason} �?))
                    return True
                qty = abs(p["amt"])
            self.log_q.put(("WARN", f"仍有持仓未平 {sym} amt={p['amt']}"))
            return False
        except Exception as e:
            self.log_q.put(("ERROR", f"平仓失败 {sym}: {e}"))
            return False

    def close_all(self, reason="close_all"):
        ok=True
        for s in list(self.symbols):
            ok = self.close_symbol(s, reason=reason) and ok
        return ok

    def panic_kill(self):
        self.log_q.put(("WARN","PANIC KILL：并发平所有仓�?))
        ths=[]
        for s in list(self.symbols):
            t = threading.Thread(target=self.close_symbol, args=(s,"panic"), daemon=True)
            t.start(); ths.append(t)
        for t in ths: t.join()

    # ====== 命令�?WS ======
    def cmd_loop(self):
        while True:
            try: line = input().strip()
            except EOFError: return
            if not line: continue
            self.cmd_q.put(line)

    def handle_cmd(self, line):
        p = line.strip().split()
        if not p: return
        op = p[0].lower()
        if op=="close" and len(p)>=2:
            self.close_symbol(p[1].upper(), reason="manual")
        elif op=="closeall":
            self.close_all("manual_all")
        elif op=="panic":
            self.panic_kill()
        elif op in ("q","quit","exit"):
            raise SystemExit
        else:
            self.log_q.put(("INFO", f"未知命令：{line}"))

    def start_ws(self):
        err_q = self.log_q
        chunks = [self.symbols[i:i+SHARD_SIZE] for i in range(0, len(self.symbols), SHARD_SIZE)]
        for c in chunks:
            shard = WSShard(c, self.price, self.stamp, err_q)
            shard.start()
            self.shards.append(shard)

    # ====== UI 组件 ======
    def _header_panel(self):
        mode_str = f"{self.mode.upper()}" + (" / DUAL" if self.dual else " / ONE-WAY")
        title = Text("WS 执行引擎（机构级彩色看板�?, style="bold")
        info = Table.grid(expand=True)
        info.add_column(ratio=1); info.add_column(ratio=1); info.add_column(ratio=1)
        info.add_row(
            Text(f"模式: {mode_str}", style="yellow"),
            Text(f"订阅分片: {len(self.shards)}", style="cyan"),
            Text(f"符号�? {len(self.symbols)}", style="magenta"),
        )
        return Panel(Group(title, info), border_style="bright_blue")

    def _status_panel(self):
        # 消息日志（最�?6 条）
        logs=[]
        try:
            for _ in range(6):
                typ, msg = self.log_q.get_nowait()
                color = dict(INFO="cyan", WARN="yellow", ERROR="red", CLOSE="green").get(typ, "white")
                logs.append(Text(f"[{typ}] {msg}", style=color))
        except queue.Empty:
            pass
        if not logs:
            logs = [Text("就绪�?输入命令：close BTCUSDT | closeall | panic | q", style="dim")]
        return Panel(Group(*logs), title="事件", border_style="white", padding=(0,1))

    def _age_color(self, ms):
        if ms < 1200: return "green"
        if ms < 3000: return "yellow"
        return "red"

    def _positions_table(self):
        # 优先显示有仓位的，其次显示价格最新更新的
        nowms = ts_ms()
        order = sorted(self.symbols, key=lambda s: (0 if (self.positions.get(s,{}).get("side","NONE")!="NONE" and abs(self.positions.get(s,{}).get("amt",0))>0) else 1,
                                                    -(self.stamp.get(s,0))))
        rows = order[:self.ui_rows]

        tbl = Table(box=box.MINIMAL_DOUBLE_HEAD, expand=True)
        tbl.add_column("Symbol", style="bold", no_wrap=True)
        tbl.add_column("Side")
        tbl.add_column("Amt", justify="right")
        tbl.add_column("Entry", justify="right")
        tbl.add_column("Last", justify="right")
        tbl.add_column("PnL%", justify="right")
        tbl.add_column("Age", justify="right")

        for s in rows:
            p = self.positions.get(s, {"side":"NONE","amt":0.0,"entry":0.0})
            last = self.price.get(s,0.0)
            age  = nowms - self.stamp.get(s,0)
            r=0.0
            if p["side"]!="NONE" and p["entry"]>0 and last>0:
                r = (last-p["entry"])/p["entry"] if p["amt"]>0 else (p["entry"]-last)/p["entry"]
            pnl_color = "green" if r>=0 else "red"
            age_color = self._age_color(age)
            tbl.add_row(
                s,
                p["side"],
                f"{p['amt']:.6f}",
                f"{p['entry']:.4f}",
                f"{last:.4f}",
                Text(f"{r*100:.2f}%", style=pnl_color),
                Text(f"{age}ms", style=age_color),
            )
        return Panel(tbl, title="仓位 & 实时价格（优先展示有仓位�?, border_style="bright_magenta")

    def _hint_panel(self):
        lines = [
            Text("指令�?, style="bold"),
            Text("close BTCUSDT", style="yellow"),
            Text(" | "),
            Text("closeall", style="yellow"),
            Text(" | "),
            Text("panic", style="yellow"),
            Text(" | "),
            Text("q", style="yellow"),
            Text("    （reduceOnly 市价平仓�?s 无心跳自动回退 REST�?, style="dim")
        ]
        return Panel(Text.assemble(*lines), border_style="green", title="操作提示", padding=(0,1))

    def _build_layout(self):
        layout = Layout(name="root")
        layout.split(
            Layout(name="top", size=5),
            Layout(name="body", ratio=1),
            Layout(name="bottom", size=3),
        )
        layout["top"].update(self._header_panel())
        # body 再分左右
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        layout["left"].update(self._positions_table())
        right_group = Group(self._status_panel(), self._hint_panel())
        layout["right"].update(right_group)
        layout["bottom"].update(Panel(Text(f"时间 {now()}  |  DB: {self.db}", style="dim"), border_style="blue"))
        return layout

    # ====== 主循�?======
    def run(self):
        if not self.symbols:
            console.print("[red]未发�?*_5m 表对应的币种，请检�?DB 或使�?--symbols-file[/red]"); return

        # 初始
        self.refresh_positions()
        self.start_ws()
        threading.Thread(target=self.cmd_loop, daemon=True).start()

        # UI 实时刷新
        with Live(self._build_layout(), refresh_per_second=10, console=console, screen=True) as live:
            last_pos_refresh = 0
            while True:
                try:
                    # 处理命令（非阻塞�?
                    try: self.handle_cmd(self.cmd_q.get_nowait())
                    except queue.Empty: pass

                    # 心跳回退：无心跳>5s则REST取价
                    now_ms = ts_ms()
                    for s in self.symbols:
                        if now_ms - self.stamp.get(s, 0) > HEARTBEAT_MS:
                            try:
                                px = self.rest.rest_book_ticker(s)
                                if px>0:
                                    self.price[s] = px
                                    self.stamp[s] = now_ms
                            except Exception:
                                pass

                    # 触发止盈止损/追踪（如果设定）
                    for s in self.symbols:
                        px = self.price.get(s,0.0)
                        if px<=0: continue
                        need, why = self.need_close(s, px)
                        if need: self.close_symbol(s, reason=why)

                    # 周期刷新持仓 & UI
                    if now_ms - last_pos_refresh > 10_000:
                        self.refresh_positions()
                        last_pos_refresh = now_ms

                    live.update(self._build_layout())
                    time.sleep(0.05)

                except SystemExit:
                    break
                except Exception as e:
                    self.log_q.put(("ERROR", f"主循环异常：{e}"))
                    time.sleep(1.0)

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB，如 D:\\quant_system_v2\\data\\market_data.db")
    ap.add_argument("--symbols-file")
    ap.add_argument("--mode", default="paper", choices=["paper","real"])
    ap.add_argument("--sl-pct", type=float, default=0.0)
    ap.add_argument("--tp-pct", type=float, default=0.0)
    ap.add_argument("--trailing-pct", type=float, default=0.0)
    ap.add_argument("--ui-rows", type=int, default=25, help="看板表格显示的最大行�?)
    args = ap.parse_args()

    if args.symbols_file and os.path.exists(args.symbols_file):
        symbols = [x.strip().upper() for x in open(args.symbols_file,"r",encoding="utf-8") if x.strip()]
    else:
        symbols = list_symbols_from_db(args.db)

    eng = ExecEngine(args.db, symbols, mode=args.mode,
                     sl_pct=args.sl_pct, tp_pct=args.tp_pct, trailing_pct=args.trailing_pct,
                     ui_rows=args.ui_rows)
    eng.run()

if __name__ == "__main__":
    main()
