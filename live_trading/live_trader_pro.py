# -*- coding: utf-8 -*-
"""
live_trader_pro.py — 机构级安全纸面/真盘双挡（默认纸面）
- 数据源：SQLite，表名 {SYMBOL}_{TF}，字段 [ts, open, high, low, close, volume]
- 策略：strategy/strategies_a1a8.py 中的 STRATS（A1..A8）
- 风控：固定单笔仓位比例，含手续费；仅做多（演示版，避免复杂度）
- 输出：彩色终端 + 落盘 CSV（trades.csv, summary.json）
"""
import argparse, os, time, sqlite3, json, math, datetime as dt
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def now_ts():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def read_latest_df(db, symbol, tf, limit=600):
    tb = f"{symbol}_{tf}"
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        if not con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tb,)).fetchone():
            return pd.DataFrame()
        df = pd.read_sql_query(f'SELECT ts, open, high, low, close, volume FROM "{tb}" ORDER BY ts DESC LIMIT {int(limit)}', con)
    if df.empty:
        return df
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def load_strategy(key):
    from strategy.strategies_a1a8 import STRATS
    if key not in STRATS:
        raise ValueError(f"未知策略键：{key}，可选：{list(STRATS.keys())}")
    return STRATS[key][1]

def sig_to_pos(sig: pd.Series):
    """信号转仓位(0/1)：简单持有逻辑，sig>0 持有，否则空仓；避免未来数据用 shift(0) 已在策略内处理。"""
    s = sig.fillna(0.0).astype(float)
    return s

class PaperBroker:
    def __init__(self, base_equity=100.0, leverage=5.0, position_pct=0.05, fee_side=0.0005):
        self.base_equity = float(base_equity)
        self.equity = float(base_equity)
        self.leverage = float(leverage)
        self.position_pct = float(position_pct)  # 按总资金比例
        self.fee_side = float(fee_side)          # 单边费率（0.0005 = 0.05%）
        self.trades = []  # {time,symbol,tf,side,price,qty,pnl,fee,reason}

        self.open_pos = {}  # symbol -> dict( entry_price, qty, value )

    def _trade_value(self):
        return self.base_equity * self.position_pct * self.leverage

    def on_open(self, symbol, tf, price, reason="signal"):
        value = self._trade_value()
        if value <= 0:
            return
        qty = value / max(price, 1e-12)
        fee = value * self.fee_side
        self.open_pos[symbol] = dict(entry_price=price, qty=qty, value=value, fee_open=fee)
        self.trades.append(dict(time=now_ts(), symbol=symbol, tf=tf, side="OPEN", price=price, qty=qty, pnl=0.0, fee=fee, reason=reason))

    def on_close(self, symbol, tf, price, reason="exit"):
        if symbol not in self.open_pos:
            return
        pos = self.open_pos.pop(symbol)
        value = pos["value"]
        qty = pos["qty"]
        pnl_gross = (price - pos["entry_price"]) * qty
        fee = (abs(price) * qty) * self.fee_side
        pnl_net = pnl_gross - pos["fee_open"] - fee
        self.equity += pnl_net
        self.trades.append(dict(time=now_ts(), symbol=symbol, tf=tf, side="CLOSE", price=price, qty=qty, pnl=round(pnl_net, 6), fee=fee, reason=reason))

    def metrics(self):
        closes = [t for t in self.trades if t["side"]=="CLOSE"]
        win = sum(1 for t in closes if t["pnl"]>0)
        loss = sum(1 for t in closes if t["pnl"]<=0)
        winrate = (win / max(win+loss,1)) * 100.0
        total_pnl = sum(t["pnl"] for t in closes)
        return dict(win=win, loss=loss, winrate=winrate, total_pnl=total_pnl, equity=self.equity)

def print_header(run_id, args):
    console.rule(f"[bold green]🔴 实盘（{args.mode}） — run_id={run_id}")
    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("参数", style="bold")
    tbl.add_column("值")
    tbl.add_row("数据库", args.db)
    tbl.add_row("周期", args.tf)
    tbl.add_row("策略", args.strategy)
    tbl.add_row("杠杆×/仓位", f"{args.leverage}× / {int(args.position_pct*100)}%/笔")
    tbl.add_row("费率(单边)", f"{args.fee_side*100:.3f}%")
    tbl.add_row("轮询秒", str(args.interval))
    tbl.add_row("模式", args.mode)
    console.print(tbl)

def print_event_open(symbol, price, pnl_summary):
    console.print(f"[yellow]🔔 OPEN[/yellow] {symbol} @ {price:.4f} | [cyan]胜率[/cyan] {pnl_summary['winrate']:.1f}%  [magenta]累计[/magenta] {pnl_summary['total_pnl']:.4f}")

def print_event_close(symbol, price, pnl, pnl_summary):
    color = "green" if pnl >= 0 else "red"
    console.print(f"[{color}]✅ CLOSE[/] {symbol} @ {price:.4f}  单笔PNL {pnl:+.4f} | [cyan]胜率[/cyan] {pnl_summary['winrate']:.1f}%  [magenta]累计[/magenta] {pnl_summary['total_pnl']:.4f}")

def save_results(out_dir, broker):
    ensure_dir(out_dir)
    df = pd.DataFrame(broker.trades)
    df.to_csv(os.path.join(out_dir, "trades.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(broker.metrics(), f, ensure_ascii=False, indent=2)

def run_loop(args):
    if args.mode.lower() == "real":
        console.print("[bold red]当前为安全挡：真实下单未启用。[/bold red] 我会照常跑纸面逻辑并落盘。需要对接交易所请让我再发 execution_engine。")

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("results", "live_trades", run_id)
    print_header(run_id, args)

    strat = load_strategy(args.strategy)
    broker = PaperBroker(base_equity=args.equity, leverage=args.leverage, position_pct=args.position_pct, fee_side=args.fee_side)

    symbols = []
    if args.symbols_file and os.path.exists(args.symbols_file):
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            symbols = [x.strip().upper() for x in f if x.strip()]
    if not symbols:
        symbols = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"]

    console.print(Panel.fit(f"本次交易对：{', '.join(symbols)}", title="Symbols", style="bold cyan"))

    last_seen_ts = {s: 0 for s in symbols}

    try:
        while True:
            any_update = False
            for s in symbols:
                df = read_latest_df(args.db, s, args.tf, limit=max(300, args.lookback+50))
                if df.empty or len(df) < max(args.lookback+10, 60):
                    continue
                latest_ts = int(df["ts"].iloc[-1])
                if latest_ts <= last_seen_ts[s]:
                    continue
                last_seen_ts[s] = latest_ts
                any_update = True

                # 计算策略信号 → 仓位 → 交易事件
                sig = strat(df)
                pos = sig_to_pos(sig)
                # 最近两根：对比仓位变化决定是否开/平
                p_now = float(df["close"].iloc[-1])
                pos_prev = int(round(pos.iloc[-2])) if len(pos) >= 2 else 0
                pos_curr = int(round(pos.iloc[-1]))

                if pos_prev == 0 and pos_curr == 1 and s not in broker.open_pos:
                    broker.on_open(s, args.tf, p_now, reason="signal_on")
                    print_event_open(s, p_now, broker.metrics())

                if pos_prev == 1 and pos_curr == 0 and s in broker.open_pos:
                    broker.on_close(s, args.tf, p_now, reason="signal_off")
                    m = broker.metrics()
                    print_event_close(s, p_now, broker.trades[-1]["pnl"], m)

            if any_update:
                save_results(out_dir, broker)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        console.print("[bold]收到中断，正在落盘…[/bold]")
        save_results(out_dir, broker)
        console.print(f"[green]已保存[/green] {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite 路径，如 D:\\quant_system_v2\\data\\market_data.db")
    ap.add_argument("--tf", default="5m", choices=["5m","15m","30m","1h","2h","4h","1d"])
    ap.add_argument("--symbols-file")
    ap.add_argument("--strategy", default="A2", help="A1..A8，见 strategies_a1a8.py")
    ap.add_argument("--interval", type=int, default=15, help="轮询秒数")
    ap.add_argument("--equity", type=float, default=100.0)
    ap.add_argument("--leverage", type=float, default=5.0)
    ap.add_argument("--position-pct", type=float, default=0.05)
    ap.add_argument("--fee-side", type=float, default=0.0005)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--mode", default="paper", choices=["paper","real"])
    args = ap.parse_args()
    run_loop(args)

if __name__ == "__main__":
    main()
