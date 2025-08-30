# -*- coding: utf-8 -*-
"""
统一实盘路由（Binance/OKX/Bitget�?
- 读取 D:\\SHUJU888\\data\\best_combo.csv（回测最优组合）
- 从同一数据库读取最新K线计算信号（口径一致）
- 名义按“每笔风�?%”预算；**冰山分单**执行；Prometheus 指标
"""
import os, time, json, pandas as pd, numpy as np
from tools.config import get_db_path, runtime_params, load_keys
from tools.db_util import connect_ro, console
from strategy.strategies_a1a8 import STRATS
from live.executors import BinanceExec, OkxExec, BitgetExec
from prometheus_client import Counter, Gauge, start_http_server

DB=get_db_path()
BEST="D:\\SHUJU888\\data\\best_combo.csv"
ROUTE="D:\\SHUJU888\\data\\route_map.csv"
TFS=["5m","15m","30m","1h","2h","4h","1d"]

# 指标
ORDERS=Counter("qs_orders_total","Total orders",["symbol","side","mode","exchange"])
PNL=Gauge("qs_daily_pnl","Daily PnL (fraction)")
HEART=Counter("qs_heartbeat_total","Heartbeats")
IMPACT=Gauge("qs_cost_impact_bps","Impact cost bps (last)")
FEE=Gauge("qs_cost_fee_bps","Fee+slippage bps (last)")
EXPO=Gauge("qs_pos_exposure","Position notional",["symbol","exchange"])

def start_metrics_only():
    rp=runtime_params()
    start_http_server(rp["metrics_port"])
    console.print(f"[blue]Metrics on :{rp['metrics_port']}[/blue]")
    while True:
        HEART.inc(); time.sleep(5)

def _read_table(db, table, n=600):
    with connect_ro(db) as con:
        try:
            df=pd.read_sql_query(f'SELECT ts, open, high, low, close, volume FROM "{table}" ORDER BY ts DESC LIMIT {n}', con)
        except Exception:
            return None
    return df.iloc[::-1].reset_index(drop=True) if df is not None and not df.empty else None

def _signal(df, strat_key, params):
    name, fn = STRATS[strat_key]
    pos=fn(df, **json.loads(params))
    if len(pos)<2: return 0
    prev,cur=pos.iloc[-2], pos.iloc[-1]
    if prev<0.5 and cur>=0.5: return +1
    if prev>=0.5 and cur<0.5: return -1
    return 0

def _load_route_map(symbols):
    if os.path.exists(ROUTE):
        d=pd.read_csv(ROUTE)
        m={str(r["Symbol"]).upper():str(r["Exchange"]).upper() for _,r in d.iterrows()}
        return {s:m.get(s,"BINANCE") for s in symbols}
    exs=["BINANCE","OKX","BITGET"]
    out={}
    for i,s in enumerate(sorted(set(symbols))):
        out[s]=exs[i%len(exs)]
    os.makedirs("data", exist_ok=True)
    pd.DataFrame([{"Symbol":k,"Exchange":v} for k,v in out.items()]).to_csv(ROUTE,index=False,encoding="utf-8-sig")
    return out

def _risk_notional(equity, rp):
    risk=rp["risk_per_trade"]*equity; stop=max(0.01,0.004)
    return max(10.0, risk/stop)

def iceberg_plan(total_notional, last_px, max_child=500.0, parts=5):
    # 冰山分单：按子单名义上限切片 + 递增/递减权重
    n=max(1,int(np.ceil(total_notional/max_child)))
    n=max(n, parts)
    weights=np.linspace(1.0,0.6,n)  # 前重后轻
    weights=weights/weights.sum()
    slots=(total_notional*weights).tolist()
    return [max(5.0, s) for s in slots]  # 每单最�?5 USDT

def main():
    rp=runtime_params()
    start_http_server(rp["metrics_port"])
    console.print(f"[blue]Metrics on :{rp['metrics_port']}[/blue]")

    if not os.path.exists(BEST):
        console.print("[red]缺少 D:\\SHUJU888\\data\\best_combo.csv，请先运行回测[/red]"); return
    best=pd.read_csv(BEST)
    symbols=sorted(set(best["Symbol"].astype(str)))
    route=_load_route_map(symbols)

    keys=load_keys()
    bn=BinanceExec(os.environ.get("BINANCE_API_KEY", keys.get("api_key","")), os.environ.get("BINANCE_API_SECRET", keys.get("api_secret","")))
    okx=OkxExec(os.environ.get("OKX_API_KEY",""), os.environ.get("OKX_API_SECRET",""), os.environ.get("OKX_PASSPHRASE",""))
    bgt=BitgetExec(os.environ.get("BITGET_API_KEY",""), os.environ.get("BITGET_API_SECRET",""), os.environ.get("BITGET_PASSPHRASE",""), os.environ.get("BITGET_FLAG","0"))

    equity=1.0; daily_pnl=0.0
    positions={}  # sym -> {"pos":0/1, "entry":px, "qty":qty, "ex":EX}
    console.rule(f"[bold cyan]实盘路由启动 {'PAPER' if rp['paper'] else 'LIVE'}[/bold cyan]")

    while True:
        t0=time.time()
        for _,r in best.iterrows():
            sym=str(r["Symbol"]).upper(); tf=str(r["时间周期"]); strat=str(r["策略"]); params=str(r["参数JSON"])
            df=_read_table(DB, f"{sym}_{tf}", n=600)
            if df is None or len(df)<60: continue
            last=float(df["close"].iloc[-1])
            sig=_signal(df, strat, params)
            st=positions.get(sym, {"pos":0,"entry":0.0,"qty":0.0,"ex":route.get(sym,"BINANCE")})
            ex=st["ex"]
            notional=_risk_notional(equity, rp)
            fee_bps=(rp["taker_fee"]+rp["slippage"])*10000.0
            FEE.set(fee_bps)

            if sig==+1 and st["pos"]==0:
                # 冰山执行
                plan=iceberg_plan(notional, last_px=last, max_child=500.0, parts=5)
                for child in plan:
                    if ex=="BINANCE":
                        qty=child/last; res=bn.market(sym,"BUY",qty,paper=rp["paper"])
                    elif ex=="OKX":
                        res=okx.market(sym,"BUY",child,last,paper=rp["paper"])
                    else:
                        res=bgt.market(sym,"BUY",child,last,paper=rp["paper"])
                    ORDERS.labels(symbol=sym, side="BUY", mode=("PAPER" if rp["paper"] else "LIVE"), exchange=ex).inc()
                    EXPO.labels(symbol=sym, exchange=ex).set(child)
                    time.sleep(0.2)
                positions[sym]={"pos":1,"entry":last,"qty":notional/last,"ex":ex}
                console.print(f"[green]开多[/] {sym} @{last:.4f} notional≈{notional:.2f} �?{ex}")

            elif sig==-1 and st["pos"]==1:
                # 平仓（同样可冰山�?
                child_notional=st["qty"]*last/3.0
                for _ in range(3):
                    if ex=="BINANCE":
                        res=bn.market(sym,"SELL",child_notional/last,paper=rp["paper"])
                    elif ex=="OKX":
                        res=okx.market(sym,"SELL",child_notional,last,paper=rp["paper"])
                    else:
                        res=bgt.market(sym,"SELL",child_notional,last,paper=rp["paper"])
                    ORDERS.labels(symbol=sym, side="SELL", mode=("PAPER" if rp["paper"] else "LIVE"), exchange=ex).inc()
                    time.sleep(0.2)
                gross=(last-st["entry"])/max(1e-8,st["entry"])
                impact_bps=15.0*np.sqrt((st["qty"]*last)/max(1e6, 1e6))
                IMPACT.set(impact_bps)
                cost = (fee_bps/10000.0)*2 + (impact_bps/10000.0)
                pnl=gross - cost
                daily_pnl += pnl; PNL.set(daily_pnl)
                positions[sym]={"pos":0,"entry":0.0,"qty":0.0,"ex":ex}
                console.print(f"[red]平多[/] {sym} @{last:.4f} pnl={pnl:.4f} 累计={daily_pnl:.4f} �?{ex}")

            if daily_pnl <= -abs(rp["max_daily_loss"]):
                console.print("[red]当日回撤触发停机[/red]"); return

        HEART.inc()
        # 节奏
        time.sleep(max(1.0, 10 - (time.time()-t0)))

if __name__=="__main__":
    main()
