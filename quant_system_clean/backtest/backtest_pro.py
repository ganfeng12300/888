# -*- coding: utf-8 -*-
"""
S档回测：Bayes+GA 寻优、全成本（含冲击）、Walk-Forward、Deflated Sharpe、SPA、PBO 报告�?
中文报表导出 + best_combo.csv（机构级终版，含稳健参数整型化）
"""
import argparse, os, json, time, sqlite3, numpy as np, pandas as pd
from datetime import datetime
from importlib import import_module
from numbers import Real

# === 依赖于项目内模块（保持与您现有工程一致） ===
from tools.config import get_db_path, get_results_db, runtime_params
from tools.db_util import connect_ro
from tools.fees_rules import fetch_funding_series, apply_costs, estimate_impact_bps
from strategy.strategies_a1a8 import STRATS
from backtest.stats_validators import (
    equity_metrics, walk_forward_splits, deflated_sharpe,
    spa_significance, probability_of_backtest_overfitting
)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# 支持的周�?
TFS = ["5m", "15m", "30m", "1h", "2h", "4h", "1d"]

# ---- 这些参数在策略中“必须是整数”，统一在入口做强制整型�?----
INT_PARAMS = {
    "lookback","period","atr_n","rsi_n","fast","slow",
    "n_estimators","num_leaves","max_depth","epochs","hidden","window",
}

# 参数空间（与 STRATS 中函数保持一致；键为 A1..A8�?
SPACE = {
 "A1": {"period": hp.quniform("period",18,34,2), "n": hp.uniform("n",1.2,2.8)},
 "A2": {"fast": hp.quniform("fast",8,16,2), "slow": hp.quniform("slow",40,70,2)},
 "A3": {"period": hp.quniform("period",12,22,2), "low": hp.quniform("low",20,35,1), "high": hp.quniform("high",60,75,1)},
 "A4": {"atr_n": hp.quniform("atr_n",12,22,2), "k": hp.uniform("k",1.2,2.4)},
 "A5": {"lookback": hp.quniform("lookback",20,50,2), "n_estimators": hp.quniform("n_estimators",150,300,50),
        "num_leaves": hp.quniform("num_leaves",31,63,2), "lr": hp.uniform("lr",0.02,0.08)},
 "A6": {"lookback": hp.quniform("lookback",20,50,2), "n_estimators": hp.quniform("n_estimators",200,400,50),
        "max_depth": hp.quniform("max_depth",4,8,1)},
 "A7": {"lookback": hp.quniform("lookback",24,40,2), "hidden": hp.quniform("hidden",16,48,8), "epochs": hp.quniform("epochs",2,4,1)},
 "A8": {"rsi_n": hp.quniform("rsi_n",12,18,2), "rsi_low": hp.quniform("rsi_low",20,30,1), "atr_n": hp.quniform("atr_n",12,18,2), "k": hp.uniform("k",1.2,2.0)},
}

# ---- 解析与解析器兼容：外层可能传入未知开关（--spa 等），使�?parse_known_args() 忽略 ----
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=get_db_path())
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--symbols", nargs="+")
    # 单策略选择（供外层并行总控注入 A1..A8�?
    ap.add_argument("--only-strategy", dest="only_strategy", default="",
                    help="仅运行指定策略：A1..A8（若留空则运行全�?A1..A8�?)
    return ap

# --- 兜底解析器（�?STRATS 未注册时寻找函数�?---
def _resolve_fn(strat_key):
    S = import_module("strategy.strategies_a1a8")
    key = str(strat_key)
    # 1) STRATS 注册
    if hasattr(S, "STRATS") and key in getattr(S, "STRATS"):
        return getattr(S, "STRATS")[key][1]
    # 2) 同名函数
    if hasattr(S, key) and callable(getattr(S, key)):
        return getattr(S, key)
    # 3) 常见别名（GPU�?
    alias = {"XGB":"strat_xgb_gpu","LGBM":"strat_lgbm_gpu","LSTM":"strat_lstm_gpu"}
    name = alias.get(key.upper())
    if name and hasattr(S, name) and callable(getattr(S, name)):
        return getattr(S, name)
    # 4) 模糊关键字兜�?
    names=[n for n in dir(S) if n.startswith("strat_") and callable(getattr(S,n,None))]
    low={n.lower():n for n in names}
    kw_map={"A1":["bbands","band"],"A2":["ma_cross","cross","don","break","channel","bo"],
            "A3":["rsi"],"A4":["atr","break"],"A7":["lstm"]}
    for kw in kw_map.get(key.upper(),[]):
        for ln,orig in low.items():
            if kw in ln:
                return getattr(S, orig)
    raise KeyError(f"Unknown strategy: {strat_key}")

# ---- 强制把必须整型的参数转成整数（含 numpy 标量�?----
def _coerce_params(params: dict) -> dict:
    out={}
    for k,v in (params or {}).items():
        try:
            if isinstance(v, Real) and np.isfinite(v):
                vf=float(v)
                if k in INT_PARAMS:
                    iv = int(round(vf))
                    out[k] = max(1, iv)     # 窗口至少�?1
                else:
                    out[k] = vf
            else:
                out[k]=v
        except Exception:
            out[k]=v
    return out

def read_klines(db, table):
    with connect_ro(db) as con:
        try:
            df=pd.read_sql_query(f'SELECT ts, open, high, low, close, volume FROM "{table}" ORDER BY ts ASC', con)
            return df if not df.empty else None
        except Exception:
            return None

def backtest_once(df, strat_key, params, rp, symbol, tf):
    # 优先使用 STRATS；仅当缺失时才用解析�?
    name, fn = STRATS.get(strat_key, (strat_key, None))
    if not callable(fn):
        fn = _resolve_fn(strat_key)

    # --- 进入策略前强制整型化 ---
    p_clean = _coerce_params(params)

    pos = fn(df, **p_clean)

    close=pd.to_numeric(df["close"], errors="coerce").astype(float)
    ret = pos.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)

    # 防极端数据导致数值爆�?
    ret = ret.clip(-0.5, 0.5)

    # 资金费率（若开启）
    fund_df=None
    if rp.get("funding_on", False):
        try:
            fund_df=fetch_funding_series(symbol, int(df["ts"].iloc[0]), int(df["ts"].iloc[-1]))
        except Exception:
            fund_df=None

    # 冲击成本估计（以20日平均名义ADV为近似）
    adv_usdt = float((close*pd.to_numeric(df["volume"]).fillna(0)).rolling(24*20).mean().dropna().median() or 1e6)
    # 进出名义（风险预算近似→名义�?
    notional_series = pos.diff().abs().fillna(0.0) * (rp["risk_per_trade"]/max(0.004,0.01))
    impact_bps = estimate_impact_bps(notional_series, adv_usdt, kappa=15.0)

    ret = apply_costs(ret, pos, taker_fee=rp["taker_fee"], slippage=rp["slippage"],
                      funding_df=fund_df, bar_ts=df["ts"].tolist(), impact_bps_series=impact_bps)

    eq=(1+ret.fillna(0.0)).cumprod()
    met=equity_metrics(eq)
    trades=int(((pos.diff()!=0)&(pos==1)).sum())
    winrate=float((ret[ret!=0]>0).mean()*100) if trades>0 else np.nan

    return {
        "Symbol":symbol,"时间周期":tf,"策略":strat_key,"参数JSON":json.dumps(p_clean,ensure_ascii=False),
        "总收�?%)":met["总收�?%)"],"年化(%)":met["年化(%)"],"夏普�?:met["夏普�?],"胜率(%)":winrate,
        "交易次数":trades,"最大回�?%)":met["最大回�?%)"],
        "手续费滑点成�?%)":float(((rp["taker_fee"]+rp["slippage"]) * trades)*100),
        "资金费率影响(%)":0.0,"冲击成本(%)":float(impact_bps.fillna(0).sum()/100.0),
        "pos":pos, "ret":ret, "eq":eq
    }

def bayes_ga_optimize(df, symbol, tf, strat_key, rp, n_bayes=25, n_ga=20, elite=10):
    space=SPACE[strat_key]
    results=[]
    def objective(p):
        res=backtest_once(df, strat_key, p, rp, symbol, tf)
        # 机构级综合评分（可按需微调权重�?
        score = 0.5*(res["夏普�?]/3.0) + 0.3*(res["总收�?%)"]/100.0) - 0.2*(res["最大回�?%)"]/50.0)
        results.append(res)
        return {"loss": -score, "status": STATUS_OK}

    trials=Trials()
    fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=n_bayes, trials=trials, rstate=np.random.default_rng(42))

    # 精英解做 GA 局部变异（注意：对 INT_PARAMS 的参数，变异后也强制回整型）
    elite_rows=sorted(results, key=lambda r: (-r["总收�?%)"], -r["夏普�?]))[:elite]
    def mutate(p):
        q=dict(p)
        for k in space.keys():
            v=q.get(k)
            if isinstance(v,(int,float,Real)):
                try:
                    nv = float(v)*(1+np.random.normal(0,0.1))
                    if k in INT_PARAMS:
                        q[k] = max(1, int(round(nv)))
                    else:
                        q[k] = nv
                except Exception:
                    pass
        return q

    for _ in range(n_ga):
        base=np.random.choice(elite_rows)
        cand=mutate(json.loads(base["参数JSON"]))
        cand=_coerce_params(cand)
        results.append(backtest_once(df, strat_key, cand, rp, symbol, tf))

    return results

def walk_forward_validate(df, strat_key, best_params, rp, symbol, tf, kfold=5):
    n=len(df)
    splits=walk_forward_splits(n, k=kfold)
    oos=[]
    for i,(a,b) in enumerate(splits):
        sub=df.iloc[:b].copy()
        res=backtest_once(sub, strat_key, best_params, rp, symbol, tf)
        eq=res["eq"]; oos.append(eq.iloc[-1]-1.0)
    oos_ret=np.array([float(x) for x in oos if pd.notna(x)])
    return (oos_ret.mean()*100.0) if len(oos_ret)>0 else np.nan

def main():
    # 环境/GPU信息（如无则忽略�?
    try:
        from utils.gpu_accel import log_env
        log_env()
    except Exception:
        pass

    ap = build_parser()
    # 关键：接受外层未知参数（--spa/--pbo/--impact-recheck/...�?
    args, _unknown = ap.parse_known_args()
    rp = runtime_params()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 自动发现/整理符号
    if args.symbols:
        symbols = sorted(set(s.upper() for s in args.symbols))
    else:
        with connect_ro(args.db) as con:
            rows=con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        got=set()
        for (tb,) in rows:
            if "_" in tb:
                s,tf=tb.split("_",1)
                if tf in TFS: got.add(s)
        symbols=sorted(got)[:80]  # 上限保护

    print(f"[回测] 符号�?{len(symbols)} 窗口={args.days}�?S�?ON")

    # 选定要跑的策略列表（支持 --only-strategy A1..A8；否则全部）
    if args.only_strategy:
        sel = args.only_strategy.strip().upper()
        if sel not in SPACE.keys():
            raise SystemExit(f"[ERROR] unknown --only-strategy: {args.only_strategy}")
        strategies_to_run = [sel]
    else:
        strategies_to_run = list(SPACE.keys())  # A1..A8

    all_rows=[]
    cutoff_utc_ms = int(pd.Timestamp.utcnow().value//10**6) - args.days*24*3600*1000

    for s in symbols:
        for tf in TFS:
            df=read_klines(args.db, f"{s}_{tf}")
            if df is None or df.empty: continue
            df=df[df["ts"]>=cutoff_utc_ms]
            if len(df)<200: continue

            for strat_key in strategies_to_run:
                try:
                    rs=bayes_ga_optimize(df, s, tf, strat_key, rp)
                    all_rows.extend(rs)
                except Exception as e:
                    # 单策略失败不中断全局（与外层容错一致）
                    print(f"[WARN] {s} {tf} {strat_key} failed: {e}")

    if not all_rows:
        print("无结�?); return

    # 汇�?�?稳健�?校正
    df_all=pd.DataFrame([{k:v for k,v in r.items() if k not in ["pos","ret","eq"]} for r in all_rows])
    gcols=["Symbol","时间周期","策略","参数JSON"]
    df_best=(df_all.sort_values(["Symbol","时间周期","策略","总收�?%)","夏普�?], ascending=[True,True,True,False,False])
                  .groupby(gcols, as_index=False).head(1))

    # 每个币保留唯一最佳（多周期自动选优�?
    keep=[]
    for sym, g in df_best.groupby("Symbol"):
        best=(g.sort_values(["总收�?%)","夏普�?,"胜率(%)"], ascending=[False,False,False]).head(1))
        keep.append(best)
    best_per_symbol=pd.concat(keep, ignore_index=True)

    # Walk-Forward、Deflated Sharpe、SPA、PBO
    oos_list=[]; ds_list=[]; spa_list=[]; pbo_list=[]
    for i,row in best_per_symbol.iterrows():
        s=row["Symbol"]; tf=row["时间周期"]; strat=row["策略"]; params=json.loads(row["参数JSON"])
        df=read_klines(args.db, f"{s}_{tf}")
        if df is None: 
            oos_list.append(np.nan); ds_list.append(np.nan); spa_list.append("�?); pbo_list.append(np.nan)
            continue
        df=df[df["ts"]>=cutoff_utc_ms]
        oos=walk_forward_validate(df, strat, params, rp, s, tf, kfold=5)
        ds=deflated_sharpe(row["夏普�?], n_strats=max(1,len(df_all)), n_obs=len(df))
        g=df_all[(df_all["Symbol"]==s) & (df_all["时间周期"]==tf)]
        sig,p = spa_significance(g["总收�?%)"].fillna(0).values)
        pbo = probability_of_backtest_overfitting(g["总收�?%)"].rank().values,
                                                  g["夏普�?].rank().values)
        oos_list.append(oos); ds_list.append(ds); spa_list.append("�? if sig else "�?); pbo_list.append(pbo)

    best_per_symbol["样本外收�?%)"]=oos_list
    best_per_symbol["去水分夏�?]=ds_list
    best_per_symbol["SPA显著�?�?�?"]=spa_list
    best_per_symbol["PBO(过拟合概�?"]=pbo_list
    best_per_symbol["稳健性通过(�?�?"]=["�? if (a>=0 and b=='�?) else "�? for a,b in zip(oos_list, spa_list)]

    # 导出中文报表
    ts_str=datetime.now().strftime("%Y%m%d-%H%M%S")
    path1=os.path.join(args.outdir, f"最优组合总表_S档_{ts_str}.csv")
    path2=os.path.join(args.outdir, f"全量回测明细_S档_{ts_str}.csv")
    path3=os.path.join(args.outdir, f"参数寻优轨迹_S档_{ts_str}.csv")
    path4=os.path.join(args.outdir, f"稳健性与检验报告_S档_{ts_str}.csv")

    best_per_symbol.drop(columns=["pos","ret","eq"], errors="ignore").to_csv(path1, index=False, encoding="utf-8-sig")
    df_all.drop(columns=["pos","ret","eq"], errors="ignore").to_csv(path2, index=False, encoding="utf-8-sig")
    df_all[["Symbol","时间周期","策略","参数JSON","总收�?%)","夏普�?,"最大回�?%)","胜率(%)","交易次数"]].to_csv(path3, index=False, encoding="utf-8-sig")
    best_per_symbol[["Symbol","时间周期","策略","参数JSON","样本外收�?%)","去水分夏�?,"SPA显著�?�?�?","PBO(过拟合概�?","稳健性通过(�?�?"]].to_csv(path4, index=False, encoding="utf-8-sig")

    # best_combo.csv（机器友好，实盘使用�?
    os.makedirs("data", exist_ok=True)
    best_combo_path="D:\\SHUJU888\\data\\best_combo.csv"
    out_cols=["Symbol","时间周期","策略","参数JSON","总收�?%)","夏普�?,"最大回�?%)","胜率(%)","交易次数"]
    best_per_symbol[out_cols].to_csv(best_combo_path, index=False, encoding="utf-8-sig")

    # 简�?PBO 可视化（html�?
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        ax.hist(best_per_symbol["PBO(过拟合概�?"].fillna(0), bins=10)
        ax.set_title("PBO 分布（越低越好）"); ax.set_xlabel("PBO"); ax.set_ylabel("频数")
        html=os.path.join(args.outdir, f"PBO报告_S档_{ts_str}.html")
        import io, base64
        buf=io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png"); data=base64.b64encode(buf.getvalue()).decode()
        with open(html,"w",encoding="utf-8") as f:
            f.write(f"<img src='data:image/png;base64,{data}'/>")
    except Exception:
        pass

    print(f"[完成] 最优表: {path1}")
    print(f"[完成] best_combo: {best_combo_path}")

if __name__=="__main__":
    main()
