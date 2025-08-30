# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from scipy.stats import norm
from statsmodels.tsa.stattools import acf

def equity_metrics(eq: pd.Series):
    if len(eq)<2: return {"总收�?%)":np.nan,"年化(%)":np.nan,"夏普�?:np.nan,"最大回�?%)":np.nan}
    ret=eq.pct_change().fillna(0.0)
    sharpe=np.sqrt(365)*ret.mean()/(ret.std()+1e-12)
    peak=eq.cummax(); dd=(eq/peak-1.0).min()
    days=max(1,len(eq)/24.0); cagr=(eq.iloc[-1]/eq.iloc[0])**(365/days)-1.0
    return {"总收�?%)":100*float(eq.iloc[-1]-1.0), "年化(%)":100*float(cagr),
            "夏普�?:float(sharpe), "最大回�?%)":100*float(-dd)}

def walk_forward_splits(n, k=5):
    # 均匀切为 k 折（简单版�?
    b=np.linspace(0,n,k+1, dtype=int)
    return [(b[i], b[i+1]) for i in range(k)]

def deflated_sharpe(sharpe, n_strats, n_obs):
    # Bailey & López de Prado 近似（简化）
    if n_obs<=1: return np.nan
    emax = (1-0.75*np.log(np.log(n_strats))-0.5/np.log(n_strats)) if n_strats>1 else 0
    return float(max(0.0, sharpe - emax*np.sqrt((1-0.01)/max(1,n_obs))))

def spa_significance(scores: np.ndarray, B=500):
    # SPA 简化：置换重采样判断最佳是否显著优于零基准
    best=scores.max()
    cnt=0
    for _ in range(B):
        perm=np.random.permutation(scores)
        if perm.max()>=best: cnt+=1
    p=cnt/max(1,B)
    return p<0.05, p

def probability_of_backtest_overfitting(ranks_in, ranks_out, bins=10):
    # PBO：In-sample 排名�?OOS 排名�?Kendall-like 反序程度
    if len(ranks_in)!=len(ranks_out) or len(ranks_in)==0:
        return np.nan
    x=pd.Series(ranks_in).rank(pct=True).values
    y=pd.Series(ranks_out).rank(pct=True).values
    # 计算反序概率（简化）
    inv=np.mean((x<0.5)&(y>0.5))
    return float(inv)
