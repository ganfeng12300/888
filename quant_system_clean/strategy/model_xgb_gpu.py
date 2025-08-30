# -*- coding: utf-8 -*-
"""XGBoost GPU 策略（分类）——自动用 GPU，无 GPU 时回退 CPU�?""
import numpy as np, pandas as pd
from utils.gpu_accel import xgb_params
try:
    import xgboost as xgb
except Exception:
    xgb=None

def _rsi(s, period=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up/(dn+1e-12)
    return 100 - 100/(1+rs)

def _features(df: pd.DataFrame):
    c = pd.to_numeric(df.get('close'), errors='coerce').astype(float)
    v = pd.to_numeric(df.get('volume', 0), errors='coerce').astype(float)
    ret1  = c.pct_change()
    ret5  = c.pct_change(5)
    ema12 = c.ewm(span=12, min_periods=12).mean()
    ema26 = c.ewm(span=26, min_periods=26).mean()
    macd  = ema12 - ema26
    hist  = macd - macd.ewm(span=9, min_periods=9).mean()
    rsi14 = _rsi(c,14)
    volz  = (v.rolling(20,min_periods=20).mean()-v.rolling(60,min_periods=60).mean())/(v.rolling(60,min_periods=60).std()+1e-9)
    X = pd.DataFrame({
        'ret1':ret1,'ret5':ret5,'ema12':ema12,'ema26':ema26,
        'macd':macd,'hist':hist,'rsi14':rsi14,'volz':volz
    }, index=df.index).replace([np.inf,-np.inf], np.nan).dropna()
    # 目标：下一根收益是否为正（0/1�?
    y = (c.pct_change().shift(-1).reindex(X.index)>0).astype(int)
    return X, y

def strat_xgb(df: 'pd.DataFrame', lookback:int=3000, train_ratio:float=0.7,
              n_estimators:int=400, max_depth:int=6, learning_rate:float=0.05,
              subsample:float=0.8, colsample_bytree:float=0.8, threshold:float=0.5):
    if xgb is None:
        # 没安�?xgboost �?返回�?0（空仓）
        return pd.Series(0, index=df.index, dtype=int)
    X, y = _features(df)
    if len(X) < max(300, int(lookback*0.6)):
        return pd.Series(0, index=df.index, dtype=int)
    X = X.iloc[-lookback:]; y = y.loc[X.index]
    ntr = max(50, int(len(X)*train_ratio))
    Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]
    ytr, yte = y.iloc[:ntr], y.iloc[ntr:]
    if len(Xte)==0:
        return pd.Series(0, index=df.index, dtype=int)
    params = xgb_params({
        'n_estimators': int(n_estimators),
        'max_depth'   : int(max_depth),
        'learning_rate': float(learning_rate),
        'subsample'   : float(subsample),
        'colsample_bytree': float(colsample_bytree),
        'random_state': 42,
        'n_jobs'      : 0,
        'verbosity'   : 0,
        'objective'   : 'binary:logistic'
    })
    model = xgb.XGBClassifier(**params)
    model.fit(Xtr, ytr)
    proba = pd.Series(model.predict_proba(Xte)[:,1], index=Xte.index)
    sig   = (proba > float(threshold)).astype(int)*2-1  # 1/-1
    pos   = pd.Series(0, index=df.index, dtype=int)
    pos.loc[sig.index] = sig.values
    return pos.shift(1).fillna(0).astype(int)
