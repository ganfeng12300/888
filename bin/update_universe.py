# -*- coding: utf-8 -*-
# USDT-M PERPETUAL 鐑棬甯佹睜鍒嗗眰锛圱1/T2/T3锛? Top50锛堝甫榛忔€э級
import os, json, time, ssl, urllib.request
OUTDIR = r'D:\SQuant_Pro\data\universe'; os.makedirs(OUTDIR, exist_ok=True)
EXCHANGE_INFO = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
TICKER_24HR   = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
STICKY_IN  = 45   # 鍚嶆<=45鎵嶇撼鍏?STICKY_OUT = 60   # 鍚嶆>60鎵嶅墧闄?def get(u, timeout=20):
    ctx=ssl.create_default_context()
    with urllib.request.urlopen(u, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode('utf-8'))
def fetch_syms():
    info=get(EXCHANGE_INFO); out=[]
    for s in info.get('symbols',[]):
        if s.get('status')!='TRADING': continue
        if s.get('contractType')!='PERPETUAL': continue
        if s.get('quoteAsset')!='USDT': continue
        out.append(s['symbol'])
    return out
def fetch_24h():
    data=get(TICKER_24HR); m={}
    for x in data:
        s=x.get('symbol'); 
        if not s: continue
        try:qv=float(x.get('quoteVolume','0'))
        except:qv=0.0
        m[s]={'qv':qv,'count':x.get('count')}
    return m
def tiering(syms,m24):
    rows=[(s,m24.get(s,{}).get('qv',0.0)) for s in syms]
    rows.sort(key=lambda z:z[1], reverse=True)
    T1,T2,T3=[],[],[]
    for i,(s,_) in enumerate(rows):
        (T1 if i<30 else T2 if i<100 else T3 if i<300 else T3).append(s)
    top50=[x[0] for x in rows[:50]]
    return T1,T2,T3,top50,rows
def sticky_select(latest50, prev50):
    if not prev50: return latest50
    out=set()
    rank={s:i+1 for i,s in enumerate(latest50)}
    # 淇濈暀鏃op50閲屾帓鍚?=STICKY_OUT鐨勶紱鏂板鍙帴鏀舵帓鍚?=STICKY_IN
    for s in prev50:
        r=rank.get(s,9999)
        if r<=STICKY_OUT: out.add(s)
    for s in latest50:
        if s not in out and rank[s]<=STICKY_IN: out.add(s)
    # 鑻ヤ笉瓒?0锛岀敤鎺掑悕琛ラ綈
    if len(out)<50:
        for s in latest50:
            if s not in out: out.add(s)
            if len(out)>=50: break
    return list(out)[:50]
def main():
    syms=fetch_syms(); m24=fetch_24h()
    T1,T2,T3,top50,rows=tiering(syms,m24)
    latest=os.path.join(OUTDIR,'symbols_latest.json')
    prev=[]
    if os.path.exists(latest):
        try: prev=json.load(open(latest,'r',encoding='utf-8')).get('universe',{}).get('Top50',[])
        except: prev=[]
    top50=sticky_select(top50, prev)
    out={
      'lastUpdated': int(time.time()),
      'universe': {'T1':T1,'T2':T2,'T3':T3,'Top50':top50,'all':T1+T2+T3},
      'meta': {'totalUSDT_PERP':len(syms),'note':'USDT-M PERP only; 24h quoteVolume ranking with sticky 45/60'}
    }
    snap=os.path.join(OUTDIR,f'symbols_{time.strftime("%Y%m%d_%H%M%S")}.json')
    json.dump(out, open(latest,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(out, open(snap,'w',encoding='utf-8'), ensure_ascii=False)
    print('universe ok:', latest, '| T1/T2/T3', len(T1),len(T2),len(T3), '| Top50', len(top50))
if __name__=='__main__': main()