import sys, os, sqlite3, time, math, statistics as st
from datetime import datetime
DB  = sys.argv[1] if len(sys.argv)>1 else r"D:\quant_system_v2\data\market_data.db"
OUT = sys.argv[2] if len(sys.argv)>2 else r"D:\SQuant_Pro\reports\backtest_quick.html"
SYMS = (sys.argv[3] if len(sys.argv)>3 else "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
TFS = ["15m","1h"]

def sma(xs, n):
    if len(xs)<n: return [None]*len(xs)
    s=0; out=[None]*(n-1)
    for i,x in enumerate(xs):
        s+=x
        if i>=n: s-=xs[i-n]
        if i>=n-1: out.append(s/n)
    return out

def rsi(closes, n=14):
    rs=[None]*len(closes); gains=[0]*len(closes); losses=[0]*len(closes)
    for i in range(1,len(closes)):
        ch=closes[i]-closes[i-1]
        gains[i]=max(0,ch); losses[i]=max(0,-ch)
        if i>=n:
            ag=sum(gains[i-n+1:i+1])/n; al=sum(losses[i-n+1:i+1])/n
            if al==0: rs[i]=100
            else: rs[i]=100-100/(1+ag/al)
    return rs

def load(con, tab, days=90):
    cur=con.execute(f"SELECT timestamp,open,high,low,close,volume FROM '{tab}' WHERE timestamp>=? ORDER BY timestamp",
                    (int(time.time())-days*86400,))
    return cur.fetchall()

def metrics(eq):
    if not eq: return 0,0,0
    rets=[]
    for i in range(1,len(eq)):
        if eq[i-1]>0: rets.append((eq[i]-eq[i-1])/eq[i-1])
    ann = (1+ (st.mean(rets) if rets else 0))**365 - 1 if rets else 0
    peak=eq[0]; dd=0
    for x in eq:
        if x>peak: peak=x
        dd=max(dd, (peak-x)/peak if peak>0 else 0)
    sharpe=(st.mean(rets)/(st.pstdev(rets)+1e-9))*math.sqrt(365) if len(rets)>1 else 0
    return ann, dd, sharpe

def run(con, sym, tf):
    tab=f"{sym}_{tf}"
    rows=load(con, tab)
    if len(rows)<50: return None
    close=[r[4] for r in rows]
    m1=sma(close,10); m2=sma(close,30); r=rsi(close,14)
    pos=0; eq=10000.0; equity=[eq]; fee=0.0005
    for i,c in enumerate(close):
        if m1[i] is None or m2[i] is None or r[i] is None:
            equity.append(eq); continue
        buy  = (m1[i]>m2[i] and r[i]>50)
        sell = (m1[i]<m2[i] and r[i]<50)
        if pos==0 and buy:
            pos=1; entry=c; eq*= (1-fee)
        elif pos==1 and sell:
            eq*= (1+(c-entry)/entry); eq*= (1-fee); pos=0
        equity.append(eq)
    if pos==1:
        eq*= (1+(close[-1]-entry)/entry); eq*= (1-fee)
    ann,dd,sh=metrics(equity)
    return {"sym":sym,"tf":tf,"ann":ann,"dd":dd,"sharpe":sh}

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    con=sqlite3.connect(DB, timeout=30)
    res=[]
    for s in SYMS:
        for tf in TFS:
            try:
                r=run(con,s,tf)
                if r: res.append(r)
            except Exception:
                pass
    con.close()
    with open(OUT,"w",encoding="utf-8") as f:
        f.write("<h3>Quick Backtest (MA+RSI)</h3><table border=1><tr><th>symbol</th><th>tf</th><th>ann%</th><th>maxDD%</th><th>sharpe</th></tr>")
        for r in res:
            f.write(f"<tr><td>{r['sym']}</td><td>{r['tf']}</td><td>{round(r['ann']*100,2)}</td><td>{round(r['dd']*100,2)}</td><td>{round(r['sharpe'],2)}</td></tr>")
        f.write("</table>")
    print("backtest quick done:", OUT)

if __name__=="__main__": main()
