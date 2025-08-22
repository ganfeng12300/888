import sys, os, sqlite3, time, csv
DB  = sys.argv[1] if len(sys.argv)>1 else r"D:\quant_system_v2\data\market_data.db"
OUT = sys.argv[2] if len(sys.argv)>2 else r"D:\SQuant_Pro\reports\optimize_quick.csv"
SYM = sys.argv[3] if len(sys.argv)>3 else "BTCUSDT"
TF  = sys.argv[4] if len(sys.argv)>4 else "1h"

def sma(xs, n):
    if len(xs)<n: return [None]*len(xs)
    s=0; out=[None]*(n-1)
    for i,x in enumerate(xs):
        s+=x
        if i>=n: s-=xs[i-n]
        if i>=n-1: out.append(s/n)
    return out

def load(db, tab, days=120):
    con=sqlite3.connect(db, timeout=30)
    cur=con.execute(f"SELECT timestamp,close FROM '{tab}' WHERE timestamp>=? ORDER BY timestamp",
                    (int(time.time())-days*86400,))
    xs=[x[1] for x in cur.fetchall()]
    con.close()
    return xs

def equity(cl, f, s):
    fee=0.0005; pos=0; eq=10000.0
    m1=sma(cl,f); m2=sma(cl,s)
    for i,c in enumerate(cl):
        if m1[i] is None or m2[i] is None: continue
        if pos==0 and m1[i]>m2[i]:
            pos=1; entry=c; eq*= (1-fee)
        elif pos==1 and m1[i]<m2[i]:
            eq*= (1+(c-entry)/entry); eq*= (1-fee); pos=0
    if pos==1:
        eq*= (1+(cl[-1]-entry)/entry); eq*= (1-fee)
    return eq

cl = load(DB, f"{SYM}_{TF}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,"w",newline="") as w:
    cw=csv.writer(w); cw.writerow(["fast","slow","equity"])
    for f in (5,10,15,20):
        for s in (20,30,40,60):
            if f>=s: continue
            cw.writerow([f,s,round(equity(cl,f,s),2)])
print("optimize quick done:", OUT)
