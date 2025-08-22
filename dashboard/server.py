# -*- coding: utf-8 -*-
from flask import Flask, render_template_string
import os, pandas as pd
ROOT=r"D:\SQuant_Pro"; app=Flask(__name__)
TPL="""
<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>交易大屏</title>
<style>body{background:#0b1220;color:#e5e9f0;font-family:Segoe UI,Arial}
.kpi{display:inline-block;background:#131c2b;padding:14px 18px;margin:8px;border-radius:12px;min-width:180px}
.kpi .v{font-size:26px;font-weight:700}.ok{color:#66d9a3}.bad{color:#ff6b6b}.mid{color:#ffd166}
table{width:100%;border-collapse:collapse;margin-top:10px}th,td{padding:8px;border-bottom:1px solid #1f2a44}tr:hover{background:#101a2b}
.green{color:#2ecc71}.red{color:#e74c3c}
</style></head><body>
<h2>📊 SQuant 交易大屏（Paper | 费率0.05% | Cross 10x | 单笔5% | 安全刹车60%）</h2>
<div>
 <div class="kpi"><div>总收益(含费)</div><div class="v {{'ok' if kpi.pnl>=0 else 'bad'}}">{{'{:.2f}'.format(kpi.pnl)}}</div></div>
 <div class="kpi"><div>胜率(当日)</div><div class="v {{'ok' if kpi.win>=0.5 else 'bad'}}">{{'{:.1%}'.format(kpi.win)}}</div></div>
 <div class="kpi"><div>成交笔数</div><div class="v">{{kpi.n}}</div></div>
</div>
<h3>📈 实时成交（最近30条）</h3>
<table><thead><tr><th>时间</th><th>合约</th><th>方向</th><th>数量</th><th>价格</th><th>手续费</th><th>实现PnL</th></tr></thead>
<tbody>
{% for r in rows %}
 <tr><td>{{r.ts}}</td><td>{{r.name_cn}} ({{r.symbol}})</td>
 <td class="{{'green' if r.side=='buy' else 'red'}}">{{r.side}}</td>
 <td>{{r.qty}}</td><td>{{r.price}}</td><td>{{r.fee}}</td>
 <td class="{{'green' if r.pnl>=0 else 'red'}}">{{r.pnl}}</td></tr>
{% endfor %}
</tbody></table></body></html>"""
def kpi():
    p=os.path.join(ROOT,'logs','paper')
    files=[os.path.join(p,x) for x in os.listdir(p) if x.startswith('trades_')]
    pnl=0; n=0; win=0
    if files:
        df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        pnl=float(df['pnl'].sum()-df['fee'].sum()); n=len(df); win=float((df['pnl']>0).mean()) if n else 0.0
    return {'pnl':pnl,'n':n,'win':win}
@app.route("/")
def home():
    p=os.path.join(ROOT,'logs','paper')
    files=[os.path.join(p,x) for x in os.listdir(p) if x.startswith('trades_')]
    rows=[]
    if files:
        df=pd.read_csv(sorted(files)[-1]).tail(30)
        rows=df.to_dict(orient='records')
    return render_template_string(TPL, kpi=kpi(), rows=rows)
if __name__=='__main__':
    app.run(host='127.0.0.1', port=8088, debug=False)
