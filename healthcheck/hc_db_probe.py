import sys, json, sqlite3, os
from datetime import datetime, timezone

OUT = r'D:\SQuant_Pro\healthcheck\db_probe_out.json'
DB  = r'D:\quant_system_v2\data\market_data.db'
IS_QUICK = bool(1)

def write(o):
    try:
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(o, f, ensure_ascii=False)
    except Exception as e:
        print(json.dumps({'error': f'write_failed: {e}'}))

try:
    import json as _json_check
    SYMS = _json_check.loads(r'["BTCUSDT","ETHUSDT","OPUSDT","RNDRUSDT","ARBUSDT","FETUSDT","SOLUSDT","LDOUSDT","INJUSDT","AVAXUSDT","APTUSDT","NEARUSDT"]')
    TFS  = _json_check.loads(r'["5m","15m","30m","1h","2h","4h","1d"]')
except Exception as e:
    write({'error': f'config_json_failed: {e}'}); sys.exit(0)

def connect_ro(path):
    try:
        uri = f'file:{path}?mode=ro'
        con = sqlite3.connect(uri, uri=True, timeout=15, isolation_level=None)
        try:
            con.execute('PRAGMA query_only=ON')
            con.execute('PRAGMA temp_store=2')        # MEMORY
            con.execute('PRAGMA cache_size=-200000')  # ~200MB
        except Exception:
            pass
        return con
    except Exception as e:
        write({'error': f'connect_failed: {e}'}); sys.exit(0)

bars_per_day = {'5m':288,'15m':96,'30m':48,'1h':24,'2h':12,'4h':6,'1d':1}
now = datetime.now(timezone.utc); now_ts = int(now.timestamp())
cut30  = now_ts - 30*24*3600
cut90  = now_ts - 90*24*3600
cut365 = now_ts - 365*24*3600

res = {'now_ts': now_ts, 'pairs': [], 'agg': {}, 'quick': IS_QUICK}

con = connect_ro(DB)
try:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = set(r[0] for r in rows)
except Exception as e:
    write({'error': f'tables_query_failed: {e}'}); sys.exit(0)

if not tables:
    write({'error': 'no_tables_found'}); sys.exit(0)

def tbl(sym, tf): return f"{sym}_{tf}"

SQL = "SELECT MAX(timestamp), " \
      "SUM(CASE WHEN timestamp>=? THEN 1 ELSE 0 END), " \
      "SUM(CASE WHEN timestamp>=? THEN 1 ELSE 0 END), " \
      "SUM(CASE WHEN timestamp>=? THEN 1 ELSE 0 END) FROM '{}'"

pairs = []
for s in SYMS:
    for tf in TFS:
        name = tbl(s, tf)
        if name not in tables:
            pairs.append({'symbol': s, 'tf': tf, 'exists': False}); continue
        try:
            q = SQL.format(name)
            row = cur.execute(q, (cut30, cut90, cut365)).fetchone()
            latest, c30, c90, c365 = row if row else (None, None, None, None)
            info = {'symbol': s, 'tf': tf, 'exists': True}
            info['latest'] = int(latest) if latest is not None else None
            info['counts'] = {'30': c30, '90': c90, '365': c365}
            bpd = bars_per_day.get(tf)
            cov = {}
            for days, cnt in info['counts'].items():
                cov[days] = None if (cnt is None or bpd is None) else round(min(1.0, (cnt or 0)/(int(days)*bpd)), 4)
            info['coverage'] = cov
            info['future_k'] = (info['latest'] is not None and info['latest'] > now_ts + 120)
            info['last_age_sec'] = (now_ts - info['latest']) if info['latest'] is not None else None
            pairs.append(info)
        except Exception as e:
            pairs.append({'symbol': s, 'tf': tf, 'exists': False, 'error': f'probe_failed: {e}'})

res['pairs'] = pairs

try:
    ok30 = sum(1 for p in pairs if p.get('exists') and p.get('coverage',{}).get('30') is not None and p['coverage']['30'] >= 0.95)
    ok90 = sum(1 for p in pairs if p.get('exists') and p.get('coverage',{}).get('90') is not None and p['coverage']['90'] >= 0.95)
    ok365= sum(1 for p in pairs if p.get('exists') and p.get('coverage',{}).get('365') is not None and p['coverage']['365']>= 0.95)
    tot  = sum(1 for p in pairs if p.get('exists'))
    fut  = sum(1 for p in pairs if p.get('exists') and p.get('future_k'))
    lag5m= [p for p in pairs if p.get('exists') and p.get('tf')=='5m' and p.get('last_age_sec') not in (None,) and p['last_age_sec']>1800]
    res['agg'] = {'exists': tot, 'ok30_ge95': ok30, 'ok90_ge95': ok90, 'ok365_ge95': ok365, 'future_k': fut, 'lag5m_over_30min': len(lag5m)}
except Exception as e:
    write({'error': f'agg_calc_failed: {e}'}); sys.exit(0)

write(res)
