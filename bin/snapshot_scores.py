# -*- coding: utf-8 -*-
import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "bin" else Path.cwd()
DATA = ROOT / "data"
SCORES_DIR = DATA / "scores"
REQ_COLS = ["symbol","tf","strategy","trades","ret","sharpe","sortino","mdd","calmar","ulcer","upi","dsr","spa_alpha","pbo","cagr","vol","turnover","asof_utc","score"]

def _find_latest_raw():
    cand = []
    for p in SCORES_DIR.glob("strategy_scores_*.csv"):
        cand.append(p)
    # fallback: any csv in scores
    if not cand:
        for p in SCORES_DIR.glob("*.csv"):
            if p.name.lower() != "latest.csv": cand.append(p)
    if not cand:
        raise FileNotFoundError("no raw scores csv found")
    cand.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cand[0]

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    src = _find_latest_raw()
    df = pd.read_csv(src)
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    num_cols = ["trades","ret","sharpe","sortino","mdd","calmar","ulcer","upi","dsr","spa_alpha","pbo","cagr","vol","turnover","score"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "asof_utc" not in df.columns or df["asof_utc"].isna().all():
        df["asof_utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    # normalize mdd to negative (if positive by mistake)
    if df["mdd"].notna().any() and (df["mdd"] > 0).mean() > 0.8:
        df["mdd"] = -df["mdd"].abs()
    # emit snapshot
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snap_dir = SCORES_DIR / ts
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_csv = snap_dir / "strategy_scores.csv"
    df.to_csv(out_csv, index=False)
    # latest.csv (overwrite)
    df.to_csv(SCORES_DIR / "latest.csv", index=False)
    # manifest
    manifest = {
        "ts_utc": ts,
        "source": str(src),
        "rows": int(len(df)),
        "md5": _md5(out_csv),
        "required_cols": REQ_COLS,
    }
    with open(snap_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] snapshot -> {out_csv}  rows={len(df)}")

if __name__ == "__main__":
    main()
