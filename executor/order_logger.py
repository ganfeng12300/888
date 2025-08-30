import json
from datetime import datetime
import os

LOG_PATH = "D:/quant_system_pro (4)/logs/orders/"
os.makedirs(LOG_PATH, exist_ok=True)

def log_order(signal, score, dry_run=True, result=None, error=None, reason=None):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": signal["symbol"],
        "side": signal["side"],
        "qty": signal["qty"],
        "strategy": signal.get("strategy", "unknown"),
        "score": round(score, 2),
        "mode": "dry-run" if dry_run else "real",
        "reason": reason,
        "result": result,
        "error": error,
    }
    filename = f"{LOG_PATH}{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    with open(filename, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\\n")
