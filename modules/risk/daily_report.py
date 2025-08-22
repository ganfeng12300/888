import os, sys, argparse, datetime as dt
from modules.common.minimal_ansi import enable_virtual_terminal, colorize, CYAN, GREEN
enable_virtual_terminal()

def main():
    ap = argparse.ArgumentParser(description="daily report stub")
    ap.add_argument("--config", required=True)
    ap.add_argument("--risk", required=True)
    args = ap.parse_args()
    outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "reports")
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(outdir, f"daily_report_{ts}.html"))
    os.makedirs(outdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><h1>Daily Report (stub)</h1>")
    print(colorize(f">>> daily report generated: {path}", CYAN))
    print(colorize("done", GREEN))

if __name__ == "__main__":
    main()
