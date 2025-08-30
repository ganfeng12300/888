# -*- coding: utf-8 -*-
# File: tools/run_1symbol_parallel_strategies_safe.py
"""
单币种机构级外层总控�?
- A1..A4(指标: MA/BOLL/ATR/REVERSAL) 并行
- A5..A8(模型: LGBM/XGB/LSTM/ENSEMBLE) 串行（防 GPU 抢占�?
- 容错不中断、彩色进度、结束大字、自动喂实盘
依赖�?
    pip install colorama pyfiglet
"""
import os, sys, time, subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from colorama import Fore, Style, init as colorama_init
import argparse

colorama_init(autoreset=True)
ROOT = Path(__file__).resolve().parents[1]
BACKTEST_PY = ROOT / "backtest" / "backtest_pro.py"
PYEXE = sys.executable
RESULTS_DIR_DEFAULT = ROOT / "results" / "parallel_run"

INDICATOR_STRATS = ["A1","A2","A3","A4"]   # MA/BOLL/ATR/REVERSAL
MODEL_STRATS     = ["A5","A6","A7","A8"]   # LGBM/XGB/LSTM/ENSEMBLE

DEFAULT_FEATURE_FLAGS = [
    "--spa","on","--spa-alpha","0.05",
    "--pbo","on","--pbo-bins","10",
    "--impact-recheck","on",
    "--tf-consistency","on","--tf-consistency-w","0.2",
]

def ts(): return datetime.now().strftime("%H:%M:%S")

def banner_big(msg, color=Fore.GREEN):
    try:
        import pyfiglet
        print(color + Style.BRIGHT + pyfiglet.figlet_format(msg, font="slant"))
    except Exception:
        print(color + Style.BRIGHT + f"\n==== {msg} ====\n")

def print_hdr(db, symbol, days, topk, outdir, workers):
    msg = f"""
┌{'─'*88}�?
�? {Style.BRIGHT}机构级回测总控（单币种并行）{Style.RESET_ALL}                                         �?
�? Symbol: {Fore.CYAN}{symbol}{Style.RESET_ALL}   Days: {days}   TopK: {topk}   并行(指标): {workers:<2}       �?
�? DB    : {db:<74}�?
�? OutDir: {outdir:<74}�?
└{'─'*88}�?
""".rstrip("\n")
    print(msg)

def print_row(status, strat, rc=None, best=None):
    col = {"RUN":Fore.CYAN, "OK":Fore.GREEN, "ERR":Fore.RED, "SKIP":Fore.YELLOW}.get(status, Fore.WHITE)
    line = f"[{ts()}] {col}{status:<4}{Style.RESET_ALL} | 策略 {Fore.MAGENTA}{strat:<9}{Style.RESET_ALL}"
    if best is not None: line += f" | {best}"
    if rc is not None:   line += f" | rc {rc}"
    print(line)

def subprocess_stream(cmd, log_file=None):
    lf = open(log_file, "w", encoding="utf-8", errors="ignore") if log_file else None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_line = None
    for line in iter(proc.stdout.readline, ""):
        if lf: lf.write(line)
        low = line.lower()
        if "best loss" in low or "best score" in low or "best" in low:
            last_line = line.strip()
    proc.wait()
    if lf: lf.close()
    return proc.returncode, last_line

def run_one_strategy(db, days, symbol, topk, outdir, strat_tag, extra_flags):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    log_file = outdir / f"run_{symbol}_{strat_tag}_{int(time.time())}.log"
    cmd = [
        PYEXE, "-u", str(BACKTEST_PY),
        "--db", str(db), "--days", str(days),
        "--topk", str(topk), "--outdir", str(outdir),
        "--symbols", symbol,
        "--only-strategy", strat_tag,
    ] + list(extra_flags)
    print_row("RUN", strat_tag)
    rc, best = subprocess_stream(cmd, log_file=str(log_file))
    if rc == 0: print_row("OK", strat_tag, rc=rc, best=best)
    else:       print_row("ERR", strat_tag, rc=rc, best=best)
    return strat_tag, rc, str(log_file), best

def feed_to_live(outdir):
    outdir = Path(outdir)
    p_params = outdir / "live_best_params.json"
    p_syms   = outdir / "top_symbols.txt"
    print("\n" + Fore.CYAN + "产物检测：" + Style.RESET_ALL)
    print(f"  live_best_params.json : {Fore.GREEN if p_params.exists() else Fore.RED}{p_params.exists()}{Style.RESET_ALL} -> {p_params}")
    print(f"  top_symbols.txt       : {Fore.GREEN if p_syms.exists() else Fore.RED}{p_syms.exists()}{Style.RESET_ALL} -> {p_syms}")
    ws_script = (ROOT / "tools" / "ws_paper.py")
    if p_params.exists() and p_syms.exists() and ws_script.exists():
        ps = f'Start-Process powershell -ArgumentList "-NoExit","-Command","python -u \\"{ws_script}\\" --params \\"{p_params}\\" --symbols \\"{p_syms}\\""'
        subprocess.Popen(["powershell","-Command", ps], creationflags=subprocess.CREATE_NO_WINDOW)
        print(Fore.GREEN + "�?已尝试启动纸面实盘窗口（独立 PowerShell）�? + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "未自动启动实盘（缺喂入脚本或产物），路径已展示，可手动喂入�? + Style.RESET_ALL)

def set_num_threads_env(num=16):
    os.environ.setdefault("OMP_NUM_THREADS", str(num))
    os.environ.setdefault("MKL_NUM_THREADS", str(num))
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(num))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",   default=r"D:\quant_system_v2\data\market_data.db")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--outdir", default=str(RESULTS_DIR_DEFAULT))
    ap.add_argument("--workers", type=int, default=3, help="指标类策略并行进程数")
    ap.add_argument("--no-feed", action="store_true", help="只导出，不启动喂实盘")
    # 强功能（如回测内核无对应参数，忽略无影响�?
    ap.add_argument("--spa", default="on", choices=["on","off"])
    ap.add_argument("--spa-alpha", default="0.05")
    ap.add_argument("--pbo", default="on", choices=["on","off"])
    ap.add_argument("--pbo-bins", default="10")
    ap.add_argument("--impact-recheck", default="on", choices=["on","off"])
    ap.add_argument("--tf-consistency", default="on", choices=["on","off"])
    ap.add_argument("--tf-consistency-w", default="0.2")
    args = ap.parse_args()

    extra_flags = [
        "--spa", args.spa, "--spa-alpha", str(args.spa_alpha),
        "--pbo", args.pbo, "--pbo-bins", str(args.pbo_bins),
        "--impact-recheck", args.impact_recheck,
        "--tf-consistency", args.tf_consistency, "--tf-consistency-w", str(args.tf_consistency_w),
    ]

    set_num_threads_env(16)
    print_hdr(args.db, args.symbol, args.days, args.topk, args.outdir, args.workers)

    errors = []

    # 1) 指标类策�?并行
    from functools import partial
    task = partial(run_one_strategy, args.db, args.days, args.symbol, args.topk, args.outdir, extra_flags=extra_flags)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = { ex.submit(task, s): s for s in INDICATOR_STRATS }
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                strat, rc, logf, best = fut.result()
                if rc != 0: errors.append((strat, rc, logf))
            except Exception as e:
                errors.append((s, -1, f"exception: {e}"))
                print_row("ERR", s, rc=-1)

    # 2) 模型类策�?串行（避免显存争抢）
    for s in MODEL_STRATS:
        strat, rc, logf, best = run_one_strategy(args.db, args.days, args.symbol, args.topk, args.outdir, s, extra_flags)
        if rc != 0: errors.append((strat, rc, logf))

    # 3) 汇�?
    print("\n" + Fore.CYAN + "�?*72 + Style.RESET_ALL)
    if errors:
        print(Fore.YELLOW + "完成（含报错策略已跳过）�? + Style.RESET_ALL)
        for s, rc, lf in errors:
            print(f"  {Fore.RED}{s}{Style.RESET_ALL} rc={rc}  日志：{lf}")
    else:
        print(Fore.GREEN + "全部策略已完成，无错误�? + Style.RESET_ALL)

    # 4) 大字
    banner_big("回測完成!", color=Fore.GREEN)

    # 5) 自动喂实�?
    if not args.no_feed:
        feed_to_live(args.outdir)
        banner_big("已喂�?/ 已啟�?, color=Fore.CYAN)
    else:
        print(Fore.YELLOW + "按需启动实盘：已跳过自动喂入�?-no-feed）�? + Style.RESET_ALL)

if __name__ == "__main__":
    main()
