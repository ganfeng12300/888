# -*- coding: utf-8 -*-
"""
机构�?· 2币种 · 彩色总进�?ETA 外层控制器（终极修正版）
- 仅回�?2 个币（默认：BTCUSDT ETHUSDT，可�?--symbols 覆盖�?
- 汇�?多策�?× Trial(25) 的全局进度（宽松匹配，�?N/25 也能推进�?
- 仅在真正生成 live_best_params.json & top_symbols.txt 后，才启动纸面面�?
- 颜色依赖 colorama（若缺失自动降级到无色）
"""
import argparse, os, sys, re, time, subprocess
from pathlib import Path

# ────────────────────────── 颜色�?──────────────────────────
try:
    from colorama import init as cinit, Fore, Style
    cinit(autoreset=True)
    C = dict(
        ok=Fore.GREEN+Style.BRIGHT,
        warn=Fore.YELLOW+Style.BRIGHT,
        err=Fore.RED+Style.BRIGHT,
        info=Fore.CYAN+Style.BRIGHT,
        emph=Fore.MAGENTA+Style.BRIGHT,
        bar=Fore.GREEN+Style.BRIGHT,
        dim=Style.DIM,
        rst=Style.RESET_ALL
    )
except Exception:
    class _D:  # 降级无色
        def __getattr__(self, _): return ""
    Fore=Style=_D()
    C = dict(ok="", warn="", err="", info="", emph="", bar="", dim="", rst="")

# ────────────────────────── 常量与正�?──────────────────────────
TRIALS_PER_STRAT = 25
RE_TRIAL = re.compile(r"(?P<n>\d{1,2})/25\b")   # 宽松匹配 N/25
RE_DONE  = re.compile(r"\b25/25\b")             # 单策略unit完成
RE_RUNID = re.compile(r"results[\\/](\d{8}-\d{6}-[0-9a-f]{8})")  # 可能�?run_id 打印

MILESTONE_NAMES = (
    "a5_optimized_params.csv",
    "a6_strategy_scores", "a7_blended_portfolio.csv",
    "final_portfolio.json", "live_best_params.json"
)

# ────────────────────────── 小工�?──────────────────────────
def draw_bar(pct: float, width=42, ch_full="�?, ch_empty="�?, color=C["bar"]):
    pct = max(0.0, min(1.0, pct))
    n = int(round(pct*width))
    return f"{color}[{'':<{width}}]{C['rst']}".replace(' ' * width, ch_full*n + ch_empty*(width-n)) + f" {pct*100:5.1f}%"

def fmt_eta(sec: float):
    if not (sec and sec > 0 and sec < 10*24*3600):
        return f"{C['dim']}ETA --:--{C['rst']}"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"ETA {h:02d}:{m:02d}:{s:02d}" if h else f"ETA {m:02d}:{s:02d}"

def ema(prev, val, alpha=0.18):
    return val if prev is None else prev*(1-alpha) + val*alpha

def latest_subdir(p: Path):
    subs = [d for d in p.iterdir() if d.is_dir()]
    if not subs: return None
    subs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subs[0]

def wait_for_exports(run_dir: Path, timeout_sec: int = 24*3600, poll_sec: float = 5.0):
    """阻塞等待真正的导出产物出现；返回 (ok, missing_list)"""
    must_files = ["live_best_params.json", "top_symbols.txt"]
    start = time.time()
    while True:
        missing = [f for f in must_files if not (run_dir/f).exists()]
        if not missing:
            return True, []
        if time.time() - start > timeout_sec:
            return False, missing
        time.sleep(poll_sec)

def start_paper_console(project_root: Path, db_path: str):
    """打开纸面实盘战情面板（独立窗口；对含空格/括号路径稳健�?""
    engine = project_root / "live_trading" / "execution_engine_binance_ws.py"
    if not engine.exists():
        print(f"{C['warn']}[WARN]{C['rst']} 未找�?{engine}，请改成你的纸面执行器路径�?)
        return
    subprocess.call([
        "cmd", "/c", "start", "", "powershell", "-NoExit", "-Command",
        f"& {{ Set-Location -LiteralPath '{project_root}'; "
        f"$env:PYTHONPATH='{project_root}'; "
        f"python '{engine}' --db '{db_path}' --mode paper --ui-rows 30 }}"
    ])

# ────────────────────────── 主流�?──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT","ETHUSDT"],  # 默认 2 个币
                    help="默认仅跑 BTCUSDT ETHUSDT；可自定义覆�?)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--strategies", type=int, default=8, help="每个币的策略数量（估算用�?)
    # 稳健层外挂参数（透传�?backtest_pro.py�?
    ap.add_argument("--spa", choices=["on","off"], default="on")
    ap.add_argument("--spa-alpha", dest="spa_alpha", type=float, default=0.05)
    ap.add_argument("--pbo", choices=["on","off"], default="on")
    ap.add_argument("--pbo-bins", dest="pbo_bins", type=int, default=10)
    ap.add_argument("--impact-recheck", dest="impact_recheck", choices=["on","off"], default="on")
    ap.add_argument("--wfo", choices=["on","off"], default="off")
    ap.add_argument("--wfo-train", dest="wfo_train", type=int, default=180)
    ap.add_argument("--wfo-test",  dest="wfo_test",  type=int, default=30)
    ap.add_argument("--wfo-step",  dest="wfo_step",  type=int, default=30)
    ap.add_argument("--tf-consistency", dest="tf_consistency", choices=["on","off"], default="on")
    ap.add_argument("--tf-consistency-w", dest="tf_consistency_w", type=float, default=0.2)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    outdir = (project_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    total_symbols = len(args.symbols)
    strategies = max(1, args.strategies)
    total_units = total_symbols * strategies  # unit=某币某策略的25个trial
    done_units = 0
    cur_trial = 0
    start_ts = time.time()
    eta_ema = None
    seen_run_id = None

    # ── Header ──
    print()
    print(f"{C['emph']}┌────────────────────────  S-A�?回测总控  ────────────────────────┐{C['rst']}")
    print(f"{C['emph']}│{C['rst']}  {C['info']}Symbols{C['rst']}: {', '.join(args.symbols)}")
    print(f"{C['emph']}│{C['rst']}  {C['info']}Trials/Strategy{C['rst']}: {TRIALS_PER_STRAT}   "
          f"{C['info']}Strategies/Symbol{C['rst']}: {strategies}   {C['info']}TopK{C['rst']}: {args.topk}")
    print(f"{C['emph']}│{C['rst']}  {C['info']}DB{C['rst']}: {args.db}")
    print(f"{C['emph']}└──────────────────────────────────────────────────────────────────┘{C['rst']}\n")

    # 内层命令（只 2 个币；可�?--symbols 覆盖�?
    inner = [
        sys.executable, "-u", str(project_root / "backtest" / "backtest_pro.py"),
        "--db", args.db,
        "--days", str(args.days),
        "--topk", str(args.topk),
        "--outdir", str(outdir),
        "--symbols", *args.symbols
    ]
    if args.spa == "on":
        inner += ["--spa", "on", "--spa-alpha", str(args.spa_alpha)]
    if args.pbo == "on":
        inner += ["--pbo", "on", "--pbo-bins", str(args.pbo_bins)]
    if args.impact_recheck == "on":
        inner += ["--impact-recheck", "on"]
    if args.wfo == "on":
        inner += ["--wfo", "on",
                  "--wfo-train", str(args.wfo_train),
                  "--wfo-test",  str(args.wfo_test),
                  "--wfo-step",  str(args.wfo_step)]
    if args.tf_consistency == "on":
        inner += ["--tf-consistency", "on",
                  "--tf-consistency-w", str(args.tf_consistency_w)]

    # 启动内层
    proc = subprocess.Popen(inner, cwd=project_root,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    # 渲染循环
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")

            # �?run_id
            mrun = RE_RUNID.search(line)
            if mrun: seen_run_id = mrun.group(1)

            # 识别 unit 完成 / trial 进度
            if RE_DONE.search(line):
                done_units += 1
                cur_trial = TRIALS_PER_STRAT
            else:
                mt = RE_TRIAL.search(line)
                if mt:
                    try:
                        cur_trial = int(mt.group("n"))
                    except Exception:
                        pass

            # 全局进度 & ETA
            unit_pct = cur_trial / TRIALS_PER_STRAT
            gprog = (done_units + unit_pct) / max(1, total_units)
            elapsed = time.time() - start_ts
            eta = elapsed * (1/gprog - 1) if gprog > 0 else None
            eta_ema = ema(eta_ema, eta) if eta is not None else eta_ema

            # 估算当前�?策略
            cur_symbol_idx = min(done_units // strategies + 1, total_symbols)
            cur_symbol_idx = max(1, cur_symbol_idx)
            cur_symbol = args.symbols[cur_symbol_idx-1]
            cur_unit_in_symbol = (done_units % strategies) + (1 if cur_trial < TRIALS_PER_STRAT else 0)
            cur_unit_in_symbol = min(cur_unit_in_symbol, strategies)

            # 绘制
            top_line = f"{C['info']}全局{C['rst']} {cur_symbol_idx}/{total_symbols}  {draw_bar(gprog)}  {fmt_eta(eta_ema)}"
            sub_line = f"{C['dim']}当前{C['rst']} {C['emph']}{cur_symbol}{C['rst']}  策略 {cur_unit_in_symbol}/{strategies}  Trials {cur_trial:02d}/{TRIALS_PER_STRAT}"

            sys.stdout.write("\r" + " " * 160 + "\r")
            sys.stdout.write(top_line + "\n")
            sys.stdout.write(sub_line + " " * 10 + "\r")
            sys.stdout.flush()

            # 仅打印真实里程碑关键字（不改变状态）
            if any(k in line for k in MILESTONE_NAMES):
                print("\n" + f"{C['ok']}�?里程碑{C['rst']}  " + line)

            # 常见告警
            if "UserWarning: X does not have valid feature names" in line:
                print("\n" + f"{C['warn']}�?Sklearn/LightGBM 特征名警告（不影响回测，但建议对齐特征列名）{C['rst']}")

        proc.wait()
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    # ── 结束判定（以文件为准�?──
    print("\n" + f"{C['info']}回测进程退出码{C['rst']}: {proc.returncode}")
    # run_id 兜底：取 results 下最新子目录
    run_dir = None
    if seen_run_id:
        run_dir = outdir / seen_run_id
    if not run_dir or not run_dir.exists():
        latest = latest_subdir(outdir)
        if latest:
            run_dir = latest
    if not run_dir:
        print(f"{C['err']}�?未找到结果目录（outdir 内无子目录）。请检�?backtest 是否启动成功。{C['rst']}")
        sys.exit(2)

    print(f"{C['info']}结果目录{C['rst']}: {run_dir}")
    ok, missing = wait_for_exports(run_dir, timeout_sec=24*3600, poll_sec=5.0)
    if not ok:
        print(f"{C['err']}�?未检测到导出产物：{', '.join(missing)}（等待超时）。不启动纸面面板。{C['rst']}")
        sys.exit(3)

    print(f"{C['ok']}�?回测阶段完成{C['rst']}  已生�?live_best_params.json / top_symbols.txt")
    # 启动纸面实盘战情面板
    start_paper_console(project_root, args.db)
    print(f"{C['emph']}�?已启动纸面实盘窗口（独立 PowerShell），参数从导出的 live_best_params.json / top_symbols.txt 读取{C['rst']}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + f"{C['warn']}中断退出{C['rst']}")
