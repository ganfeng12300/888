# -*- coding: utf-8 -*-
"""
Enterprise · 2 Symbols · Robust Progress + Real Export Gate + Preflight Checks
- 默认�?BTCUSDT/ETHUSDT�?-symbols 可覆盖）
- 预检：Python/DB/backtest_pro.py/写权�?
- 进度：宽松匹�?N/25，忽�?A6-SHIM；全局 & 当前�?策略 & 平滑 ETA
- 完成：只有当 D:\\SHUJU888\\results\\<run_id>/ 出现 live_best_params.json �?top_symbols.txt 才视为完成并启动纸面面板
- 路径含空�?括号安全
"""
import argparse, os, sys, re, time, subprocess, shutil
from pathlib import Path
from datetime import datetime

# ─────────────── 色彩�?───────────────
try:
    from colorama import init as cinit, Fore, Style
    cinit(autoreset=True)
    C = dict(ok=Fore.GREEN+Style.BRIGHT, warn=Fore.YELLOW+Style.BRIGHT, err=Fore.RED+Style.BRIGHT,
             info=Fore.CYAN+Style.BRIGHT, emph=Fore.MAGENTA+Style.BRIGHT, bar=Fore.GREEN+Style.BRIGHT,
             dim=Style.DIM, rst=Style.RESET_ALL)
except Exception:
    class _D:  # 降级
        def __getattr__(self, _): return ""
    Fore=Style=_D()  # type: ignore
    C = dict(ok="", warn="", err="", info="", emph="", bar="", dim="", rst="")

# ─────────────── 常量与正�?───────────────
TRIALS_PER_STRAT = 25
RE_TRIAL = re.compile(r"(?P<n>\d{1,2})/25\b")     # 宽松 N/25
RE_DONE  = re.compile(r"\b25/25\b")               # 单策略unit完成
RE_RUNID = re.compile(r"results[\\/](\d{8}-\d{6}-[0-9a-f]{8})")
A6_SHIM_HINT = "A6-SHIM"                           # 忽略该行对状态的影响

MUST_EXPORTS = ("live_best_params.json", "top_symbols.txt")

# ─────────────── 工具函数 ───────────────
def draw_bar(pct: float, width=42, ch_full="�?, ch_empty="�?, color=C["bar"]):
    pct = max(0.0, min(1.0, pct)); n = int(round(pct*width))
    return f"{color}[{'':<{width}}]{C['rst']}".replace(' ' * width, ch_full*n + ch_empty*(width-n)) + f" {pct*100:5.1f}%"

def fmt_eta(sec: float):
    if not (sec and sec>0 and sec<10*24*3600): return f"{C['dim']}ETA --:--{C['rst']}"
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"ETA {h:02d}:{m:02d}:{s:02d}" if h else f"ETA {m:02d}:{s:02d}"

def ema(prev, val, alpha=0.18): return val if prev is None else prev*(1-alpha)+val*alpha

def latest_subdir(p: Path):
    subs=[d for d in p.iterdir() if d.is_dir()]
    if not subs: return None
    subs.sort(key=lambda d: d.stat().st_mtime, reverse=True); return subs[0]

def ensure_writable(dirpath: Path):
    try:
        dirpath.mkdir(parents=True, exist_ok=True)
        test = dirpath / ".perm_test"
        test.write_text("ok", encoding="utf-8"); test.unlink(missing_ok=True)
        return True, ""
    except Exception as e:
        return False, str(e)

def start_paper_console(project_root: Path, db_path: str):
    """纸面面板（独立窗口）"""
    engine = project_root / "live_trading" / "execution_engine_binance_ws.py"
    if not engine.exists():
        print(f"{C['warn']}[WARN]{C['rst']} 未找�?{engine}，请改成你的纸面执行器�?)
        return
    subprocess.call([
        "cmd","/c","start","", "powershell","-NoExit","-Command",
        f"& {{ Set-Location -LiteralPath '{project_root}'; "
        f"$env:PYTHONPATH='{project_root}'; "
        f"python '{engine}' --db '{db_path}' --mode paper --ui-rows 30 }}"
    ])

def tail_write(fh, text): 
    try: fh.write(text); fh.flush()
    except Exception: pass

# ─────────────── 主流�?───────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT","ETHUSDT"])
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--strategies", type=int, default=8)

    # 稳健层透传
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
    backtest_py = project_root / "backtest" / "backtest_pro.py"
    outdir = (project_root / args.outdir).resolve()
    logs_dir = (project_root / "logs").resolve()

    # ── 预检 ──
    print(f"{C['emph']}══════════ 预检 Preflight ══════════{C['rst']}")
    print(f"{C['info']}Python{C['rst']}: {sys.executable}")
    if not Path(args.db).exists():
        print(f"{C['err']}�?DB 不存在：{args.db}{C['rst']}"); sys.exit(21)
    if not backtest_py.exists():
        print(f"{C['err']}�?未找到回测脚本：{backtest_py}{C['rst']}")
        print(f"{C['warn']}请确认项目路径、或�?backtest_pro.py 放到 backtest/ 下。{C['rst']}"); sys.exit(22)
    ok,msg = ensure_writable(outdir)
    if not ok:
        print(f"{C['err']}�?输出目录不可写：{outdir}  原因：{msg}{C['rst']}"); sys.exit(23)
    ok2,_ = ensure_writable(logs_dir)
    if not ok2:
        print(f"{C['warn']}�?无法创建日志目录：{logs_dir}（不影响回测）{C['rst']}")
    print(f"{C['ok']}�?预检通过{C['rst']}\n")

    # ── Header ──
    print(f"{C['emph']}┌────────────  S-A �?回测总控  ────────────┐{C['rst']}")
    print(f"{C['emph']}│{C['rst']}  {C['info']}DB{C['rst']}: {args.db}")
    print(f"{C['emph']}│{C['rst']}  {C['info']}Symbols{C['rst']}: {', '.join(args.symbols)}")
    print(f"{C['emph']}│{C['rst']}  {C['info']}Trials/Strat{C['rst']}: {TRIALS_PER_STRAT}   "
          f"{C['info']}Strats/Symbol{C['rst']}: {args.strategies}   {C['info']}TopK{C['rst']}: {args.topk}")
    print(f"{C['emph']}└──────────────────────────────────────────┘{C['rst']}\n")

    # ── 内层命令 ──
    inner = [
        sys.executable, "-u", str(backtest_py),
        "--db", args.db, "--days", str(args.days), "--topk", str(args.topk),
        "--outdir", str(outdir), "--symbols", *args.symbols
    ]
    if args.spa == "on": inner += ["--spa","on","--spa-alpha",str(args.spa_alpha)]
    if args.pbo == "on": inner += ["--pbo","on","--pbo-bins",str(args.pbo_bins)]
    if args.impact_recheck == "on": inner += ["--impact-recheck","on"]
    if args.wfo == "on":
        inner += ["--wfo","on","--wfo-train",str(args.wfo_train),"--wfo-test",str(args.wfo_test),"--wfo-step",str(args.wfo_step)]
    if args.tf_consistency == "on":
        inner += ["--tf-consistency","on","--tf-consistency-w",str(args.tf_consistency_w)]

    cmd_str = " ".join([f'"{p}"' if " " in p else p for p in inner])
    print(f"{C['info']}启动命令{C['rst']}: {cmd_str}\n")

    # ── 启动 & 日志落盘 ──
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outer_log = logs_dir / f"outer_wrap_{ts}.log"
    lf = outer_log.open("w", encoding="utf-8")

    total_symbols = len(args.symbols)
    strategies = max(1, args.strategies)
    total_units = total_symbols * strategies
    done_units = 0; cur_trial = 0
    start_ts = time.time(); eta_ema = None
    seen_run_id = None

    proc = subprocess.Popen(inner, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            # 写日�?
            tail_write(lf, line + "\n")

            # 忽略 A6-SHIM 提示
            if A6_SHIM_HINT in line:
                print(f"{C['ok']}�?里程碑{C['rst']}  {line}")  # 仅展示，不影响状�?
                continue

            # run_id
            mrun = RE_RUNID.search(line)
            if mrun: seen_run_id = mrun.group(1)

            # unit 完成 / trial
            if RE_DONE.search(line):
                done_units += 1; cur_trial = TRIALS_PER_STRAT
            else:
                mt = RE_TRIAL.search(line)
                if mt:
                    try: cur_trial = int(mt.group("n"))
                    except: pass

            # 进度
            unit_pct = cur_trial / TRIALS_PER_STRAT
            gprog = (done_units + unit_pct) / max(1, total_units)
            elapsed = time.time() - start_ts
            eta = elapsed * (1/gprog - 1) if gprog>0 else None
            eta_ema = ema(eta_ema, eta) if eta is not None else eta_ema

            cur_symbol_idx = min(done_units // strategies + 1, total_symbols); cur_symbol_idx = max(1, cur_symbol_idx)
            cur_symbol = args.symbols[cur_symbol_idx-1]
            cur_unit_in_symbol = (done_units % strategies) + (1 if cur_trial < TRIALS_PER_STRAT else 0)
            cur_unit_in_symbol = min(cur_unit_in_symbol, strategies)

            top_line = f"{C['info']}全局{C['rst']} {cur_symbol_idx}/{total_symbols}  {draw_bar(gprog)}  {fmt_eta(eta_ema)}"
            sub_line = f"{C['dim']}当前{C['rst']} {C['emph']}{cur_symbol}{C['rst']}  策略 {cur_unit_in_symbol}/{strategies}  Trials {cur_trial:02d}/{TRIALS_PER_STRAT}"

            sys.stdout.write("\r" + " " * 160 + "\r"); sys.stdout.write(top_line + "\n"); sys.stdout.write(sub_line + " " * 10 + "\r"); sys.stdout.flush()

        proc.wait()
    finally:
        try:
            lf.close()
            if proc.poll() is None: proc.terminate()
        except Exception: pass

    print("\n" + f"{C['info']}回测进程退出码{C['rst']}: {proc.returncode}")
    if proc.returncode not in (0,):
        print(f"{C['warn']}�?回测进程非零退出，已将原始日志写入：{outer_log}{C['rst']}")

    # ── 结果目录定位（run_id 兜底�?──
    run_dir = (outdir / seen_run_id) if seen_run_id else latest_subdir(outdir)
    if not run_dir or not run_dir.exists():
        print(f"{C['err']}�?未找到结果目录；请检�?{outer_log} 获取错误详情。{C['rst']}")
        sys.exit(31)

    print(f"{C['info']}结果目录{C['rst']}: {run_dir}")

    # ── 等待真实导出 ──
    print(f"{C['info']}等待导出产物{C['rst']}: {', '.join(MUST_EXPORTS)}")
    start_wait = time.time()
    missing = list(MUST_EXPORTS)
    while True:
        missing = [f for f in MUST_EXPORTS if not (run_dir / f).exists()]
        if not missing: break
        if time.time() - start_wait > 24*3600:
            print(f"{C['err']}�?等待导出超时，仍缺：{', '.join(missing)}；不启动纸面面板。{C['rst']}")
            print(f"{C['info']}请查看日志{C['rst']}: {outer_log}")
            sys.exit(32)
        time.sleep(5)

    print(f"{C['ok']}�?回测阶段完成{C['rst']}  已检测到导出文件：{', '.join(MUST_EXPORTS)}")

    # ── 启动纸面面板 ──
    start_paper_console(project_root, args.db)
    print(f"{C['emph']}�?已启动纸面窗口（独立 PowerShell），将按导出参数实时模拟 PAPER{C['rst']}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + f"{C['warn']}中断退出{C['rst']}")
