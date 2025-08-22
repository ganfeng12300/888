import os, sys, argparse, time, configparser
from modules.common.minimal_ansi import enable_virtual_terminal, colorize, CYAN, GREEN
enable_virtual_terminal()

def main():
    ap = argparse.ArgumentParser(description="backtest stub")
    ap.add_argument("--config", required=True)
    ap.add_argument("--risk", required=True)
    ap.add_argument("--wl", required=True)
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    print(colorize(">>> backtest engine (stub)", CYAN))
    print("modes.paper =", cfg.getboolean("modes","paper"))
    # TODO: real walk-forward CV, slippage/fees, metrics, model registry.
    time.sleep(0.5)
    print(colorize("backtest done (stub)", GREEN))

if __name__ == "__main__":
    main()
