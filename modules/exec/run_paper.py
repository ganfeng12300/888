import os, sys, argparse, time, configparser, json
from modules.common.minimal_ansi import enable_virtual_terminal, colorize, CYAN, GREEN, YELLOW, RED
enable_virtual_terminal()

def main():
    ap = argparse.ArgumentParser(description="paper trading stub")
    ap.add_argument("--config", required=True)
    ap.add_argument("--risk", required=True)
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    print(colorize(">>> paper trading (stub)", CYAN))
    print("db:", cfg.get("paths","db"))
    print("guard rails (defaults): position 5%, leverage 5x, daily DD 10%, cap 80%, kill 20%/24h")
    # TODO: portfolio targeting, OMS, EMS, fills, PnL book.
    for i in range(1,6):
        print(f"paper step {i}/5 ..."); time.sleep(0.2)
    print(colorize("paper done (stub)", GREEN))

if __name__ == "__main__":
    main()
