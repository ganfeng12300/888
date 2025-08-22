import os, sys, ctypes
RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"; UNDER="\033[4m"
BLACK,RED,GREEN,YELLOW,BLUE,MAGENTA,CYAN,WHITE=[f"\033[3{n}m" for n in range(8)]
B_BLACK,B_RED,B_GREEN,B_YELLOW,B_BLUE,B_MAGENTA,B_CYAN,B_WHITE=[f"\033[9{n}m" for n in range(8)]
BG_BLACK,BG_RED,BG_GREEN,BG_YELLOW,BG_BLUE,BG_MAGENTA,BG_CYAN,BG_WHITE=[f"\033[4{n}m" for n in range(8)]
def enable_virtual_terminal():
    if os.name != "nt": return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception: pass
def colorize(msg, color): return f"{color}{msg}{RESET}"
def bar(i, n, width=40):
    p = 0 if n<=0 else min(1.0, i/float(n))
    fill = int(p*width); return f"[{'#'*fill}{'.'*(width-fill)}] {p*100:5.1f}%"
def stepprint(step, total, text):
    sys.stdout.write(f"\r{BOLD}{text}{RESET}  {bar(step,total)}"); sys.stdout.flush()
def doneprint(text): print(f"\r{text}  {bar(1,1)}")
