# -*- coding: utf-8 -*-
"""
check_all_compile.py �?全系统编译自检（机构级�?
遍历项目下所�?.py 文件，尝试编译，报告错误文件
"""

import pathlib, py_compile, sys

fails = []
for p in pathlib.Path(".").rglob("*.py"):
    if "__pycache__" in str(p):
        continue
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as e:
        fails.append((str(p), e))

if not fails:
    print("�?ALL OK �?全部 .py 文件编译通过")
    sys.exit(0)
else:
    print("�?以下文件编译失败�?)
    for a, b in fails:
        print(f"  {a}: {b}")
    sys.exit(1)
