"""Minimal pytest-like runner (pytest is not installed in the 2dgs-slam env).
usage: python tests/run_tests.py tests/test_plane_anchor.py [name_filter]
"""
import importlib.util
import sys
import time
import traceback
import types

# test modules import pytest only for decorators; a stub is enough
sys.modules.setdefault("pytest", types.ModuleType("pytest"))

path = sys.argv[1]
flt = sys.argv[2] if len(sys.argv) > 2 else ""
spec = importlib.util.spec_from_file_location("tmod", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
names = [n for n in dir(mod) if n.startswith("test_") and flt in n]
names.sort(key=lambda n: mod.__dict__[n].__code__.co_firstlineno)
n_fail = 0
for n in names:
    t0 = time.perf_counter()
    try:
        mod.__dict__[n]()
        print(f"{path}::{n} PASSED ({time.perf_counter() - t0:.1f} s)", flush=True)
    except Exception:
        n_fail += 1
        traceback.print_exc()
        print(f"{path}::{n} FAILED ({time.perf_counter() - t0:.1f} s)", flush=True)
print(f"== {len(names) - n_fail} passed, {n_fail} failed ==")
sys.exit(1 if n_fail else 0)
