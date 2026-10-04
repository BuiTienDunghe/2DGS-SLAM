"""Deformation-graph loop-closure correction for 2DGS-SLAM (handoff plan v1).

Pure-torch modules shared by the offline tools (P1-P3) and the online backend hook (P4).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "gaussian_splatting")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
