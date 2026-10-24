"""Compatibility shim: the uncertainty API lives in utils/pw/uncertainty.py (PLAN A2G-P1 core §2)."""
from .uncertainty import (ALPHA, CHI2_2, CHI2_3, OffsetEstimator, PredNoise, SensorNoise, ang_nis, cell_rows,  # noqa: F401
                          chi2, jac_a, jac_b, jac_xi, nis, residual, residual_cov, resolvable, segment_cov,
                          shift_matrix)
