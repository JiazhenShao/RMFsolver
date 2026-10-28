"""Instant synthetic evaluators for the isothermal contour engine tests."""
import time

import numpy as np


def evaluate(method, x, T):
    speed = 50.0 * np.exp(x) * (1 + T / 100.0)
    record = dict(method=method, x=float(x), T=float(T), status="accepted", accepted=True,
                  speed_m_s=float(speed), message="", attempts=1, a_0plus=0.5, runtime_s=0.0,
                  I2=1e-12, lambda_n=0.6)
    profile = dict(x=np.linspace(0, 1, 256), a=np.linspace(0.5, 0, 256)) if method == "numerical" else None
    return record, profile


def slow(method, x, T):
    if abs(x - 2.0) < 1e-12:
        time.sleep(30)
    return evaluate(method, x, T)


def crash(method, x, T):
    if abs(x - 2.0) < 1e-12:
        import os
        os._exit(1)   # a worker dying in C code (segfault, OOM kill)
    return evaluate(method, x, T)
