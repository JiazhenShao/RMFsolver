# core-n3.py
# Parallel evaluation of uQ and uN on two 2D parameter slices:
#   (1) (T, jB) at fixed Pi_over_crit
#   (2) (T, Pi_over_crit) at fixed jB
# The outputs are saved as .npz files containing the grids and the corresponding
# uQ/uN arrays, ready for later contour plotting.

import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.Solver import _set_couplings
from RMFsolver.phase_velocity import uQ_uN

# ---------------- threading / multiprocessing ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

NPROCS = int(os.getenv("NPROCS", cpu_count()))
VERSION = os.getenv("RUN_TAG", "v2")

def _init_worker():
    _set_couplings(para.paraQMCRMF3)

# Also initialize once in the parent process
_set_couplings(para.paraQMCRMF3)

# ---------------- physical inputs ----------------
B_one_forth = 180.0
lambda_val = 0.5
n0 = 0.16 * const.MeV_fm**3

# ---------------- grids ----------------
TT = np.linspace(0.01, 100.0, 30)
jB_list = n0 * np.logspace(-8, -3, 28)
Pi_over_crit_list = np.linspace(1.1, 5, 28)

NT = TT.size
NJ = jB_list.size
NP = Pi_over_crit_list.size

# ---------------- fixed parameters for 2D slices ----------------
# For the (T, jB) contour, Pi_over_crit must be fixed.
PI_OVER_CRIT_FIXED_FOR_JB_SCAN = 2

# For the (T, Pi_over_crit) contour, jB must be fixed.
JB_FIXED_FOR_PI_OVER_CRIT_SCAN = n0 * 1.0e-5

# ---------------- worker wrappers ----------------
def _uq_un_at_T_jB(args):
    """
    Return (iT, ijB, uQ, uN) for the (T, jB) scan at fixed Pi_over_crit.
    If evaluation fails, return NaNs so the full run can continue.
    """
    iT, ijB, T, jB, Pi_over_crit = args
    try:
        uQ, uN = uQ_uN(T=T, B_one_forth=B_one_forth, lambda_val=lambda_val, jB=jB, Pi_over_crit=Pi_over_crit)
        return iT, ijB, float(uQ), float(uN)
    except Exception:
        return iT, ijB, np.nan, np.nan


def _uq_un_at_T_Pi_over_crit(args):
    """
    Return (iT, iPi, uQ, uN) for the (T, Pi_over_crit) scan at fixed jB.
    If evaluation fails, return NaNs so the full run can continue.
    """
    iT, iPi, T, jB, Pi_over_crit = args
    try:
        uQ, uN = uQ_uN(T=T, B_one_forth=B_one_forth, lambda_val=lambda_val, jB=jB, Pi_over_crit=Pi_over_crit)
        return iT, iPi, float(uQ), float(uN)
    except Exception:
        return iT, iPi, np.nan, np.nan


# ---------------- build task lists ----------------
tasks_T_jB = [
    (iT, ijB, T, jB, PI_OVER_CRIT_FIXED_FOR_JB_SCAN)
    for iT, T in enumerate(TT)
    for ijB, jB in enumerate(jB_list)
]

tasks_T_Pi_over_crit = [
    (iT, iPi, T, JB_FIXED_FOR_PI_OVER_CRIT_SCAN, Pi_over_crit)
    for iT, T in enumerate(TT)
    for iPi, Pi_over_crit in enumerate(Pi_over_crit_list)
]

# ---------------- allocate output arrays ----------------
uQ_T_jB = np.full((NT, NJ), np.nan, dtype=float)
uN_T_jB = np.full((NT, NJ), np.nan, dtype=float)

uQ_T_Pi_over_crit = np.full((NT, NP), np.nan, dtype=float)
uN_T_Pi_over_crit = np.full((NT, NP), np.nan, dtype=float)

# ---------------- parallel run: (T, jB) scan ----------------
with Pool(processes=NPROCS, initializer=_init_worker) as pool:
    results = pool.imap_unordered(_uq_un_at_T_jB, tasks_T_jB, chunksize=4)
    for iT, ijB, uQ, uN in tqdm(results, total=len(tasks_T_jB),
                                desc="Computing uQ,uN on (T, jB) grid",
                                dynamic_ncols=True):
        uQ_T_jB[iT, ijB] = uQ
        uN_T_jB[iT, ijB] = uN

np.savez(
    f"uQ_uN_contour_T_jB__{VERSION}.npz",
    TT=TT,
    jB_list=jB_list,
    Pi_over_crit_fixed=PI_OVER_CRIT_FIXED_FOR_JB_SCAN,
    B_one_forth=B_one_forth,
    lambda_val=lambda_val,
    uQ=uQ_T_jB,
    uN=uN_T_jB,
)

# ---------------- parallel run: (T, Pi_over_crit) scan ----------------
with Pool(processes=NPROCS, initializer=_init_worker) as pool:
    results = pool.imap_unordered(_uq_un_at_T_Pi_over_crit, tasks_T_Pi_over_crit, chunksize=4)
    for iT, iPi, uQ, uN in tqdm(results, total=len(tasks_T_Pi_over_crit),
                                desc="Computing uQ,uN on (T, Pi_over_crit) grid",
                                dynamic_ncols=True):
        uQ_T_Pi_over_crit[iT, iPi] = uQ
        uN_T_Pi_over_crit[iT, iPi] = uN

np.savez(
    f"uQ_uN_contour_T_Pi_over_crit__{VERSION}.npz",
    TT=TT,
    Pi_over_crit_list=Pi_over_crit_list,
    jB_fixed=JB_FIXED_FOR_PI_OVER_CRIT_SCAN,
    B_one_forth=B_one_forth,
    lambda_val=lambda_val,
    uQ=uQ_T_Pi_over_crit,
    uN=uN_T_Pi_over_crit,
)

print(f"Done. Saved:")
print(f"  uQ_uN_contour_T_jB__{VERSION}.npz")
print(f"  uQ_uN_contour_T_Pi_over_crit__{VERSION}.npz")