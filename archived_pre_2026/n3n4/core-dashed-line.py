# core-dashed-line.py
# Fast Δn_cut(T) via sparse sampling + fsolve, then velocity field on the safe side.

import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.Solver import _set_couplings
from RMFsolver.phase_velocity import Get_aQstar_etaQ

# ---------------- threading ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
NPROCS = int(os.getenv("NPROCS", cpu_count()))
VERSION = os.getenv("RUN_TAG", "v3")  # simple version tag for saved files

def _init_worker():
    _set_couplings(para.paraQMCRMF3)

# Also init once in the parent
_set_couplings(para.paraQMCRMF3)

# ---------------- physics & grids ----------------
n0 = 0.16 * const.MeV_fm**3
ncrit = 3.0 * n0
m_s = 0

DD = np.logspace(np.log10(0.01*n0), np.log10(0.9*n0), num=30)
ND = DD.size

NT = 160
EPS = 0.005 * n0

# ---------------- fast boundary solve ----------------
def solve_T_at_Deln(Deln, ncrit, N_samp=10):
    # Narrow band first (fast path)
    del_list = np.logspace(-2, 2, N_samp)
    diff_list = []
    for T in del_list:
        aQ, etaQ = Get_aQstar_etaQ(T, ncrit, Deln, m_s, para.paraQMCRMF3, "PNM")
        diff_list.append(aQ*aQ - 2*etaQ)

    # Interpolate and solve; bias the guess toward small Δn (physics prior)
    f = interp1d(del_list, diff_list, kind="linear", fill_value="extrapolate")
    guess = 1.0
    # If sampling found a value closer to zero, use it as a refined guess
    idx_near = int(np.argmin(np.abs(diff_list)))
    guess = float(del_list[idx_near]) if np.abs(diff_list[idx_near]) < np.abs(f(guess)) else guess

    sol = float(fsolve(f, guess)[0])
    return float(np.clip(sol, 0.01, 100.0))

# stream progress with imap_unordered
def _solve_wrap(args):
    i, Deln = args
    return i, solve_T_at_Deln(Deln, ncrit)

T_cut = np.zeros(ND)
with Pool(processes=NPROCS, initializer=_init_worker) as pool:
    for i, val in tqdm(
        pool.imap_unordered(_solve_wrap, enumerate(DD)),
        total=ND, desc="Computing 3*n0 T_cut", dynamic_ncols=True, mininterval=0.5
    ):
        T_cut[i] = val

np.save(f"D_line_n_crit=3__{VERSION}.npy", np.array([DD, T_cut]))

