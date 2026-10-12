# core-n4.py
# Fast Δn_cut(T) via sparse sampling + fsolve, then velocity field.

import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.Solver import _set_couplings
from RMFsolver.phase_velocity import Get_Two_as_dens, vNtoQ_nc

# ---------------- threading ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
NPROCS = int(os.getenv("NPROCS", cpu_count()))
VERSION = os.getenv("RUN_TAG", "v4")

def _init_worker():
    _set_couplings(para.paraQMCRMF3)

_set_couplings(para.paraQMCRMF3)

# ---------------- physics & grids ----------------
n0 = 0.16 * const.MeV_fm**3
ncrit = 4.0 * n0
m_s = 0

TT = np.logspace(np.log10(0.01), np.log10(100.0), num=80)
NT = TT.size

ND = 30
EPS = 0.002 * n0

# ---------------- fast boundary solve ----------------
def solve_Deln_at_T(T, ncrit, N_samp=7):
    # Only sample in the physically relevant high-Δn band
    del_list = np.linspace(0.1 * n0, 0.6 * n0, N_samp)
    diff_list = []
    for d in del_list:
        aQ, aN = Get_Two_as_dens(T, ncrit, d, m_s, para.paraQMCRMF3, "PNM")
        diff_list.append(aN - aQ)
    diff_list = np.array(diff_list)

    # If always positive → entire band is safe → boundary is simply max
    if np.all(diff_list > 0):
        return 0.6 * n0

    # Interpolate and solve
    f = interp1d(del_list, diff_list, kind="linear", fill_value="extrapolate")
    idx_near = np.argmin(np.abs(diff_list))
    guess = float(del_list[idx_near])
    sol = float(fsolve(f, guess)[0])
    return float(np.clip(sol, 0.01 * n0, 0.6 * n0))

def _solve_wrap(args):
    i, T = args
    return i, solve_Deln_at_T(T, ncrit)

Deln_cut = np.zeros(NT)
with Pool(processes=NPROCS, initializer=_init_worker) as pool:
    for i, val in tqdm(pool.imap_unordered(_solve_wrap, enumerate(TT)),
                       total=NT, desc="Computing 4*n0 Δn_cut(T)", dynamic_ncols=True):
        Deln_cut[i] = val

np.save(f"Shade_n_crit=4_TTpiece__{VERSION}.npy", np.array([Deln_cut, TT]))

# ---------------- velocity field ----------------
def compute_v(T, ncrit, Deln):
    v, _ = vNtoQ_nc(T=T, n_crit=ncrit, Deln=Deln, m_s=m_s,
                    param=para.paraQMCRMF3, NM_type="PNM")
    return v

V = np.zeros((NT, ND))
with Pool(processes=NPROCS, initializer=_init_worker) as pool:
    for i in tqdm(range(NT), desc="Computing 4*n0 velocity", dynamic_ncols=True):
        upper = float(Deln_cut[i] - EPS)
        upper = float(np.clip(upper, 0.01 * n0 + EPS, 0.5 * n0))
        DD_local = np.linspace(0.01 * n0, upper, ND)
        V[i, :] = pool.starmap(compute_v, [(TT[i], ncrit, d) for d in DD_local], chunksize=2)

np.save(f"v_n_crit=4_TTpiece__{VERSION}.npy", V)

