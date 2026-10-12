# core-n4-local.py  (local serial)
import os, numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.Solver import _set_couplings
from RMFsolver.phase_velocity import Get_Two_as_dens, vNtoQ_nc

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_set_couplings(para.paraQMCRMF3)

n0     = 0.16 * const.MeV_fm**3
ncrit  = 4.0 * n0
m_s    = 0
VERSION = "v2_local"

TT_log = np.logspace(np.log10(0.01), np.log10(1.0), num=40)
TT_lin = np.linspace(1.0, 100.0, num=21)[1:]
TT     = np.concatenate([TT_log, TT_lin])
NT     = TT.size

ND  = 30
EPS = 0.002 * n0

def solve_Deln_at_T(T, ncrit, N_samp=7):
    del_list  = np.linspace(0.10*n0, 0.50*n0, N_samp)
    diff_list = []
    for d in del_list:
        aQ, aN = Get_Two_as_dens(T, ncrit, d, m_s, para.paraQMCRMF3, "PNM")
        diff_list.append(aN - aQ)
    if np.all(np.array(diff_list) > 0):
        return 0.50*n0
    f = interp1d(del_list, diff_list, kind="linear", fill_value="extrapolate")
    guess    = 0.10*n0
    idx_near = int(np.argmin(np.abs(diff_list)))
    if abs(diff_list[idx_near]) < abs(f(guess)):
        guess = float(del_list[idx_near])
    sol = float(fsolve(f, guess)[0])
    return float(np.clip(sol, 0.01*n0, 0.50*n0))

Deln_cut = np.zeros(NT)
for i in tqdm(range(NT), desc="Computing Δn_cut(T)"):
    Deln_cut[i] = solve_Deln_at_T(TT[i], ncrit)
np.save(f"Shade_n_crit=4_TTpiece__{VERSION}.npy", np.array([Deln_cut, TT]))

def compute_v(T, ncrit, Deln):
    v, _ = vNtoQ_nc(T=T, n_crit=ncrit, Deln=Deln, m_s=m_s,
                    param=para.paraQMCRMF3, NM_type="PNM")
    return v

V = np.zeros((NT, ND))
for i in tqdm(range(NT), desc="Computing velocity field"):
    upper = float(np.clip(Deln_cut[i] - EPS, 0.01*n0 + EPS, 0.50*n0))
    DD    = np.linspace(0.01*n0, upper, ND)
    for j, d in enumerate(DD):
        V[i, j] = compute_v(TT[i], ncrit, d)

np.save(f"v_n_crit=4_TTpiece__{VERSION}.npy", V)