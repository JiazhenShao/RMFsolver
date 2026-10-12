# core-2-par.py
import os
import numpy as np
from tqdm import tqdm
import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.phase_velocity import Get_Two_as_dens0, extract_contour_coords_num

from multiprocessing import Pool, cpu_count
from RMFsolver.Solver import _set_couplings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

NPROCS = int(os.getenv("NPROCS", cpu_count()))

# -------------------- constants & grids --------------------
n0 = 0.16 * const.MeV_fm**3

# NEW: piecewise TT sampling — 0.01→10 (log, 20 pts), 10→100 (linear, 20 pts)
#      We drop the first element of the linear segment to avoid a duplicate "10".
TT_log = np.logspace(np.log10(0.01), np.log10(10.0), num=20)           # NEW
TT_lin = np.linspace(10.0, 100.0, num=20)[1:]                           # NEW
TT = np.concatenate([TT_log, TT_lin])                                   # NEW
NT = TT.size                                                            # NEW: y-axis length

# KEEP: Δn grid (x-axis)
N = 30
DD = np.linspace(0.01*n0, 0.5*n0, N)
ND = DD.size                                                            # NEW: x-axis length

nc2, nc3, nc4 = 2.0*n0, 3.0*n0, 4.0*n0
m_strange = 0

def _init_worker():
    _set_couplings(para.paraQMCRMF3)

def _compute_adiff(T, n_crit, Deln):
    aQ, aN = Get_Two_as_dens0(
        T=T,
        n_crit=n_crit,
        Deln=Deln,
        m_s=m_strange,
        param=para.paraQMCRMF3,
        NM_type="PNM",
    )
    return aN - aQ

# -------------------- 2 * n0 --------------------
a_diff2 = np.zeros((NT, ND))                                           # NEW: use NT, ND
with Pool(processes=NPROCS, initializer=_init_worker) as pool:
    for i in tqdm(range(NT), desc="Computing for 2 n0"):
        tasks = [(TT[i], nc2, d) for d in DD]
        row = pool.starmap(_compute_adiff, tasks, chunksize=2)
        a_diff2[i, :] = row
Deln_cut2, T_cut2 = extract_contour_coords_num(DD, TT, a_diff2, level=0) # NOTE: no transpose
np.save("Shade_n_crit=2_TTpiece.npy", np.array([Deln_cut2, T_cut2]))     # NEW: new filename
np.save("aQstar_aN_n_crit=2_TTpiece_field.npy", a_diff2)                 # NEW: new filename

# -------------------- 3 * n0 --------------------
a_diff3 = np.zeros((NT, ND))                                           # NEW
with Pool(processes=NPROCS, initializer=_init_worker) as pool:
    for i in tqdm(range(NT), desc="Computing for 3 n0"):
        tasks = [(TT[i], nc3, d) for d in DD]
        row = pool.starmap(_compute_adiff, tasks, chunksize=2)
        a_diff3[i, :] = row
Deln_cut3, T_cut3 = extract_contour_coords_num(DD, TT, a_diff3, level=0) # NEW: no transpose
np.save("Shade_n_crit=3_TTpiece.npy", np.array([Deln_cut3, T_cut3]))     # NEW
np.save("aQstar_aN_n_crit=3_TTpiece_field.npy", a_diff3)                 # NEW

# -------------------- 4 * n0 --------------------
a_diff4 = np.zeros((NT, ND))                                           # NEW
with Pool(processes=NPROCS, initializer=_init_worker) as pool:
    for i in tqdm(range(NT), desc="Computing for 4 n0"):
        tasks = [(TT[i], nc4, d) for d in DD]
        row = pool.starmap(_compute_adiff, tasks, chunksize=2)
        a_diff4[i, :] = row
Deln_cut4, T_cut4 = extract_contour_coords_num(DD, TT, a_diff4, level=0) # NEW: no transpose
np.save("Shade_n_crit=4_TTpiece.npy", np.array([Deln_cut4, T_cut4]))     # NEW
np.save("aQstar_aN_n_crit=4_TTpiece_field.npy", a_diff4)                 # NEW

