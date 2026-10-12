import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from numdifftools import Gradient, Derivative
import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.phase_velocity import vNtoQ_nc

# NEW: minimal multiprocessing import so we can parallelize the inner loop
from multiprocessing import Pool, cpu_count  # NEW: Pool for parallel map; cpu_count to size it
# NEW: import the coupling setter used inside RMF to ensure identical state in each worker
from RMFsolver.Solver import _set_couplings    # NEW: per-process RMF initialization
# NEW: let user cap workers via env; fall back to all cores
import os                                     # NEW: read NPROCS and set BLAS threads (optional but harmless)
NPROCS = int(os.getenv("NPROCS", cpu_count()))  # NEW: number of worker processes

# (optional but recommended to avoid BLAS oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")      # NEW: keep one BLAS thread per worker
os.environ.setdefault("MKL_NUM_THREADS", "1")      # NEW
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") # NEW
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")  # NEW

# NEW: tiny initializer; runs once in each worker so RMF's module-level globals are set
def _init_worker():
    _set_couplings(para.paraQMCRMF3)  # NEW: guarantees parallel == serial physics

# NEW: tiny worker function; computes one grid point and returns only v
def _compute_v(T, n_crit, Deln):
    v, _ = vNtoQ_nc(T=T, n_crit=n_crit, Deln=Deln, m_s=0,
                    param=para.paraQMCRMF3, NM_type="PNM")
    return v  # NEW: starmap needs a function that returns a scalar per task


n0 = 0.16 * const.MeV_fm**3

# === Grids ===
# NEW: piecewise T sampling (y-axis): 0.01–10 (log, 20pts) + 10–100 (lin, 19pts) -> no duplicate 10
TT_log = np.logspace(np.log10(0.01), np.log10(10.0), num=20)   # NEW
TT_lin = np.linspace(10.0, 100.0, num=20)[1:]                  # NEW (drop first 10.0)
TT = np.concatenate([TT_log, TT_lin])                          # NEW
NT = TT.size                                                   # NEW: number of T samples (rows)

# Keep Δn sampling the same (x-axis)
M = 30                                                         # NEW: columns per row (Δn samples)
nc2, nc3, nc4 = 2.0*n0, 3.0*n0, 4.0*n0
m_strange = 0

# -------------------- 2 * n0 --------------------
# NEW: use the TTpiece shade file (matches your new sampling)
Deln_cut2, T_cut2 = np.load("Shade_n_crit=2_TTpiece.npy", allow_pickle=True)  # NEW
Cut2 = interp1d(T_cut2, Deln_cut2, kind='linear', fill_value="extrapolate")

V2 = np.zeros((NT, M))                                          # NEW: shape by NT × M

# NEW: create a Pool once for this block; initializer sets RMF state in each process
with Pool(processes=NPROCS, initializer=_init_worker) as pool:  # NEW
    for i in tqdm(range(NT), desc="Computing for 2 n0"):
        DD2 = np.linspace(0.01*n0, np.minimum(0.5*n0, Cut2(TT[i]) - 0.002*n0), M)  # NEW: M cols
        # NEW: build tasks for all columns at this row (same arguments you looped over)
        tasks = [(TT[i], nc2, d) for d in DD2]                 # NEW
        # NEW: parallel map of the inner loop; chunksize cuts scheduler overhead a bit
        V2[i, :] = pool.starmap(_compute_v, tasks, chunksize=2)  # NEW

np.save(f"v_n_crit={nc2/n0:.0f}_TTpiece.npy", V2)               # NEW: distinct filename


# -------------------- 3 * n0 --------------------
Deln_cut3, T_cut3 = np.load("Shade_n_crit=3_TTpiece.npy", allow_pickle=True)  # NEW
Cut3 = interp1d(T_cut3, Deln_cut3, kind='linear', fill_value="extrapolate")

V3 = np.zeros((NT, M))                                          # NEW

with Pool(processes=NPROCS, initializer=_init_worker) as pool:  # NEW
    for i in tqdm(range(NT), desc="Computing for 3 n0"):
        DD3 = np.linspace(0.01*n0, np.minimum(0.5*n0, Cut3(TT[i]) - 0.002*n0), M)  # NEW
        tasks = [(TT[i], nc3, d) for d in DD3]                 # NEW
        V3[i, :] = pool.starmap(_compute_v, tasks, chunksize=2)  # NEW

np.save(f"v_n_crit={nc3/n0:.0f}_TTpiece.npy", V3)               # NEW


# -------------------- 4 * n0 --------------------
Deln_cut4, T_cut4 = np.load("Shade_n_crit=4_TTpiece.npy", allow_pickle=True)  # NEW
Cut4 = interp1d(T_cut4, Deln_cut4, kind='linear', fill_value="extrapolate")

V4 = np.zeros((NT, M))                                          # NEW

with Pool(processes=NPROCS, initializer=_init_worker) as pool:  # NEW
    for i in tqdm(range(NT), desc="Computing for 4 n0"):
        DD4 = np.linspace(0.01*n0, np.minimum(0.5*n0, Cut4(TT[i]) - 0.002*n0), M)  # NEW
        tasks = [(TT[i], nc4, d) for d in DD4]                 # NEW
        V4[i, :] = pool.starmap(_compute_v, tasks, chunksize=2)  # NEW

np.save(f"v_n_crit={nc4/n0:.0f}_TTpiece.npy", V4)               # NEW

