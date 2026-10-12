import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from numdifftools import Gradient, Derivative
import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.phase_velocity import vNtoQ_nc


n0 = 0.16 * const.MeV_fm**3
N = 30
nc2, nc3, nc4 = 2.0*n0, 3.0*n0, 4.0*n0
m_strange = 0
TT = np.logspace(np.log10(0.01), np.log10(100), num=N)


# -------------------- 2 * n0 --------------------
Deln_cut2, T_cut2 = np.load("Shade_n_crit=2_log.npy")
Cut2 = interp1d(T_cut2, Deln_cut2, kind='linear', fill_value="extrapolate")

V2 = np.zeros([N, N])

for i in tqdm(range(N), desc="Computing for 2 n0"):
    DD2 = np.linspace(0.01*n0, np.minimum(0.5*n0, Cut2(TT[i]) - 0.002*n0), N)
    for j in range(N):
        V2[i,j], _ = vNtoQ_nc(
            T=TT[i],
            n_crit=nc2,
            Deln=DD2[j],
            m_s=m_strange,
            param=para.paraQMCRMF3,
            NM_type="PNM",
        )
np.save(f"v_n_crit={nc2/n0:.0f}.npy", V2)


# -------------------- 3 * n0 --------------------
Deln_cut3, T_cut3 = np.load("Shade_n_crit=3_log.npy")
Cut3 = interp1d(T_cut3, Deln_cut3, kind='linear', fill_value="extrapolate")

V3 = np.zeros([N, N])

for i in tqdm(range(N), desc="Computing for 3 n0"):
    DD3 = np.linspace(0.01*n0, np.minimum(0.5*n0, Cut3(TT[i]) - 0.002*n0), N)
    for j in range(N):
        V3[i,j], _ = vNtoQ_nc(
            T=TT[i],
            n_crit=nc3,
            Deln=DD3[j],
            m_s=m_strange,
            param=para.paraQMCRMF3,
            NM_type="PNM",
        )
np.save(f"v_n_crit={nc3/n0:.0f}.npy", V3)


# -------------------- 4 * n0 --------------------
Deln_cut4, T_cut4 = np.load("Shade_n_crit=4_log.npy")
Cut4 = interp1d(T_cut4, Deln_cut4, kind='linear', fill_value="extrapolate")

V4 = np.zeros([N, N])

for i in tqdm(range(N), desc="Computing for 4 n0"):
    DD4 = np.linspace(0.01*n0, np.minimum(0.5*n0, Cut4(TT[i]) - 0.002*n0), N)
    for j in range(N):
        V4[i,j], _ = vNtoQ_nc(
            T=TT[i],
            n_crit=nc4,
            Deln=DD4[j],
            m_s=m_strange,
            param=para.paraQMCRMF3,
            NM_type="PNM",
        )
np.save(f"v_n_crit={nc4/n0:.0f}.npy", V4)


