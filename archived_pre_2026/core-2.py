# core-2-par.py
import os
import numpy as np
from tqdm import tqdm
import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.phase_velocity import Get_Two_as_dens, extract_contour_coords_num

# -------------------- constants & grids --------------------
n0 = 0.16 * const.MeV_fm**3
N = 30
nc2, nc3, nc4 = 2.0*n0, 3.0*n0, 4.0*n0
m_strange = 0
TT = np.logspace(np.log10(0.01), np.log10(100), num=N)   # T grid
DD = np.linspace(0.01*n0, 0.5*n0, N)                     # Δn grid


# -------------------- 2 * n0 --------------------
a_diff2 = np.zeros((N, N))
for i in tqdm(range(N), desc="Computing for 2 n0"):  # loop over T (rows)
    for j in range(N):  # loop over B (columns)
        aQ_tmp, aN_tmp = Get_Two_as_dens(
            T = TT[i],
            n_crit = nc2,
            Deln = DD[j],
            m_s = m_strange,
            param = para.paraQMCRMF3,
            NM_type = "PNM",
        )
        a_diff2[i,j] = aN_tmp - aQ_tmp
Deln_cut2, T_cut2 = extract_contour_coords_num(DD, TT, a_diff2.T, level=0)
np.save("Shade_n_crit=2__v2_local.npy", np.array([Deln_cut2, T_cut2]))
np.save("aQstar_aN_n_crit=2__v2_local.npy", a_diff2)


# -------------------- 3 * n0 --------------------
a_diff3 = np.zeros((N, N))
for i in tqdm(range(N), desc="Computing for 3 n0"):  # loop over T (rows)
    for j in range(N):  # loop over B (columns)
        aQ_tmp, aN_tmp = Get_Two_as_dens(
            T = TT[i],
            n_crit = nc3,
            Deln = DD[j],
            m_s = m_strange,
            param = para.paraQMCRMF3,
            NM_type = "PNM",
        )
        a_diff3[i,j] = aN_tmp - aQ_tmp
Deln_cut3, T_cut3 = extract_contour_coords_num(DD, TT, a_diff3.T, level=0)
np.save("Shade_n_crit=3__v2_local.npy", np.array([Deln_cut3, T_cut3]))
np.save("aQstar_aN_n_crit=3__v2_local.npy", a_diff3)


# -------------------- 4 * n0 --------------------
a_diff4 = np.zeros((N, N))
for i in tqdm(range(N), desc="Computing for 4 n0"):  # loop over T (rows)
    for j in range(N):  # loop over B (columns)
        aQ_tmp, aN_tmp = Get_Two_as_dens(
            T = TT[i],
            n_crit = nc4,
            Deln = DD[j],
            m_s = m_strange,
            param = para.paraQMCRMF3,
            NM_type = "PNM",
        )
        a_diff4[i,j] = aN_tmp - aQ_tmp
Deln_cut4, T_cut4 = extract_contour_coords_num(DD, TT, a_diff4.T, level=0)
np.save("Shade_n_crit=4_log.npy", np.array([Deln_cut4, T_cut4]))
np.save("aQstar_aN_n_crit=4_log_field.npy", a_diff4)

