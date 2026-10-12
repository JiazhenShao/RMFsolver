# Project Context Pack

Task: design z_time_evolution in RMFsolver/phase_velocity.py based on backup/phase_velocity.py using uNmax and temperature evolution

Use the mapped files below as the initial context. Verify source before editing.

## `RMFsolver_Pub/RMFsolver/phase_velocity.py`

- Role: PY source defining E_f, Get_Delta_P_max, Get_Delta_n_max, Get_Two_as_dens, Get_Two_as_dens0, Get_Two_as_pres, Get_Two_as_pres0, Get_aN_aQstar_diff.
- Selected because: matches task terms: phase_velocity, py, rmfsolver
- Confirmed links: `RMFsolver/RMFparameter.py`, `RMFsolver/Solver.py`, `RMFsolver/Solver.py`, `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `matplotlib`, `matplotlib.pyplot`, `matplotlib.pyplot`, `numdifftools`, `numpy`, `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`, `tqdm.auto`

## `backup/phase_velocity copy.py`

- Role: PY source defining Get_Delta_P_max, Get_Delta_n_max, Get_Two_as_dens, Get_Two_as_dens0, Get_Two_as_pres, Get_aN_aQstar_diff, PNM, PNM_n.
- Selected because: matches task terms: backup, phase_velocity, py
- Confirmed links: `RMFsolver/RMFparameter.py`, `RMFsolver/Solver.py`, `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `matplotlib`, `matplotlib.pyplot`, `matplotlib.pyplot`, `numdifftools`, `numpy`, `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`

## `backup/phase_velocity.py`

- Role: PY source defining Get_Delta_P_max, Get_Delta_n_max, Get_Two_as_dens, Get_Two_as_pres, Get_aN_aQstar_diff, PNM, PNM_n, PQM.
- Selected because: matches task terms: backup, phase_velocity, py
- Confirmed links: `RMFsolver/RMFparameter.py`, `RMFsolver/Solver.py`, `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `matplotlib`, `matplotlib.pyplot`, `matplotlib.pyplot`, `numdifftools`, `numpy`, `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`

## `RMFsolver/RMFparameter.py`

- Role: PY source. import numpy as np
- Selected because: matches task terms: py, rmfsolver
- Confirmed links: `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `numpy`

## `RMFsolver/Solver.py`

- Role: PY source defining DeltaMU, Eq1, Eq2, Eq3, Eq4, Eq5, Eq6, EqSYM.
- Selected because: matches task terms: py, rmfsolver
- Confirmed links: none
- Inferred links: none
- Unresolved references: `numdifftools`, `numpy`, `numpy.linalg`, `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`, `scipy.optimize`, `scipy.optimize`

## `RMFsolver/constants.py`

- Role: PY source. import math
- Selected because: matches task terms: py, rmfsolver
- Confirmed links: none
- Inferred links: none
- Unresolved references: `numpy`

## `RMFsolver/phase_velocity.py`

- Role: PY source defining PNM, PNM_n, Pi_NM, Pi_QM, _Pi_QM_state, _TQ_from_param, _adiabatic_default_TQ_guesses, _adiabatic_initial_state_failed.
- Selected because: matches task terms: on, phase_velocity, py, rmfsolver
- Confirmed links: `RMFsolver/RMFparameter.py`, `RMFsolver/SQMsolver.py`, `RMFsolver/Solver.py`, `RMFsolver/Solver.py`, `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `numpy`, `scipy.integrate`, `scipy.optimize`

## `RMFsolver_Cluster_backup/phase_velocity.py`

- Role: PY source defining E_f, Get_Delta_P_max, Get_Delta_n_max, Get_Two_as_dens, Get_Two_as_dens0, Get_Two_as_pres, Get_Two_as_pres0, Get_aN_aQstar_diff.
- Selected because: matches task terms: backup, phase_velocity, py, rmfsolver
- Confirmed links: `RMFsolver/RMFparameter.py`, `RMFsolver/Solver.py`, `RMFsolver/Solver.py`, `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `matplotlib`, `matplotlib.pyplot`, `matplotlib.pyplot`, `numdifftools`, `numpy`, `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`, `tqdm.auto`
