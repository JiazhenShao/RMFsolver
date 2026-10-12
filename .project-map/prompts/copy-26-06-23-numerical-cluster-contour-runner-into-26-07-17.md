# Project Context Pack

Task: copy 26-06-23 numerical cluster contour runner into 26-07-17 as a distinct linear-temperature finite-T-critical-density runner and bump the two existing 26-07-17 run tags to v2

Use the mapped files below as the initial context. Verify source before editing.

## `backup/phase_velocity copy.py`

- Role: PY source defining Get_Delta_P_max, Get_Delta_n_max, Get_Two_as_dens, Get_Two_as_dens0, Get_Two_as_pres, Get_aN_aQstar_diff, PNM, PNM_n.
- Selected because: matches task terms: as, contour, copy, two
- Confirmed links: `RMFsolver/RMFparameter.py`, `RMFsolver/Solver.py`, `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `matplotlib`, `matplotlib.pyplot`, `matplotlib.pyplot`, `numdifftools`, `numpy`, `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`

## `lib/python3.12/site-packages/pexpect/run.py`

- Role: PY source defining run, runu.
- Selected because: matches task terms: run
- Confirmed links: `lib/python3.12/site-packages/pexpect/exceptions.py`, `lib/python3.12/site-packages/pexpect/pty_spawn.py`
- Inferred links: none
- Unresolved references: none

## `lib/python3.12/site-packages/setuptools/_vendor/wheel/cli/tags.py`

- Role: PY source defining _compute_tags, tags.
- Selected because: matches task terms: tags
- Confirmed links: `lib/python3.12/site-packages/setuptools/_vendor/wheel/wheelfile.py`
- Inferred links: none
- Unresolved references: none

## `RMFsolver/RMFparameter.py`

- Role: PY source. import numpy as np
- Selected because: connected to a task-matching file
- Confirmed links: `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `numpy`

## `RMFsolver/Solver.py`

- Role: PY source defining DeltaMU, Eq1, Eq2, Eq3, Eq4, Eq5, Eq6, EqSYM.
- Selected because: matches task terms: density
- Confirmed links: none
- Inferred links: none
- Unresolved references: `numdifftools`, `numpy`, `numpy.linalg`, `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`, `scipy.optimize`, `scipy.optimize`

## `RMFsolver/constants.py`

- Role: PY source. import math
- Selected because: connected to a task-matching file
- Confirmed links: none
- Inferred links: none
- Unresolved references: `numpy`

## `lib/python3.12/site-packages/pexpect/exceptions.py`

- Role: PY source defining EOF, ExceptionPexpect, TIMEOUT, __init__, __str__, get_trace.
- Selected because: connected to a task-matching file
- Confirmed links: none
- Inferred links: none
- Unresolved references: none

## `lib/python3.12/site-packages/pexpect/pty_spawn.py`

- Role: PY source defining __init__, __interact_copy, __interact_read, __interact_writen, __str__, _log_control, _spawn, _spawnpty.
- Selected because: matches task terms: copy
- Confirmed links: `lib/python3.12/site-packages/pexpect/exceptions.py`, `lib/python3.12/site-packages/pexpect/spawnbase.py`, `lib/python3.12/site-packages/pexpect/utils.py`
- Inferred links: none
- Unresolved references: `ptyprocess`, `ptyprocess.ptyprocess`
