# Project Context Pack

Task: diagnose why solve_front_energy_conserving_uNmax velocity does not approach zero at zero metastability

Use the mapped files below as the initial context. Verify source before editing.

## `lib/python3.12/site-packages/bs4/diagnose.py`

- Role: PY source defining AnnouncingParser, _p, benchmark_parsers, diagnose, handle_charref, handle_comment, handle_data, handle_decl.
- Selected because: matches task terms: diagnose
- Confirmed links: none
- Inferred links: none
- Unresolved references: `bs4`, `bs4`, `bs4._typing`, `bs4.builder`, `html5lib`, `html5lib`, `lxml`, `lxml`, `lxml`

## `lib/python3.12/site-packages/pandas/tests/arithmetic/conftest.py`

- Role: Test code defining not_daily, one, scalar_td, three_days, two_hours, zero.
- Selected because: matches task terms: not, zero
- Confirmed links: none
- Inferred links: lib/python3.12/site-packages/IPython/conftest.py (test basename matches source basename 'conftest'), lib/python3.12/site-packages/matplotlib/testing/conftest.py (test basename matches source basename 'conftest'), lib/python3.12/site-packages/nbconvert/conftest.py (test basename matches source basename 'conftest'), lib/python3.12/site-packages/numpy/conftest.py (test basename matches source basename 'conftest'), lib/python3.12/site-packages/pandas/conftest.py (test basename matches source basename 'conftest'), lib/python3.12/site-packages/scipy/conftest.py (test basename matches source basename 'conftest'), lib/python3.12/site-packages/sympy/conftest.py (test basename matches source basename 'conftest')
- Unresolved references: `numpy`, `pandas`, `pandas`, `pytest`

## `RMFsolver_Cluster_backup/phase_velocity_old.py`

- Role: PY source defining Get_Delta_P_max, Get_Delta_n_max, Get_aN_aQstar_diff, PNM, PNM_n, PQM, PQM_em, P_f.
- Selected because: matches task terms: at, velocity, zero
- Confirmed links: `RMFsolver/RMFparameter.py`, `RMFsolver/Solver.py`, `RMFsolver/constants.py`
- Inferred links: none
- Unresolved references: `matplotlib.pyplot`, `numdifftools`, `numpy`, `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`

## `lib/python3.12/site-packages/IPython/conftest.py`

- Role: PY source defining get_ipython, inject, nopage, pytest_collection_modifyitems, work_path, xsys.
- Selected because: connected to a task-matching file
- Confirmed links: `lib/python3.12/site-packages/IPython/core/__init__.py`, `lib/python3.12/site-packages/IPython/terminal/interactiveshell.py`, `lib/python3.12/site-packages/IPython/testing/__init__.py`
- Inferred links: none
- Unresolved references: `pytest`

## `lib/python3.12/site-packages/matplotlib/testing/conftest.py`

- Role: PY source defining mpl_test_settings, pd, pytest_configure, pytest_unconfigure, xr.
- Selected because: connected to a task-matching file
- Confirmed links: none
- Inferred links: none
- Unresolved references: `matplotlib`, `matplotlib`, `matplotlib.pyplot`, `matplotlib.testing.decorators`, `pandas.plotting`, `pytest`

## `lib/python3.12/site-packages/nbconvert/conftest.py`

- Role: PY source. """pytest configuration."""
- Selected because: connected to a task-matching file
- Confirmed links: none
- Inferred links: none
- Unresolved references: none
