# Project Context Pack

Task: debug 26SU-8 z_time_evolution uNmax failure at T=10 MeV near finite-temperature critical density

Use the mapped files below as the initial context. Verify source before editing.

## `lib/python3.12/site-packages/debugpy/_vendored/pydevd/_pydev_bundle/pydev_log.py`

- Role: PY source defining _LoggingGlobals, _compute_filename_with_pid, _pydevd_log, _pydevd_log_exception, critical, debug, debug_once, error_once.
- Selected because: matches task terms: critical, debug
- Confirmed links: none
- Inferred links: none
- Unresolved references: `_pydevd_bundle.pydevd_constants`

## `lib/python3.12/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/winappdbg/debug.py`

- Role: PY source defining Debug, MixedBitsWarning, __cleanup_process, __enter__, __exit__, __init__, __len__, __setSystemKillOnExitMode.
- Selected because: matches task terms: debug
- Confirmed links: none
- Inferred links: none
- Unresolved references: `winappdbg`, `winappdbg.breakpoint`, `winappdbg.event`, `winappdbg.interactive`, `winappdbg.module`, `winappdbg.process`, `winappdbg.system`, `winappdbg.thread`, `winappdbg.window`

## `lib/python3.12/site-packages/sympy/physics/quantum/density.py`

- Role: PY source defining Density, _eval_args, _eval_trace, _generate_outer_prod, _print_operator_name_latex, _print_operator_name_pretty, _represent, apply_op.
- Selected because: matches task terms: density
- Confirmed links: none
- Inferred links: none
- Unresolved references: `numpy`, `sympy.core.add`, `sympy.core.containers`, `sympy.core.function`, `sympy.core.mul`, `sympy.core.singleton`, `sympy.functions.elementary.exponential`, `sympy.matrices.dense`, `sympy.physics.quantum.dagger`, `sympy.physics.quantum.matrixutils`, `sympy.physics.quantum.operator`, `sympy.physics.quantum.represent`, `sympy.physics.quantum.trace`, `sympy.printing.pretty.stringpict`
