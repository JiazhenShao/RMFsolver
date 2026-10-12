# Project Context Pack

Task: edit both 26-07-17 cluster contour runners to compute finite-temperature critical density with fixed zero-temperature-calibrated bag constant and define nB_N as nBcrit(T)+metastability*n0

Use the mapped files below as the initial context. Verify source before editing.

## `lib/python3.12/site-packages/sympy/stats/frv.py`

- Role: PY source defining ConditionalFiniteDomain, FiniteDensity, FiniteDomain, FinitePSpace, ProductFiniteDomain, ProductFinitePSpace, SingleFiniteDistribution, SingleFiniteDomain.
- Selected because: matches task terms: as, compute, density, finite
- Confirmed links: none
- Inferred links: none
- Unresolved references: `sympy.concrete.summations`, `sympy.core.basic`, `sympy.core.cache`, `sympy.core.containers`, `sympy.core.function`, `sympy.core.logic`, `sympy.core.mul`, `sympy.core.numbers`, `sympy.core.relational`, `sympy.core.relational`, `sympy.core.singleton`, `sympy.core.symbol`, `sympy.core.sympify`, `sympy.core.sympify`, `sympy.functions.elementary.exponential`, `sympy.functions.elementary.piecewise`, `sympy.logic.boolalg`, `sympy.sets.sets`, `sympy.sets.sets`, `sympy.stats.rv`

## `lib/python3.12/site-packages/debugpy/_vendored/pydevd/_pydev_bundle/pydev_log.py`

- Role: PY source defining _LoggingGlobals, _compute_filename_with_pid, _pydevd_log, _pydevd_log_exception, critical, debug, debug_once, error_once.
- Selected because: matches task terms: compute, critical, with
- Confirmed links: none
- Inferred links: none
- Unresolved references: `_pydevd_bundle.pydevd_constants`

## `lib/python3.12/site-packages/sympy/physics/quantum/density.py`

- Role: PY source defining Density, _eval_args, _eval_trace, _generate_outer_prod, _print_operator_name_latex, _print_operator_name_pretty, _represent, apply_op.
- Selected because: matches task terms: density
- Confirmed links: none
- Inferred links: none
- Unresolved references: `numpy`, `sympy.core.add`, `sympy.core.containers`, `sympy.core.function`, `sympy.core.mul`, `sympy.core.singleton`, `sympy.functions.elementary.exponential`, `sympy.matrices.dense`, `sympy.physics.quantum.dagger`, `sympy.physics.quantum.matrixutils`, `sympy.physics.quantum.operator`, `sympy.physics.quantum.represent`, `sympy.physics.quantum.trace`, `sympy.printing.pretty.stringpict`
