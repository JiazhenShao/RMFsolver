# Project Context Pack

Task: trace contour plot u(0-) versus temperature and metastability m=0 nBcrit(T=0)

Use the mapped files below as the initial context. Verify source before editing.

## `lib/python3.12/site-packages/sympy/plotting/plot.py`

- Role: PY source defining _build_line_series, _create_series, _plot3d_plot_contour_helper, _process_summations, _set_labels, check_arguments, new_bound, plot.
- Selected because: matches task terms: contour, plot
- Confirmed links: none
- Inferred links: none
- Unresolved references: `sympy.concrete.summations`, `sympy.core.containers`, `sympy.core.expr`, `sympy.core.function`, `sympy.core.symbol`, `sympy.external`, `sympy.functions`, `sympy.plotting.backends.base_backend`, `sympy.plotting.backends.base_backend`, `sympy.plotting.backends.matplotlibbackend`, `sympy.plotting.backends.matplotlibbackend`, `sympy.plotting.backends.textbackend`, `sympy.plotting.plotgrid`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.series`, `sympy.plotting.textplot`, `sympy.plotting.utils`, `sympy.tensor.indexed`, `sympy.vector`

## `lib/python3.12/site-packages/sympy/matrices/expressions/trace.py`

- Role: PY source defining Trace, __new__, _eval_derivative, _eval_derivative_matrix_lines, _eval_rewrite_as_Sum, _eval_transpose, _normalize, arg.
- Selected because: matches task terms: trace
- Confirmed links: `lib/python3.12/site-packages/sympy/matrices/expressions/matexpr.py`
- Inferred links: none
- Unresolved references: `sympy.concrete.summations`, `sympy.concrete.summations`, `sympy.core.basic`, `sympy.core.expr`, `sympy.core.singleton`, `sympy.core.sorting`, `sympy.core.symbol`, `sympy.core.sympify`, `sympy.matrices.exceptions`, `sympy.matrices.expressions.matmul`, `sympy.matrices.expressions.transpose`, `sympy.matrices.matrixbase`, `sympy.tensor.array.expressions.array_expressions`

## `lib/python3.12/site-packages/aiohttp/tracing.py`

- Role: PY source defining Trace, TraceConfig, TraceConnectionCreateEndParams, TraceConnectionCreateStartParams, TraceConnectionQueuedEndParams, TraceConnectionQueuedStartParams, TraceConnectionReuseconnParams, TraceDnsCacheHitParams.
- Selected because: matches task terms: trace
- Confirmed links: `lib/python3.12/site-packages/aiohttp/client.py`, `lib/python3.12/site-packages/aiohttp/client_reqrep.py`
- Inferred links: none
- Unresolved references: `aiosignal`, `attr`, `multidict`, `yarl`

## `lib/python3.12/site-packages/sympy/plotting/__init__.py`

- Role: PY source. from .plot import plot_backends
- Selected because: connected to a task-matching file
- Confirmed links: `lib/python3.12/site-packages/sympy/plotting/plot.py`, `lib/python3.12/site-packages/sympy/plotting/plot.py`, `lib/python3.12/site-packages/sympy/plotting/plot.py`, `lib/python3.12/site-packages/sympy/plotting/plot_implicit.py`, `lib/python3.12/site-packages/sympy/plotting/pygletplot/__init__.py`, `lib/python3.12/site-packages/sympy/plotting/textplot.py`
- Inferred links: none
- Unresolved references: none

## `lib/python3.12/site-packages/sympy/matrices/expressions/matexpr.py`

- Role: PY source defining I, MatrixElement, MatrixExpr, MatrixSymbol, T, _LeftRightArgs, __abs__, __add__.
- Selected because: connected to a task-matching file
- Confirmed links: `lib/python3.12/site-packages/sympy/matrices/expressions/applyfunc.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/determinant.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/inverse.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matadd.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matmul.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matpow.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/special.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/transpose.py`, `lib/python3.12/site-packages/sympy/tensor/array/expressions/array_expressions.py`, `lib/python3.12/site-packages/sympy/tensor/array/expressions/array_expressions.py`
- Inferred links: none
- Unresolved references: `numpy`, `sympy.concrete.summations`, `sympy.core`, `sympy.core.assumptions`, `sympy.core.decorators`, `sympy.core.expr`, `sympy.core.logic`, `sympy.core.symbol`, `sympy.core.sympify`, `sympy.external.gmpy`, `sympy.functions`, `sympy.functions.special.tensor_functions`, `sympy.matrices.exceptions`, `sympy.matrices.expressions.adjoint`, `sympy.matrices.expressions.adjoint`, `sympy.matrices.expressions.determinant`, `sympy.matrices.expressions.slice`, `sympy.matrices.expressions.slice`, `sympy.matrices.expressions.transpose`, `sympy.matrices.immutable`, `sympy.matrices.immutable`, `sympy.matrices.kind`, `sympy.matrices.matrixbase`, `sympy.multipledispatch`, `sympy.simplify`, `sympy.tensor.array.array_derivatives`, `sympy.tensor.array.expressions.arrayexpr_derivatives`, `sympy.tensor.array.expressions.from_array_to_matrix`, `sympy.tensor.array.expressions.from_array_to_matrix`, `sympy.tensor.array.expressions.from_array_to_matrix`, `sympy.tensor.array.expressions.from_indexed_to_array`, `sympy.tensor.array.expressions.from_matrix_to_array`, `sympy.utilities.misc`

## `lib/python3.12/site-packages/sympy/matrices/expressions/__init__.py`

- Role: PY source. """ A module which handles Matrix Expressions """
- Selected because: connected to a task-matching file
- Confirmed links: `lib/python3.12/site-packages/sympy/matrices/expressions/adjoint.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/blockmatrix.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/companion.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/determinant.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/diagonal.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/dotproduct.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/funcmatrix.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/hadamard.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/inverse.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/kronecker.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matadd.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matexpr.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matmul.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matpow.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/permutation.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/sets.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/slice.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/special.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/trace.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/transpose.py`
- Inferred links: none
- Unresolved references: none

## `lib/python3.12/site-packages/sympy/matrices/expressions/kronecker.py`

- Role: PY source defining KroneckerProduct, __new__, _entry, _eval_adjoint, _eval_conjugate, _eval_determinant, _eval_expand_kroneckerproduct, _eval_inverse.
- Selected because: matches task terms: trace
- Confirmed links: `lib/python3.12/site-packages/sympy/matrices/expressions/determinant.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matadd.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matmul.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/matpow.py`, `lib/python3.12/site-packages/sympy/matrices/expressions/trace.py`
- Inferred links: none
- Unresolved references: `sympy.core`, `sympy.functions`, `sympy.matrices.exceptions`, `sympy.matrices.expressions.inverse`, `sympy.matrices.expressions.matexpr`, `sympy.matrices.expressions.special`, `sympy.matrices.expressions.transpose`, `sympy.matrices.matrixbase`, `sympy.strategies`, `sympy.strategies.traverse`, `sympy.utilities`

## `lib/python3.12/site-packages/sympy/matrices/expressions/matadd.py`

- Role: PY source defining MatAdd, __new__, _entry, _eval_adjoint, _eval_derivative_matrix_lines, _eval_trace, _eval_transpose, _evaluate.
- Selected because: matches task terms: trace
- Confirmed links: `lib/python3.12/site-packages/sympy/matrices/expressions/trace.py`
- Inferred links: none
- Unresolved references: `sympy.core`, `sympy.core.add`, `sympy.core.sorting`, `sympy.functions`, `sympy.matrices.expressions._shape`, `sympy.matrices.expressions.matexpr`, `sympy.matrices.expressions.special`, `sympy.matrices.expressions.transpose`, `sympy.matrices.matrixbase`, `sympy.strategies`, `sympy.utilities.exceptions`, `sympy.utilities.iterables`
