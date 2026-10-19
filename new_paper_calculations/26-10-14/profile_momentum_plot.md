## Nearly Constant Baryon Chemical Potential: Original Profile Figure

The original plotting code is in `26-04-09.ipynb`, cell 9 (zero based), and its stored output matches the requested figure.
The upstream calculation cell was later changed to a different run, so executing that old notebook from the top does not reproduce the stored figure.
The cell below redraws the original contour vertices, four sampled profile points, and endpoints recovered from the saved **vector PDF**, `jargon/entropy_tqguess_uagrad_pi_qm_contour.pdf`.
This is a faithful replot of a historical EOS/solver result, not a new BVP calculation or a validation with the current EOS.

All endpoint labels use $x=0^-$, $x=0^+$, and $x=\infty$.
The fixed quantity is the momentum flux $\Pi=P+hu^2$; the colored curves are different temperature slices at the same flux.
The nearly vertical numerical path supports neglecting the change in $\mu_B$ **for this example**: the plotted endpoint change is about $0.073\,\mathrm{MeV}$, or $0.0064\%$.
This percentage is reconstructed at vector-figure precision and is not a bound for every front in the parameter scan.
All visual controls are in the tuning block, and the cell runs from the saved JSON without importing a solver.
