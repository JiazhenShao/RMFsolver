## Accuracy of the Isothermal Composition Ceiling: Eq. (35)

The exact neutral-EOS ceiling is the reference on every panel and the horizontal coordinate.
The upper panel compares composition values; the middle panel shows the signed difference; the lower panel shows the relative difference with the exact ceiling in the denominator.

The 292 colored points come from the [independently verified audit](local_analysis_results/26-10-16/aiso-accuracy/results.json), covering successful targets in the original seven-level isothermal checkpoint, with temperatures from 0.01 to 83.86 MeV.
Temperature sets the marker color; the points are not connected or presented as fixed-temperature curves.
The blue dashed line is the exact cold-isobar relation substituted into Eq. (35); its 1% crossing at exact composition 0.495544 was checked with Wolfram.
The black reference is the identity in the upper panel and zero in the difference panels.

There are 113 points above 1% error, with a largest error of 8.48%.
The static expansion uses the equilibrated-endpoint baryon chemical potential, not the interface value.
The production solver already computes the ceiling from the exact EOS; this figure checks the manuscript's explicit approximation, not the accuracy of the numerical front speed.

Run the following cell independently from the repository root; it loads saved results and exports PNG/PDF without new EOS or BVP solves.
Plot settings, fonts, and output paths are at the top of the cell.
