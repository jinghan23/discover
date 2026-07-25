# Estimator variance across fresh MLPs

Scores below use corrected FLOPs-only adjusted score. Lower is better.

| Method | Fresh rank | Mean ×1e7 | Variance ×1e14 | SD ×1e7 | CV | q05–q95 ×1e7 | 95% mean half-width ×1e7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| #317723 AutoEvolve shifted Box-Muller lattice with strengthened early-layer moment corrections, 12% | 10 | 4.0178 | 10.6141 | 3.2579 | 0.811 | 1.212–10.171 | ±0.646 |
| #317463 AutoEvolve f64 antithetic whitening with exact first- and second-layer ReLU-kernel moment matching, 14.3% | 1 | 3.3755 | 6.4723 | 2.5441 | 0.754 | 0.848–8.011 | ±0.505 |
| #317465 AutoEvolve Gaussian whitening with first-layer marginal quantiles and second-layer ReLU-kernel moment matching, 22% | 6 | 3.7396 | 10.2489 | 3.2014 | 0.856 | 0.934–8.964 | ±0.635 |
| #317448 AutoEvolve LHS Gaussian with exact first-layer marginal quantile matching, 28% | 7 | 3.7455 | 11.0585 | 3.3254 | 0.888 | 0.935–9.166 | ±0.660 |
| #317469 AutoEvolve Korobov spherical RQMC with angular whitening, radial RB, and first-layer calibration, 13% | 15 | 4.3052 | 24.7536 | 4.9753 | 1.156 | 1.015–8.864 | ±0.987 |
| #317473 AutoEvolve shifted Box-Muller lattice with first-two-layer analytic moment corrections, 10.7% | 2 | 3.4171 | 6.3605 | 2.5220 | 0.738 | 1.119–8.025 | ±0.500 |
| #317475 AutoEvolve cached two-shift spherical RQMC with gate-propagated first-layer control, 10.24% | 3 | 3.5009 | 7.0859 | 2.6619 | 0.760 | 0.930–8.031 | ±0.528 |
| #317457 AutoEvolve QR-orthogonal antithetic probes with early-layer diagonal moment matching, 9% | 8 | 3.7584 | 6.9653 | 2.6392 | 0.702 | 1.098–9.090 | ±0.524 |
| #317426 TTT spherical RQMC with analytic first-layer second-moment matching, 10% | 9 | 3.8492 | 9.4455 | 3.0734 | 0.798 | 0.997–10.569 | ±0.610 |
| #317417 AutoEvolve LHS-normal antithetic whitening, 22% | 5 | 3.6800 | 11.1102 | 3.3332 | 0.906 | 0.933–9.531 | ±0.661 |
| #317453 TTT orthogonal antithetic sphere with propagated first-layer control and final Gaussian smoothing, 18% | 4 | 3.5576 | 9.6473 | 3.1060 | 0.873 | 0.951–8.887 | ±0.616 |
| #317433 AutoEvolve antithetic Box-Muller lattice Gaussian with empirical whitening, 13% | 13 | 4.1499 | 9.7884 | 3.1286 | 0.754 | 1.086–9.136 | ±0.621 |
| #316625 Whitened-antithetic spherical RQMC + radial RB, 10% | 14 | 4.1915 | 12.5884 | 3.5480 | 0.846 | 1.137–13.208 | ±0.704 |
| #316299 AutoEvolve Gaussian CV 25d94716 | 12 | 4.0985 | 16.5479 | 4.0679 | 0.993 | 0.961–8.910 | ±0.807 |
| #316298 AutoEvolve Edgeworth c02503e1 | 11 | 4.0559 | 16.4182 | 4.0519 | 0.999 | 0.850–9.283 | ±0.804 |

## Paired candidate-versus-incumbent thresholds

Each entry summarizes all method pairs. Values are score differences ×1e7.

| MLPs | One-sided 95% median / q90 | Two-sided 95% median / q90 | 15-candidate Bonferroni median / q90 |
|---:|---:|---:|---:|
| 50 | 0.908 / 1.191 | 1.088 / 1.427 | 1.534 / 2.013 |
| 100 | 0.636 / 0.834 | 0.759 / 0.997 | 1.061 / 1.392 |
| 200 | 0.447 / 0.587 | 0.534 / 0.700 | 0.742 / 0.974 |
| 400 | 0.316 / 0.414 | 0.376 / 0.494 | 0.522 / 0.685 |

Promotion rule: for paired `candidate - incumbent` scores, promote only if `mean_delta + threshold < 0`.
