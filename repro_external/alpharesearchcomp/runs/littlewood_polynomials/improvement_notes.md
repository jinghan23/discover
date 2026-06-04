# Littlewood Polynomials Second-Round Notes

Date: 2026-06-03

Outcome: no candidate with score greater than 0.03125 was found, so no
`work/improved_program.py` was written.

Baseline:

- Length: 512
- Coefficients: Rudin-Shapiro signs from `work/initial_program.py`
- Evaluator grid: 16384-point FFT
- Baseline supnorm: 32.0
- Baseline score: 0.03125

Second-round attempts focused on exploiting the discrete FFT grid rather than
generic random search:

1. Peak inspection of the Rudin-Shapiro sequence

   The only exact maximum is at DC:

   - `argmax = 0`
   - `sum(coeffs) = 32`
   - `supnorm = 32.0`

   However, the FFT grid has many near-peaks. Examples from the baseline:

   - 7 grid points at or above 31.99
   - 101 grid points at or above 31.9
   - 463 grid points at or above 31.5

   This means suppressing only the DC peak is insufficient; small edits tend to
   raise nearby non-DC peaks above 32.

2. Exhaustive single-flip and double-flip checks from Rudin-Shapiro

   Single flips all made the full-grid supnorm worse. The best single flip seen
   had:

   - flip index: 190
   - resulting supnorm: about 33.621888

   All pairs of coefficient flips were then exhaustively evaluated. The best
   double flip seen had:

   - flip indices: 318 and 395
   - resulting supnorm: about 33.942718

   This ruled out simple one- or two-coordinate peak repair.

3. Rudin-Shapiro/Golay recursive sign variants

   I enumerated the 1024 variants obtained by changing the signs in the
   length-doubling Rudin-Shapiro/Golay recursion and taking either companion
   polynomial. These preserve the same general complementary structure, but on
   the 16384-point evaluator grid every variant still had minimum supnorm 32.0.
   No strict improvement was found.

4. Grid-specific projection/clipping attempt

   I tried clipping large FFT-grid magnitudes and projecting the inverse FFT
   back to the first 512 signs. This approach immediately left the
   Rudin-Shapiro basin and produced sequences with much larger full-grid
   supnorms, so it did not yield a usable candidate.

5. Peak-aware constrained search attempts

   I tried DC-aware flip sets and a peak-aware simulated annealing variant that
   preserved `sum(coeffs) = 30`, so DC would no longer be the maximum. These
   candidates still produced new non-DC peaks well above 32, often in the
   37-40 range after larger flip sets.

   I also tried a cutting-plane-style MILP approximation over selected peak
   frequencies, replacing complex magnitude constraints by directional linear
   projections. The first MILP candidates successfully reduced the constrained
   baseline peaks, but full 16384-point FFT evaluation exposed new unconstrained
   peaks. Example candidates had supnorms around 37-39. Adding the newly
   violated frequencies made the MILP slower without producing a candidate near
   32.

6. Quadratic Boolean phase variants

   Since Rudin-Shapiro can be viewed as a quadratic Boolean phase on the binary
   expansion of the index, I sampled and checked quadratic forms of the shape
   `(-1)^(x^T A x + b^T x)`. The baseline graph and sampled variants did not
   produce a strict full-grid improvement; the best observed value remained
   32.0.

Reason no improved program was written:

The only candidates that reduced the original DC/near-peak set generated new
FFT-grid peaks above 32 elsewhere. The Rudin-Shapiro structure appears to be a
local barrier for small flip neighborhoods on this evaluator: preserving its
flatness keeps the score tied at 0.03125, while breaking it enough to lower DC
causes other grid points to exceed the baseline supnorm.

Recommended next direction:

Continue with a more global constrained formulation if more time is available:
iteratively add all newly violated frequency clusters, but optimize over a
larger structured move set or use a solver that can handle second-order cone
magnitude constraints directly. The linearized MILP showed the right diagnostic
behavior, but the reduced peak set was too narrow and the expanded model became
too slow for this pass.
