# MSTD Improvement Notes

Initial score: 1.04.

Improved candidate:

```text
A = {0, 1, 2, 4, 5, 9, 12, 13, 17, 20, 21, 22, 24, 25}
```

This set has span 25, full sumset on that span, and four missing
differences:

```text
|A + A| = 51
|A - A| = 47
missing differences in [-25, 25]: {-14, -6, 6, 14}
score = 51 / 47 = 1.0851063829787233
```

Search summary:

- Started from the Conway baseline and ran randomized local search over
  bitset-encoded subsets of `{0, ..., 29}`.
- Polished candidates with single-, double-, and triple-flip neighborhoods.
- Verified the best value by exact span-normalized enumeration: for each
  span `m = 1..29`, fixed `min(A)=0` and `max(A)=m`, enumerated all internal
  bits, and scored `|A+A|/|A-A|`.
- The best normalized span found was `m=25`, with ratio `51/47`.

The implementation in `work/improved_program.py` is deterministic and returns
the candidate directly; no runtime search is required.
