"""Fast n=16 Heilbronn construction for the unit square.

The point set is the 180-degree rotationally symmetric 31-by-33 integer-grid
configuration with minimum triangle area 7/341.
"""

import numpy as np


_LATTICE_POINTS = np.array(
    [
        [8, 33],
        [29, 33],
        [0, 31],
        [21, 31],
        [10, 23],
        [31, 23],
        [2, 21],
        [23, 21],
        [8, 12],
        [29, 12],
        [0, 10],
        [21, 10],
        [10, 2],
        [31, 2],
        [2, 0],
        [23, 0],
    ],
    dtype=float,
)

_SCALE = np.array([31.0, 33.0])


def main() -> np.ndarray:
    """Return a copy so callers cannot mutate the module-level candidate."""
    return _LATTICE_POINTS / _SCALE


points = main()
