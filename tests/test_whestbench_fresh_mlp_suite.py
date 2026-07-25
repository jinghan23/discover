import numpy as np

from repro.aicrowd_whestbench.evaluate_fresh_mlp_suite import (
    _pair_inversion_fraction,
    _spearman,
)


def test_rank_comparisons_identify_preserved_and_reversed_order():
    ordered = np.array([1.0, 2.0, 3.0])
    reversed_order = ordered[::-1]

    assert _spearman(ordered, ordered) == 1.0
    assert _spearman(ordered, reversed_order) == -1.0
    assert _pair_inversion_fraction(ordered, ordered) == 0.0
    assert _pair_inversion_fraction(ordered, reversed_order) == 1.0
