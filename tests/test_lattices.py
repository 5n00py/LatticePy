import pytest
import numpy as np

from LatticePy import lattices

def test_compute_gram_matrix() -> None:
    # Test vector from Bremner, 2012, Example 1.16
    X = np.array([[-7, -7, 4, -8, -8],
                  [1, 6, -5, 8, -1],
                  [-1, 1, 4, -7, 8]])
    GM = lattices.compute_gram_matrix(X)
    E = np.array([[242, -125, 8],
                  [-125, 127, -79],
                  [8, -79, 131]])
    assert np.array_equal(GM, E)
