import pytest
import numpy as np

from LatticePy import lattices

def test_transform_basis() -> None:
    for _ in range(100):
        X = np.array([[1, 0], [0, 1]])
        T = lattices.transform_basis(X, 5, 5)
        assert lattices.lattice_determinant(T) == pytest.approx(1.0, abs=1e-3)

def test_compute_gram_matrix() -> None:
    # Test vector from Bremner, 2012, Example 1.16
    X = np.array([[-7, -7, 4, -8, -8], [1, 6, -5, 8, -1], [-1, 1, 4, -7, 8]])
    GM = lattices.compute_gram_matrix(X)
    E = np.array([[242, -125, 8], [-125, 127, -79], [8, -79, 131]])
    assert np.array_equal(GM, E)


def test_lattice_determinant() -> None:
    # Test vector from Bremner, 2012, Example 1.16
    X = np.array([[-7, -7, 4, -8, -8], [1, 6, -5, 8, -1], [-1, 1, 4, -7, 8]])
    det = lattices.lattice_determinant(X)

    # Note: The tolerance is relative to the expected value and is 1e-6
    # (i.e., can differ by up to 1e-6 * expected_value), but this could be
    # changed by passing a different value to pytest.approx if needed...
    assert det == pytest.approx(np.sqrt(618829))
