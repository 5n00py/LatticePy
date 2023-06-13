import pytest
import numpy as np

from LatticePy import lattices


def test_transform_basis() -> None:
    for _ in range(100):
        X = np.array([[1, 0], [0, 1]])
        T = lattices.transform_basis(X, 5, 5)
        print("Transformed Basis: ", T)
        assert lattices.ldet(T) == pytest.approx(1.0, abs=1e-3)


def test_gram_matrix() -> None:
    # Test vector from Bremner, 2012, Example 1.16
    X = np.array([[-7, -7, 4, -8, -8], [1, 6, -5, 8, -1], [-1, 1, 4, -7, 8]])
    GM = lattices.gram_matrix(X)
    E = np.array([[242, -125, 8], [-125, 127, -79], [8, -79, 131]])
    assert np.array_equal(GM, E)


def test_ldet() -> None:
    # Test vector from Bremner, 2012, Example 1.16
    X = np.array([[-7, -7, 4, -8, -8], [1, 6, -5, 8, -1], [-1, 1, 4, -7, 8]])
    det = lattices.ldet(X)

    # Note: The tolerance is relative to the expected value and is 1e-6
    # (i.e., can differ by up to 1e-6 * expected_value), but this could be
    # changed by passing a different value to pytest.approx if needed...
    assert det == pytest.approx(np.sqrt(618829))


def test_gram_schmidt() -> None:
    # Test vector from Bremner, 2012, Example 3.3
    X = np.array([[3, -1, 5], [-5, 2, -1], [-3, 9, 2]])
    M, Y = lattices.gram_schmidt(X)
    assert np.allclose(
        M,
        np.array(
            [
                [3, -1, 5],
                [-109 / 35, 48 / 35, 15 / 7],
                [1521 / 566, 1859 / 283, -169 / 566],
            ]
        ),
    )
    assert np.allclose(
        Y, np.array([[1, 0, 0], [-22 / 35, 1, 0], [-8 / 35, 909 / 566, 1]])
    )


def test_reduce_lll() -> None:
    X = np.array([[-2, 7, 7, -5], [3, -2, 6, -1], [2, -8, -9, -7], [8, -9, 6, -4]])
    Y = lattices.reduce_lll(X, 1)
    Y_exp = np.array([[2, 3, 1, 1], [2, 0, -2, -4], [-2, 2, 3, -3], [3, -2, 6, -1]])
    assert np.array_equal(Y, Y_exp)
