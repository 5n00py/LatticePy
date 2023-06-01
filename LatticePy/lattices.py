"""
This python file contains a collection of functions for performing various 
computations on lattices.

The functions provided in this file can be used to perform various operations 
on lattices such as transforming a basis with unimodular operations, computing 
the Gram matrix of a lattice, calculating the determinant of a lattice, 
performing the Gram-Schmidt orthogonalization process, executing the 
Lenstra-Lenstra-Lovasz (LLL) lattice basis reduction algorithm, decomposing a 
symmetric positive definite matrix using the Cholesky method, and finally 
performing the Fincke-Pohst enumeration algorithm to find all short lattice vectors.

These functions are useful for tasks involving lattice operations and can be 
used as standalone tools or incorporated into larger systems as needed.

All functions take NumPy ndarrays as inputs and return NumPy ndarrays or Python 
lists as outputs, ensuring compatibility with a wide range of Python libraries 
and tools.

Function Descriptions:

1. transform_basis(): Performs unimodular row operations on a matrix, providing 
    another basis of the same lattice.
2. compute_gram_matrix(): Computes the Gram matrix of a given lattice.
3. compute_lattice_determinant(): Computes the determinant of a given lattice.
4. perform_gram_schmidt(): Performs the classical Gram-Schmidt algorithm to 
        convert an arbitrary basis of R^n into an orthogonal basis.
5. perform_lll_reduction(): Performs the Lenstra-Lenstra-Lovasz (LLL) lattice 
        basis reduction algorithm.
6. cholesky_DU(): Performs a Cholesky decomposition of a symmetric positive 
        definite mxm matrix.
7. fincke_pohst(): The Fincke-Pohst Algorithm for finding all vectors in a 
        lattice with length less than a given upper bound.
8. fincke_pohst_with_lll(): Combines the Fincke-Pohst algorithm with the LLL 
        reduction for increased efficiency.

NOTE: This file is intended for use in conjunction with the NumPy library for 
        numerical computations in Python.
"""

import numpy as np
import math

from copy import deepcopy
from numpy import ndarray
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.random import randint
from typing import List
from typing import Tuple


def transform_basis(X: ndarray, k: int, r: int) -> ndarray:
    """
    Apply unimodular row operations on a matrix, providing another basis of the
    same lattice.

    A unimodular row operation on a matrix can be one of the following
    elementary row operations:
    * Multiply any row by -1,
    * Interchange any two rows,
    * Add an integral multiple of any row to any other row.
    By applying these operations, we can obtain another basis of the same lattice.

    Parameters
    ----------
    X : ndarray
        A 2D ndarray representing a matrix whose rows form a basis of a lattice.
    k : int
        The number of unimodular row operations to apply to the matrix.
    r : int
        A range parameter to limit the integral multiples.

    Returns
    -------
    ndarray
        A 2D ndarray whose row vectors form another basis of the same lattice.
    """

    m, _ = X.shape

    for _ in range(k):
        operation = randint(1, 4)

        # Multiply a row by -1
        if operation == 1:
            row = randint(0, m)
            X[row] *= -1

        # Interchange two rows
        elif operation == 2:
            row1, row2 = randint(0, m, 2)
            X[[row1, row2]] = X[[row2, row1]]

        # Add an integral multiple of a row to another row
        else:
            row1, row2 = randint(0, m, 2)
            while row1 == row2:
                row2 = randint(0, m)

            scalar = randint(1, r + 1)
            X[row1] += scalar * X[row2]

    return X


def compute_gram_matrix(X: ndarray) -> ndarray:
    """
    Computes the Gram matrix of a given lattice.

    The Gram matrix delta(L) of the lattice L is the m x m matrix in which the
    (i,j) entry is the scalar product of the i-th and j-th basis vectors. It's
    associated with a set of vectors that spans an m-dimensional lattice in R^n.

    Parameters
    ----------
    X : ndarray
        A 2D ndarray representing a matrix. Its rows x1, x2, ..., xm are linearly
        independent vectors in R^m (m <= n) that span an m-dimensional lattice L.

    Returns
    -------
    ndarray
        The m x m Gram matrix of the lattice L.

    Example
    -------
    >>> import numpy as np
    >>> from LatticePy import lattices
    >>> X = np.array([[-7, -7, 4, -8, -8], [1, 6, -5, 8, -1],[-1, 1, 4, -7, 8]])
    >>> G = lattices.compute_gram_matrix(X)
    >>> assert np.array_equal(G, np.array([[242, -125, 8], [-125, 127, -79], [8, -79, 131]]))
    """
    Xt = X.transpose()
    G = X.dot(Xt)
    return G


def lattice_determinant(X: ndarray) -> float:
    """
    Compute the determinant of a given lattice.

    The determinant of a lattice L is defined as the square root of the
    determinant of its Gram matrix. The determinant of a lattice does not depend
    on the choice of basis.
    There is a geometric interpretation: The determinant is the m-dimensional
    volume of the parallelipiped in R^n whose edges are the lattice basis vectors.

    Parameters
    ----------
    X : ndarray
        A ndarray representing a matrix. Its rows x1, x2, ..., xm are
        linearly independent vectors in R^m (m <= n) that span an m-dimensional
        lattice L.

    Returns
    -------
    float
        The determinant of the Lattice L.

    Example
    -------
    >>> import numpy as np
    >>> from LatticePy import lattices
    >>> X = np.array([[-7, -7, 4, -8, -8], [1, 6, -5, 8, -1],[-1, 1, 4, -7, 8]])
    >>> G = lattices.compute_gram_matrix(X)
    >>> det = lattices.lattice_determinant(X)
    >>> assert np.allclose(det, np.sqrt(618829))
    """
    G = compute_gram_matrix(X)
    detG = np.linalg.det(G)
    result = math.sqrt(detG)
    return result


def perform_gram_schmidt(X: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Perform the classical Gram-Schmidt algorithm for converting an arbitrary basis of
    R^n into an orthogonal basis. Vectors are not normalized in this function.

    Parameters
    ----------
    X : ndarray
        A 2D ndarray representing an nxn matrix in which row i is the vector xi.
        The rows x1,x2,...xn build a basis in R^n.

    Returns
    -------
    Tuple[ndarray, ndarray]
        A tuple containing two 2D ndarrays. The first represents an nxn matrix Y in
        which row i is the vector yi. The rows y1,y2,...,yn build an orthogonal basis in R^n.
        The second represents the Gram-matrix M (the matrix of GSO coefficients) with X = MY.

    Note
    ----
    The Gram-Schmidt basis vectors y1, y2, ... ,yn are usually not in the lattice
    generated by x1,...,xn. In general y1,...,yn are not integral linear combinations of x1,...,xn.
    """
    n, m = X.shape

    X = X.astype(float)

    M = np.zeros((n, m))
    Y = deepcopy(X)

    for i in range(n):
        for j in range(i):
            M[i][j] = X[i].dot(Y[j]) / Y[j].dot(Y[j])
            Y[i] = Y[i] - M[i][j] * Y[j]

    for i in range(n):
        M[i][i] = 1

    return Y, M


def perform_lll_reduction(X: ndarray) -> ndarray:
    """
    Perform Lenstra-Lenstra-Lovasz (LLL) algorithm, a polynomial time lattice
    reduction algorithm.

    Given a "bad" basis of a lattice, this algorithm finds a "good" basis where
    each vector is "almost orthogonal" to the span of the previous vectors.
    The basis is called reduced if
    ||x_i*||^2 <= 2 ||x_i+1||^2 for 1 <= i < n.

    Parameters
    ----------
    X : ndarray
        A 2D ndarray representing a matrix where rows are linearly independent
        vectors in R^n that span an m-dimensional lattice L.

    Returns
    -------
    ndarray
        A 2D ndarray representing a matrix X' whose rows are a reduced basis of
        the basis X.

    Raises
    ------
    ValueError
        If the number of basis vectors exceeds the space dimensionality.
    """
    m, n = X.shape

    if m > n:
        raise ValueError("The number of basis vectors exceeds the space dimensionality")

    X = X.astype(float)

    Y, M = perform_gram_schmidt(X)

    i = 1
    while i < m:
        for j in range(i - 1, -1, -1):
            mu = int(round(M[i, j]))
            X[i] -= mu * X[j]
            if abs(M[i, j]) > 0.5:
                Y, M = perform_gram_schmidt(X)

        if (i > 0) and (norm(Y[i - 1]) ** 2 > 2 * norm(Y[i]) ** 2):
            X[i], X[i - 1] = X[i - 1], X[i]  # Interchange X[i] and X[i-1]
            Y, M = perform_gram_schmidt(X)
            i -= 1
        else:
            i += 1

    return X


def cholesky_DU(G: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Perform a Cholesky decomposition of a symmetric positive definite mxm matrix G.
    The output is a diagonal mxm matrix D and an upper triangular matrix U with ones on
    the diagonal such that G = U^tDU. This is a slightly modified version of the classical
    Cholesky decomposition R^tR.

    Parameters
    ----------
    G : ndarray
        A 2D ndarray representing a symmetric positive definite mxm matrix.

    Returns
    -------
    D : ndarray
        A 2D ndarray representing the diagonal matrix D.

    U : ndarray
        A 2D ndarray representing the upper triangular matrix U.
    """
    U = G.astype(float)
    m = len(U)

    for j in range(m - 1):
        for i in range(j + 1, m):
            q = -U[i, j] / U[j, j]
            U[i, :] += q * U[j, :]

    D = np.diag(np.diag(U))  # Diagonal matrix with positive entries on the diagonal

    # Ensure U has ones on the diagonal
    U = U / np.sqrt(D)

    return D, U


def fincke_pohst(Bt: ndarray, C: float) -> List[ndarray]:
    """
    The Fincke-Pohst Algorithm is an algorithm for finding not merely one short vector
    in a lattice, but for enumerating all vectors in a lattice with length less than
    a given upper bound C. In particular, this allows us to determine a shortest nonzero
    lattice vector.

    Parameters
    ----------
    Bt : ndarray
        A 2D ndarray representing a matrix whose rows are linearly independent vectors in R^n
        that span an m-dimensional lattice L.

    C : float
        An upper bound for the euclidian square length of the lattice vectors.

    Returns
    -------
    vector_list : list of ndarray
        The lattice vectors with square length at most C.
    """
    m, n = Bt.shape

    if m > n:
        raise ValueError("The number of basis vectors exceeds the space dimensionality")

    Bt = Bt.astype(float)
    B = Bt.transpose()

    G = Bt @ B  # Compute the Gram matrix

    D, U = cholesky_DU(G)  # Perform Cholesky decomposition

    m = len(G)

    results = [np.zeros(m)]

    for k in reversed(range(m)):
        new_results = []

        for r in results:
            x = r.copy()

            Sk = sum(
                D[i, i] * (x[i] + sum(U[i, j] * x[j] for j in range(i + 1, m))) ** 2
                for i in range(k + 1, m)
            )
            Tk = sum(U[k, j] * x[j] for j in range(k + 1, m))

            lower_bound = math.ceil(-math.sqrt((C - Sk) / D[k, k]) - Tk)
            upper_bound = math.floor(math.sqrt((C - Sk) / D[k, k]) - Tk)

            for xk in range(int(lower_bound), int(upper_bound) + 1):
                x[k] = xk
                new_results.append(x.copy())

        results = new_results

    vector_list = [B @ x for x in results]

    vector_list.sort(key=np.linalg.norm)

    return vector_list


def fincke_pohst_with_lll(Bt: ndarray, C: float) -> List[ndarray]:
    """
    Combined with the LLL algorithm, the FP algorithm becomes a more efficient
    hybrid. This function uses the LLL algorithm to modify the quadratic form
    obtained from the Gram matrix of the lattice basis, which diminishes the
    ranges for the components of the partial coordinate vectors.

    Also, it reorders the vectors in the computation of the rational Cholesky
    decomposition of the Gram matrix, increasing the chance that a partial
    coordinate vector cannot be extended.

    Parameters
    ----------
    Bt : ndarray
        The input matrix.
    C : float
        The constant for the range of the result vectors.

    Returns
    -------
    list
        List of sorted short lattice vectors.
    """

    m, n = Bt.shape

    if m > n:
        raise ValueError("The number of basis vectors exceeds the space dimensionality")

    Bt = Bt.astype(float)
    B = Bt.transpose()

    # Compute the Gram matrix.
    G = Bt.dot(B)

    # Perform classical Cholesky decomposition
    L = np.linalg.cholesky(G)
    R = L.transpose()

    Rinv = inv(R)
    Sinv = perform_lll_reduction(Rinv)

    X = inv(Sinv.dot(R))
    S = inv(Sinv)

    permutation = np.argsort(norm(Sinv, axis=0))
    Pinv = np.zeros((m, m))
    Pinv[np.arange(m), permutation] = 1

    P = inv(Pinv)
    SP = S.dot(P)
    H = SP.T.dot(SP)

    # Apply rational cholesky decomposition on H:
    E, V = cholesky_DU(H)

    results = [np.zeros((m))]
    for k in range(m - 1, -1, -1):
        new_results = []

        for r in results:
            z = r.copy()
            Sk = sum(
                E[i, i] * (z[i] + sum(V[i, j] * z[j] for j in range(i + 1, m))) ** 2
                for i in range(k + 1, m)
            )
            Tk = sum(V[k, j] * z[j] for j in range(k + 1, m))

            lower_bound = math.ceil(-math.sqrt((C - Sk) / E[k, k]) - Tk)
            upper_bound = math.floor(math.sqrt((C - Sk) / E[k, k]) - Tk)

            for zk in range(lower_bound, upper_bound + 1):
                z[k] = zk
                new_results.append(z.copy())

        results = new_results

    vector_list = []
    for z in results:
        y = P.dot(z)
        x = X.dot(y)
        w = B.dot(x)

        w = np.round(w).astype(int)
        vector_list.append(w)

    vector_list.sort(key=norm)

    return vector_list
