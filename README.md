
# LatticePy

LatticePy is a Python library for lattice algorithms, primarily based on a 
number of seminal publications in the area, including Lenstra, Lenstra and 
Lovász's introduction of the LLL algorithm and the work of Fincke and Pohst. 
For a comprehensive overview, M. R. Bremner's "Lattice Basis Reductions" (2012) 
is recommended.

This library contains a collection of functions for performing various 
computations on lattices, which can be used as standalone tools or incorporated 
into larger systems as needed.

## Features

The following functions are included in LatticePy:

1. `transform_basis_with_unimodular()`: Performs unimodular row operations on a matrix, providing another basis of the same lattice.
2. `compute_gram_matrix()`: Computes the Gram matrix of a given lattice.
3. `compute_lattice_determinant()`: Computes the determinant of a given lattice.
4. `perform_gram_schmidt()`: Performs the classical Gram-Schmidt algorithm to convert an arbitrary basis of R^n into an orthogonal basis.
5. `perform_lll_reduction()`: Performs the Lenstra-Lenstra-Lovasz (LLL) lattice basis reduction algorithm.
6. `cholesky_DU()`: Performs a Cholesky decomposition of a symmetric positive definite mxm matrix.
7. `fincke_pohst()`: The Fincke-Pohst Algorithm for finding all vectors in a lattice with length less than a given upper bound.
8. `fincke_pohst_with_lll()`: Combines the Fincke-Pohst algorithm with the LLL reduction for increased efficiency.

All functions accept NumPy ndarrays as inputs and return either NumPy ndarrays 
or Python lists as outputs, ensuring compatibility with a wide range of Python 
libraries and tools.

## Usage

```python
import LatticePy
import numpy as np

# creating a lattice
lattice = np.array([[1, 0], [0, 1]])  # example 2D lattice

# computing the Gram matrix
gram_matrix = LatticePy.compute_gram_matrix(lattice)
```

## Note

This library is intended for use in conjunction with the NumPy library for 
numerical computations in Python.

## License

LatticePy is released under the [MIT License](https://choosealicense.com/licenses/mit/).
