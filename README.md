
# LatticePy

LatticePy is a Python library for lattice algorithms, primarily based on a 
number of seminal publications in the area, including Lenstra, Lenstra and 
Lovász's introduction of the LLL algorithm and the work of Fincke and Pohst. 
For a comprehensive overview, M. R. Bremner's "Lattice Basis Reductions" (2012) 
is recommended.

This library contains a collection of functions for performing various 
computations on lattices, which can be used as standalone tools or incorporated 
into larger systems as needed.

This library is intended for use in conjunction with the NumPy library for 
numerical computations in Python.

## Introduction to Lattices

A lattice is a mathematical structure that can be visualized as a regular, 
repeating arrangement of points in space, spanning multiple dimensions. 
In more precise terms, a lattice can be defined as follows:

Let `n ≥ 1` and let `x1, x2, ..., xn` be a basis of `R^n`. The lattice with 
dimension `n` and basis `x1, x2, ..., xn` is the set `L` of all linear 
combinations of the basis vectors with integral coefficients:

`L = Zx1 + Zx2 + · · · + Zxn = Σ ai xi (for i = 1 to n)`,

where `a1, a2, ..., an` are integers. The basis vectors `x1, x2, ..., xn` are 
said to generate or span the lattice.

Each basis vector `xi` is an `n`-tuple `(xi1, ..., xin)`, and these vectors 
can be arranged to form an `n × n` matrix.

**In this library, the basis vectors of a lattice are regarded as row vectors.** 
This approach allows for easier manipulation and transformation of lattices, 
especially in higher-dimensional spaces. Keep this in mind when creating or 
transforming lattices using LatticePy.


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

## Key Applications

Lattices are an integral part of numerous fields of study, including 
mathematics, physics, computer science, cryptography, and many more:

- Cryptography: The hardness of certain problems in lattice theory forms the
  basis for post-quantum croyptography. The Shortest Vector Problem (SVP) and
  the Closest Vector Problem (CVP) which are believed to be "hard" to solve on
  lattices, are fundamental to many lattice-based cryptographic schemes.

- Cryptanalysis: Lattices play also a rone in analyzing and breaking
  cryptographic systems. For example the knapsack cryptosystem was found to be
  vulnerable to attacks utilizing lattice-based techniques. Similarly,
  Coppersmith's attack is a lattice-based method that can be used to find small
  roots of polynomial equations modulo a composite number or a prime and can be
  applied to attack cryptographic schemes like RSA, when a part of the private
  key is known or when padding is deterministic and messages follow a specific
  pattern.

- Physics: Lattices are used to model crystal structures in solid state
  physics. Ech point in the lattice represents an atom or molescule.

- Coding Theory: Lattices have applications mainly in designing error detection
  and error correction.

Beyond these, lattices are used in various branches of mathematics and computer
science.

## License

LatticePy is released under the [MIT License](https://choosealicense.com/licenses/mit/).
