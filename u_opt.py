"""
Usage: u_opt.py

This script generates a convex combination that minimizes the L2 norm of the difference between said convex combination
and a random unitary matrix.
"""

from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import minimize
from datetime import datetime as dt
import numpy as np

MATRIX_SIZE = 4

# Define extended exponent arrays
II_s = np.array([
    [0.5403, 0, 0, 0, 0.8415, 0, 0, 0],
    [0, 0.5403, 0, 0, 0, 0.8415, 0, 0],
    [0, 0, 0.5403, 0, 0, 0, 0.8415, 0],
    [0, 0, 0, 0.5403, 0, 0, 0, 0.8415]
])
XX_s = np.array([
    [0, 0, 0, 0, -0.8415, 0, 0, 0.8415],
    [0, 0.5403, 0, 0, 0, 0, 0.8415, 0],
    [0, 0, 0.5403, 0, 0, 0.8415, 0, 0],
    [0, 0, 0, 0.5403, 0.8415, 0, 0, 0]
])
YY_s = np.array([
    [0.5403, 0, 0, 0, 0, 0, 0, -0.8415],
    [0, 0.5403, 0, 0, 0, 0, 0.8415, 0],
    [0, 0, 0.5403, 0, 0, 0.8415, 0, 0],
    [0, 0, 0, 0.5403, -0.8415, 0, 0, 0]
])
ZZ_s = np.array([
    [0.5403, 0, 0, 0, 0.8415, 0, 0, 0],
    [0, 0.5403, 0, 0, 0, -0.8415, 0, 0],
    [0, 0, 0.5403, 0, 0, 0, -0.8415, 0],
    [0, 0, 0, 0.5403, 0, 0, 0, 0.8415]
])
HH_s = np.array([
    [0.5403, 0, 0, 0, 0.4207, 0.4207, 0.4207, 0.4207],
    [0, 0.5403, 0, 0, 0.4207, -0.4207, 0.4207, -0.4207],
    [0, 0, 0.5403, 0, 0.4207, 0.4207, -0.4207, -0.4207],
    [0, 0, 0, 0.5403, 0.4207, -0.4207, -0.4207, 0.4207]
])
ext_mat_arr = [II_s, XX_s, YY_s, ZZ_s, HH_s]

SET_SIZE = len(ext_mat_arr)


# Utility functions
def gen_random_unitary(n):
    """
    Generates a random unitary matrix via QR Decomposition.
    Parameters:
        n - size of matrix
    Returns:
        a unitary matrix.
    """
    m = np.reshape(np.random.random(n * n) +
                   np.random.random(n * n) * 1j, (n, n))
    q, r = np.linalg.qr(m, "complete")
    r_diag = np.diagonal(r)
    d = np.diagflat(r_diag)
    l = d / np.absolute(r_diag)
    return q.dot(l)


def euclidean_distance(m1):
    """
    Calculates the Euclidean distances between the matrix m1 and the
    matrices exp**(i I4), exp**(i XX), exp**(i YY), exp**(i ZZ) and exp**(i HH)
    Parameters:
        m1 - matrix 1
    Returns:
        list of euclidean distances between the matrices
        max euclidean distance
        min euclidean distance
    """
    edl = [None] * 5
    max_val = 0
    min_val = MATRIX_SIZE

    for i in range(5):
        edl[i] = np.linalg.norm(m1 - ext_mat_arr[i])
        if edl[i] > max_val:
            max_val = edl[i]
        if edl[i] < min_val:
            min_val = edl[i]

    return edl, max_val, min_val


def pretty_print_matrix(m):
    s = [[str(e) for e in row] for row in m]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


# Define constraints for trust region approach
def obj_func(x, A):
    """
    Objective function definition for optimization.
    :param x: current position vector
    :param A: target matrix A
    :return: the result of the objective function
    """
    t_mat = np.sum(ext_mat_arr * x[:, None, None], axis=0)
    return np.sum(np.square(A - t_mat))


cvx_ineq_bounds = Bounds([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
cvx_eq_constr = LinearConstraint([[1, 1, 1, 1, 1]], [1], [1])

# Construct unitary matrix representation
A = gen_random_unitary(MATRIX_SIZE)
print("Generated random unitary matrix: ")
pretty_print_matrix(A)
m_r = A.real
m_i = A.imag
A = np.concatenate((m_r, m_i), axis=1)
dist_metrics = euclidean_distance(A)
print("Distances for trivial cases:", dist_metrics[0])
print("Greatest distance for trivial case:", dist_metrics[1])
print("Least distance for trivial case:", dist_metrics[2])

# Use trust region approach
print("\nRunning trust region approach...")
tStart = dt.now()
x_initial = np.array([1, 0, 0, 0, 0])
res = minimize(obj_func, x_initial, method='trust-constr', args=A, constraints=cvx_eq_constr, bounds=cvx_ineq_bounds)
print("Finished trust region approach — time elapsed:", dt.now() - tStart)

# Calculate result matrix
res_mat_ext = np.sum(ext_mat_arr * res.x[:, None, None], axis=0)
res_mat = res_mat_ext[:, :MATRIX_SIZE] + res_mat_ext[:, MATRIX_SIZE:] * 1j
print("\nApproximation matrix:")
pretty_print_matrix(res_mat)

# Print results
print("\nConvex Coefficients:", res.x)
print("Approximation distance from random matrix:", np.linalg.norm(res_mat_ext - A))
# %%
