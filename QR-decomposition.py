# This file proposes implementations of QR-decomposition methods for solving
# quadratic and overdetermined linear systems of algebraic equations

import numpy as np

def givens_rotation(A):
    """
    QR-decomposition of rectangular matrix A using the Givens rotation method.
    """

    def rotation_matrix(a, b):

        r = np.sqrt(a**2 + b** 2)
        return a/r, -b/r

    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
    n, m = A.shape
    Q = np.eye(n)
    R = np.copy(A)

    rows, cols = np.tril_indices(n, -1, m)
    for (row, col) in zip(rows, cols):
        # Calculation of the Givens rotation matrix and zero values
        # of the lower elements of the triangular matrix.
        if R[row, col] != 0:
            c, s = rotation_matrix(R[col, col], R[row, col])

            R[col], R[row] = R[col]*c + R[row]*(-s), R[col]*s + R[row]*c
            Q[:, col], Q[:, row] = Q[:, col]*c + Q[:, row]*(-s), Q[:, col]*s + Q[:, row]*c

    return Q[:, :m], R[:m]


def Hausholder(A):
    """
    QR-decomposition of a rectangular matrix A using the Householder reflection method.
    """

    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
    n, m = A.shape
    Q = np.eye(n)
    R = np.copy(A)

    for k in range(m):
        v = np.copy(R[k:, k]).reshape((n-k, 1))
        v[0] = v[0] + np.sign(v[0]) * np.linalg.norm(v)
        v = v / np.linalg.norm(v)
        R[k:, k:] = R[k:, k:] - 2 * v @ v.T @ R[k:, k:]
        Q[k:] = Q[k:] - 2 * v @ v.T @ Q[k:]

    return Q[:m].T, R[:m]


# To check the solutions, we use the standard deviation of SME
def SME(A, b, x):
    return 1/max(b) * np.sqrt(1/len(b) * np.sum(abs(np.dot(A, x) - b) ** 2))



if __name__=='__main__':

    # Consider an example of a square matrix:
    shape = (100, 100)
    A = np.random.normal(0, 1, shape)
    b = np.random.normal(0, 1, shape[0])

    Q, R = givens_rotation(A)
    x = np.linalg.solve(R, Q.T @ b)
    print('Givens: ', SME(A, b, x))

    Q, R = Hausholder(A)
    x = np.linalg.solve(R, Q.T @ b)
    print('Hausholder: ', SME(A, b, x))


    # Now consider an overdetermined system:
    shape = (100, 5)
    A = np.random.normal(0, 1, shape)
    b = np.random.normal(0, 1, shape[0])

    Q, R = givens_rotation(A)
    x = np.linalg.solve(R, Q.T @ b)
    print('Overdetermined Givens: ', SME(A, b, x))

    Q, R = Hausholder(A)
    x = np.linalg.solve(R, Q.T @ b)
    print('Overdetermined Hausholder: ', SME(A, b, x))
