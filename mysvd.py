import numpy as np
import math

np.random.seed(0)

def compang(a, b, c, d):
    x = np.arctan2(b - c, a + d)
    y = np.arctan2(b + c, a - d)
    return (x - y) / 2, (x + y) / 2

def dense_rot(n, i, j, theta):
    J = np.eye(n)
    J[i, i] = np.cos(theta)
    J[j, j] = np.cos(theta)
    J[i, j] = -np.sin(theta)
    J[j, i] = np.sin(theta)
    return J

def jacobirot(A, i, j):
    n_rows = A.shape[0]
    n_cols = A.shape[1]
    a = A[i, i]
    b = A[i, j]
    c = A[j, i]
    d = A[j, j]
    theta, phi = compang(a, b, c, d)
    J_theta = dense_rot(n_rows, i, j, theta)
    J_phi = dense_rot(n_cols, i, j, phi)
    return J_theta, J_phi

def svd(A_in):
    A = np.copy(A_in)
    n_rows, n_cols = A.shape
    is_square = n_rows == n_cols

    U = np.eye(n_rows)
    V = np.eye(n_cols)

    max_iterations = 10000
    tolerance = 1e-15 * np.linalg.norm(A_in)

    for _ in range(max_iterations):
        max_off_diag_val_squared = 0.0
        p, q = 0, 1
        for i in range(min(n_rows, n_cols)):
            for j in range(i + 1, min(n_rows, n_cols)):
                if i < n_rows and j < n_cols:
                    current_off_diag_sq = abs(A[i, j])**2 + abs(A[j, i])
                    if current_off_diag_sq > max_off_diag_val_squared:
                        max_off_diag_val_squared = current_off_diag_sq
                        p, q = i, j

        if math.sqrt(max_off_diag_val_squared) < tolerance: # Convergence check
            break

        J_theta, J_phi = jacobirot(A, p, q)
        A = J_theta @ A @ J_phi
        U = U @ J_theta.T
        V = V @ J_phi

    S = np.diag(A).copy()
    VT = V.T.copy()

    abs_S = np.abs(S)
    sorted_indices = np.argsort(abs_S)[::-1]
    S_sorted = abs_S[sorted_indices]

    U_sorted = U[:, sorted_indices] if U.shape[1] == n_rows else U[:, sorted_indices[:U.shape[1]]]
    VT_sorted = VT[sorted_indices, :] if VT.shape[0] == n_cols else VT[sorted_indices[:VT.shape[0]], :]
    S_sorted = S_sorted[:min(n_rows, n_cols)]
    U_sorted = U_sorted[:, :min(n_rows, n_cols)]
    VT_sorted = VT_sorted[:min(n_rows, n_cols), :]

    return U_sorted, S_sorted, VT_sorted
