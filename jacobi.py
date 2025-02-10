import numpy as np
import math

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
    a = A[i, i]
    b = A[i, j]
    c = A[j, i]
    d = A[j, j]

    theta, phi = compang(a, b, c, d)

    J_theta = dense_rot(A.shape[0], i, j, theta)
    J_phi = dense_rot(A.shape[1], i, j, phi) # Corrected to A.shape[1] as V is right matrix

    return J_theta, J_phi

def svd(A_in):
    A = np.copy(A_in) # avoid modifying input
    n_rows, n_cols = A.shape
    is_square = n_rows == n_cols

    U = np.eye(n_rows)
    V = np.eye(n_cols)

    max_iterations = 100 # Set a limit to iterations for now
    tolerance = 1e-14

    for _ in range(max_iterations):
        off_diagonal_sum_squared = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if i != j:
                    off_diagonal_sum_squared += A[i, j]**2

        if off_diagonal_sum_squared < tolerance:
            break

        max_off_diag_val = 0.0
        p, q = 0, 1
        for i in range(n_rows):
            for j in range(i + 1, n_cols): # Iterate over upper triangle to avoid repeating pairs
                if abs(A[i, j]) > max_off_diag_val:
                    max_off_diag_val = abs(A[i, j])
                    p, q = i, j

        if max_off_diag_val < tolerance: # Convergence check based on max off-diagonal element
            break


        J_theta, J_phi = jacobirot(A, p, q)

        if is_square: # For square matrices, U and V are square of same size
            A = J_theta.T @ A @ J_phi
            U = U @ J_theta.T
            V = V @ J_phi
        else: # For non-square matrices, U and V have different shapes.
            A = J_theta.T @ A @ J_phi
            U_temp = J_theta.T @ U if U.shape[1] == n_rows else U @ J_theta.T # Check if rotation needs to be applied from left or right
            U = U_temp
            V = V @ J_phi


    S = np.diag(A).copy() # Extract diagonal values as singular values
    VT = V.T.copy() # Transpose V to get VT

    # Ensure singular values are non-negative and sorted in descending order
    abs_S = np.abs(S)
    sorted_indices = np.argsort(abs_S)[::-1]
    S_sorted = abs_S[sorted_indices]

    U_sorted = U[:, sorted_indices] if U.shape[1] == n_rows else U[:, sorted_indices[:U.shape[1]]] # Handle case where U is not square
    VT_sorted = VT[sorted_indices, :] if VT.shape[0] == n_cols else VT[sorted_indices[:VT.shape[0]], :] # Handle case where VT is not square


    return U_sorted, S_sorted, VT_sorted


if __name__ == '__main__':
    # Example usage (for testing purposes, not for submission)
    n = 3
    A = np.random.randn(n, n)
    U, S, VT = svd(A)

    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("VT shape:", VT.shape)

    print("Reconstruction error:", np.linalg.norm(U @ np.diag(S) @ VT - A) / np.linalg.norm(A))
    print("U Orthogonality error:", np.linalg.norm(U.T @ U - np.eye(n)) / n)
    print("V Orthogonality error:", np.linalg.norm(VT @ VT.T - np.eye(n)) / n)
    print("S positive error:", np.linalg.norm(S - np.abs(S)))
    print("S sorted error:", np.linalg.norm(S - np.sort(np.abs(S))[::-1]))
