import numpy as np

def Jacobi(A, B, n, x0, tol=1e-6, max_iter=1000):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    x0 = np.array(x0, dtype=float)
    x1 = np.zeros((n, 1))
    for _ in range(max_iter):
        for i in range(n):
            s = sum([A[i, j] * x0[j, 0] for j in range(n) if i != j])
            x1[i, 0] = (B[i] - s) / A[i, i]
        if np.linalg.norm(x1 - x0, np.inf) < tol:
            break
        x0 = x1.copy()
    return x1

def GaussSiedel(A, B, n, x0, tol=1e-6, max_iter=1000):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    x1 = np.zeros((n, 1))
    for _ in range(max_iter):
        for i in range(n):
            s1 = sum([A[i, j] * x1[j, 0] for j in range(i)])
            s2 = sum([A[i, j] * x0[j, 0] for j in range(i, n)])
            x1[i, 0] = (B[i] - s1 - s2) / A[i, i]
            if np.linalg.norm(x1 - x0, np.inf) < tol:
                break
            x0 = x1.copy()
    return x1

def SOR(A, B, n, x0, tol=1e-6, max_iter=1000):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    x1 = np.zeros((n, 1))
    omega = 1 # (0 < omega < 2)
    for _ in range(max_iter):
        for i in range(n):
            s1 = sum([A[i, j] * x1[j, 0] for j in range(i)])
            s2 = sum([A[i, j] * x0[j, 0] for j in range(i, n)])
        x1[i, 0] = (1 - omega) * x0[i, 0] + omega * (B[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x1 - x0, np.inf) < tol:
            break
        x0 = x1.copy()
    return x1