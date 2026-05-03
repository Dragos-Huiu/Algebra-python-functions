import numpy as np

# Class for Lower-Upper matrix factorization
class LU:
    def Doolittle(A):
        A = np.array(A, dtype=float)
        n = np.size(A, 1)
        L = np.eye(n)
        U = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i <= j:
                    U[i, j] = A[i, j] - sum([L[i, k] * U[k, j] for k in range(i)])
                else:
                    L[i, j] = (A[i, j] - sum([L[i, k] * U[k, j] for k in range(j)])) / U[j, j]
        return L, U
    
    def Crout(A):
        A = np.array(A, dtype=float)
        n = np.size(A, 1)
        L = np.zeros((n, n))
        U = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i >= j:
                    L[i, j] = A[i, j] - sum([L[i, k] + U[k, j] for k in range(j)])
                else:
                    U[i, j] = (A[i, j] - sum([L[i, k] + U[k, j] for k in range(i)])) / L[i, i]
        return L, U
    
    def Cholesky(A):
        A = np.array(A, dtype=float)
        n = np.size(A, 1)
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    L[i, i] = np.sqrt(A[i, i] - sum([L[i, k] ** 2 for k in range(j)]))
                else:
                    L[i, j] = (A[i, j] - sum([L[i, k] * L[j, k] for k in range(j)])) / L[j, j]
        return L