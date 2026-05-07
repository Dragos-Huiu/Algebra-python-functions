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
    
class QR:
    def Gram_Schmidt(A):
        A = np.array(A, dtype=float)
        m, n = A.shape
        Q = np.zeros((m, n))
        Q[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0])
        for j in range(1, n):
            u = A[:, j];
            u = u - sum([np.dot(u, Q[:, i]) * Q[:, i] for i in range(j)])
            Q[:, j] = u / np.linalg.norm(u)
        return Q, np.transpose(Q) @ A

    def Givens(A):
        A = np.array(A, dtype=float)
        R = A
        n, m = A.shape
        Q = np.eye(n)
        for j in range(m - 1):
            for i in range(j + 1, n):
                a = R[j, j]
                b = R[i, j]
                r = np.sqrt(a ** 2 + b ** 2)
                c = a / r
                s = b / r
                G = np.eye(n)
                G[i, i] = c
                G[j, j] = c
                G[i, j] = -s
                G[j, i] = s
                R = G @ R
                Q = Q @ np.transpose(G)
                
        return Q, R

    def HouseHolder(A):
        A = np.array(A, dtype=float)
        R = A
        m, n = A.shape
        Q = np.eye(n)
        I = np.eye(n)
        for j in range(min(m, n)):
            x = R[j:m, j]
            s = np.sign(x[0])
            if s == 0:
                s = 1
            v = np.zeros((n, 1))
            v[j:m, 0] = x
            v = v + s * np.linalg.norm(x) * I[:, j:j+1]
            H = I - 2 * (v @ np.transpose(v)) / (np.transpose(v) @ v)
            R = H @ R
            Q = Q @ H
        
        return Q, R
            