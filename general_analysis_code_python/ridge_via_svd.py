import numpy as np

def ridge_via_svd(y, U, s, V, k):
    """
    Efficiently calculates beta weights from ridge regression analysis given the
    SVD (F = U*diag(S)*V') of a feature matrix.
    """
    D = V.shape[0]
    nK = len(k)

    # correlation of principal components with the demeaned data vector
    Uty = U.T @ y

    # betas
    b = np.empty((D, nK))
    for i in range(nK):
        r = s / (s**2 + k[i])
        b[:, i] = V @ (np.diag(r) @ Uty)  # weight principal components and transform back
    
    return b

if __name__ == '__main__':
    # Define y, U, s, V and k
    X = np.random.rand(10000, 3)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    b = np.array([1, 0.5, 1.0])
    y = X @ b+ np.random.randn(10000)/10
    k = np.array([0,0.1, 0.5, 1.0])


    # Compute ridge regression weights
    bh = ridge_via_svd(y, U, s, Vt.T, k)

    print("Ridge regression weights:")
    print(bh)
    for j in range(bh.shape[1]):
        print("k = %g, bh = %s,b=%s" % (k[j], bh[:, j],b))
