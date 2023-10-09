import numpy as np
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from general_analysis_code.ridge_via_svd import ridge_via_svd




def regression_weights(y, U, s, V, mF, normF, method, K, demean_feats):
    """
    A function that computes regression weights based on a method
    y is 1d array of data
    """
    assert not np.isnan(y).any()

    assert (len(y.shape) == 1), 'y must be a 1d array'

    # de-mean data
    ym = y - np.mean(y) if demean_feats else y

    # weights using all of the data
    if method == 'least-squares':
        B = V @ (np.diag(1./s) @ (U.T @ ym))
        
    elif method == 'pcreg': # principal components regression
        n_K = len(K)
        B = np.empty((V.shape[0], n_K))
        for j in range(n_K):
            B[:, j] = V[:, :K[j]] @ (1./s[:K[j]] * (U[:, :K[j]].T @ ym))
        
    elif method == 'ridge':
        B = ridge_via_svd(ym, U, s, V, K)
        
    elif method == 'pls':
        n_K = len(K)
        B = np.empty((U.shape[1]+1, n_K))
        for j in range(n_K):
            Z = U * s
            pls = PLSRegression(n_components=K[j])
            pls.fit(Z, ym)
            B[:, j] = pls.coef_
        B = B[1:, :]
        B = V @ B
        
    elif method == 'lasso':
        lasso = linear_model.Lasso(alpha=K, normalize=False)
        B = lasso.fit(U @ np.diag(s) @ V.T, ym).coef_
        
    else:
        raise ValueError('No valid method for {}'.format(method))

    # rescale weights to remove effect of normalization
    #dot devide
    if (len(B.shape) == 2) & (len(normF.shape) == 1):
        B = B / normF[:, np.newaxis]
    else:
        B = B/normF
    if len(B.shape) ==1:
        B = B[:, np.newaxis]
    new_B = np.empty((B.shape[0]+1, B.shape[1]))
    for j in range(B.shape[1]):
        Bj = B[:, j]
        # add ones regressor
        if demean_feats:
            Bj = np.hstack((np.mean(y) - mF @ Bj, Bj))
        else:
            Bj = np.hstack((0,Bj))
        new_B[:, j] = Bj
    B = new_B

    return B

if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler

    # Assume we have data y and X
    np.random.seed(0)

    X = np.random.rand(10000, 10)

    true_weights = np.random.rand(10)
    y = X @ true_weights + np.random.randn(10000)/10

    # Apply Standard Scaler to the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform SVD on scaled data
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)

    # Some parameters
    demean_feats = True
    method = 'ridge'
    K = [1,10,100]  # Not used in 'least-squares', just for the function signature

    # Call function
    B = regression_weights(y, U, s, Vt.T, scaler.mean_, scaler.scale_, method, K, demean_feats)

    # 
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    Yhat = X @ B
    err = Yhat-y[:, np.newaxis]
    print(err)

