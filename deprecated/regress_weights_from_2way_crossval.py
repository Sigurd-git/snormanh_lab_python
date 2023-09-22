import numpy as np

from general_analysis_code_python.svd_for_regression import svd_for_regression
from general_analysis_code_python.subdivide import subdivide
from general_analysis_code_python.regression_weights import regression_weights
def regress_weights_from_2way_crossval(F, Y, **kwargs):



    n_samples, n_features = F.shape
    assert Y.shape[0] == n_samples
    n_data_vecs = Y.shape[1] # number of data vectors
    
    # optional arguments
    I = {
        "folds": 5,
        "method": "ridge",
        "K": None,
        "std_feats": True,
        "groups": np.ones(n_features),
        "demean_feats": True,
        "regularization_metric": "unnormalized-squared-error",
        "warning": True
    }
    
    I.update(kwargs)  # update defaults with user provided arguments

    # regularization parameter
    if I["K"] is None:
        if I["method"] == 'least-squares':
            I["K"] = []
        elif I["method"] in ['ridge', 'lasso']:
            I["K"] = 2. ** np.arange(-100, 101)
        elif I["method"] in ['pls', 'pcreg']:
            I["K"] = np.arange(1, round(n_features / 3) + 1)
        else:
            raise ValueError(f"No valid method for {I['method']}")
    else:
        raise NotImplementedError("K must be None now.")

    # groups
    I["groups"] = I["groups"].flatten()
    n_groups = np.max(I["groups"])
    assert np.array_equal(np.arange(1, n_groups + 1), np.unique(I["groups"]))
    assert len(I["groups"]) == n_features

    # folds
    if np.isscalar(I["folds"]):
        n_folds = I["folds"]
        fold_indices = subdivide(n_samples, I["folds"])
    else:
        assert np.ndim(I["folds"]) == 1
        fold_indices = I["folds"]
    all_folds = np.unique(fold_indices)
    n_folds = len(all_folds)
    # number of components to test
    n_K = len(I["K"])

    # calculate predictions
    mse = np.full((n_folds, max(n_K, 1), n_data_vecs), np.nan)
    r = mse.copy()
    demeaned_mse = mse.copy()
    normalized_mse = mse.copy()
    
    for index,test_fold in enumerate(all_folds):
        # train and testing folds
        test_fold_indices = fold_indices == test_fold
        train_fold_indices = np.logical_not(test_fold_indices)
        
        # concatenate training data
        y_train = Y[train_fold_indices, :]
        F_train = F[train_fold_indices, :]
        
        # format features and compute svd
        U, s, V, mF, normF = svd_for_regression(F_train, I["std_feats"], I["demean_feats"], I["groups"])
        
        # prediction from test features
        F_test = F[test_fold_indices, :]
        F_test = np.hstack((np.ones((F_test.shape[0], 1)), F_test))
        for i in range(n_data_vecs):
            # estimate weights from training data

            #keepdim
            y_train_i = y_train[:, i]
            B = regression_weights(y_train_i, U, s, V, mF, normF, I["method"], I["K"], I['demean_feats'])
            yh = F_test @ B
            err = yh - Y[test_fold_indices, i: i+1]
            mse[index, :, i] = np.nanmean(err ** 2, axis=0)
            for k in range(n_K):
                r[index, k, i] = np.corrcoef(yh[:,k], Y[test_fold_indices, i])[0, 1]
            # demeaned_mse[test_fold, :, i] = np.corrcoef_variance_sensitive_symmetric(yh, Y[test_fold_indices, i])
            # normalized_mse[test_fold, :, i] = np.corrcoef_normalized_squared_error(yh, Y[test_fold_indices, i])

    match I["regularization_metric"]:
        case 'pearson':
            stat = r
        case 'unnormalized-squared-error':
            stat = -mse
        case 'demeaned-squared-error':
            stat = demeaned_mse
        case 'normalized-squared-error':
            stat = normalized_mse

    if I['method'] == 'least-squares':
        best_K = np.full(n_data_vecs, np.nan)
    
    else:
        best_K = np.full(n_data_vecs, np.nan)
        for i in range(n_data_vecs):
            # best regularization value
            best_K_index = np.nanargmax(np.nanmean(stat[:, :, i], axis=0))
            best_K[i] = I["K"][best_K_index]
            
            # check if the best regularizer is on the boundary
            if I["warning"]:
                if I["method"] == 'ridge' and (best_K_index == 0 or best_K_index == n_K - 1):
                    print(f"Best regularizer is on the boundary of possible values\nK={best_K[i]}")
                elif I["method"] == 'pls' and best_K_index == n_K - 1:
                    print(f"Best regularizer is on the boundary of possible values\nK={best_K[i]}")
                elif I["method"] == 'pcreg' and best_K_index == n_K - 1:
                    print(f"Best regularizer is on the boundary of possible values\nK={best_K[i]}")

    #estimate weights from all data
    U, s, V, mF, normF = svd_for_regression(F, I["std_feats"], I["demean_feats"], I["groups"])
    B = np.full((n_features + 1, n_data_vecs), np.nan)
    for i in range(n_data_vecs):
        B[:, i] = regression_weights(Y[:, i], U, s, V, mF, normF, I["method"], [best_K[i]], I["demean_feats"]).reshape(-1)
    
    return B, best_K, mse, r, 0, 0


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=10000,
                        n_features=100, 
                        noise=100,
                        random_state=5,
                        effective_rank=20,
                        n_targets=1)
    y = y.reshape(-1,1)
    #demean and std y
    mean_y, std_y = np.mean(y), np.std(y)
    y = (y-mean_y)/std_y
    groups = np.array([1]*200+[2]*200+[3]*200+[4]*200+[5]*200)

    B, best_K, mse, r, _, _ = regress_weights_from_2way_crossval(X, y, method='least-squares', folds=5, demean_feats=True, std_feats=True,regularization_metric='pearson')
    #all of these should be close to zero
    # print(X@B[1:]+B[0]-y)
    xxx = X@B[1:]
    yyy = y
    xxx = xxx.reshape(-1)
    yyy = yyy.reshape(-1)
    print(np.corrcoef(xxx,yyy))

    B, best_K, mse, r, _, _ = regress_weights_from_2way_crossval(X, y, method='ridge', folds=5, demean_feats=True, std_feats=True,regularization_metric='pearson')

    #all of these should be close to zero
    xxx = X@B[1:]
    yyy = y
    xxx = xxx.reshape(-1)
    yyy = yyy.reshape(-1)
    print(np.corrcoef(xxx,yyy))    # print(X@B[1:]+B[0]-y)
    pass