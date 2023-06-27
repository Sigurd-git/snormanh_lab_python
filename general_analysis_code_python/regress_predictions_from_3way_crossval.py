from general_analysis_code_python.regress_weights_from_2way_crossval import regress_weights_from_2way_crossval
import numpy as np
from general_analysis_code_python.subdivide import subdivide
from general_analysis_code_python.svd_for_regression import svd_for_regression
from general_analysis_code_python.regression_weights import regression_weights

def regress_predictions_from_3way_crossval(F, Y, **kwargs):
    # defaults
    I = {
        'test_folds': 10,
        'train_folds': 10,
        'method': 'ridge',
        'K': None,
        'std_feats': True,
        'groups': np.ones(F.shape[1]),
        'demean_feats': True,
        'regularization_metric': 'unnormalized-squared-error',
        'MAT_file': '',
        'F_test': None
    }
    
    # update defaults with user provided arguments
    I.update(kwargs)

    n_samples, n_features = F.shape
    n_data_vecs = Y.shape[1] # number of data vectors

    # initializations
    Yh = np.empty_like(Y)

    
    # folds
    if np.isscalar(I["test_folds"]):
        n_folds = I["test_folds"]
        fold_indices = subdivide(n_samples, I["test_folds"])
    else:
        assert np.ndim(I["test_folds"]) == 1
        fold_indices = I["test_folds"]
        n_folds = np.max(fold_indices)
    r = np.zeros((n_folds, n_data_vecs))
    mse = np.zeros((n_folds, n_data_vecs))
    B = np.zeros((n_features + 1, n_data_vecs, n_folds))
    # iterate through each fold
    for test_fold in range(1,n_folds+1):
        # train and testing folds
        test_indices = fold_indices == test_fold
        train_indices = np.logical_not(test_indices)
        F_train = F[train_indices]
        Y_train = Y[train_indices]
        
        if I['method']=='least-squares':
        #since we are not doing cross-validation, we can just use the least-squares solution.
        #This is actually 2-way.
            U, s, V, mF, normF = svd_for_regression(F_train, I["std_feats"], I["demean_feats"], I["groups"])
            B_train = np.full((n_features + 1, n_data_vecs), np.nan)
            for i in range(n_data_vecs):
                y_train_i = Y_train[:, i]
                xi = ~np.isnan(y_train_i)

                B_train[:, i] = regression_weights(y_train_i[xi], U, s, V, mF, normF, I["method"], I["K"], I['demean_feats']).squeeze()
        else:
            # Call the function regress_weights_from_2way_crossval for the training set
            B_train, best_K, mse, r, _, _ = regress_weights_from_2way_crossval(
                F_train, 
                Y_train, 
                folds=I['train_folds'], 
                method=I['method'],
                K=I['K'],
                std_feats=I['std_feats'],
                groups=I['groups'],
                demean_feats=I['demean_feats'],
                regularization_metric=I['regularization_metric']
            )

        if I['F_test'] is not None:
            F_test = I['F_test'][test_indices]
        else:
            F_test = F[test_indices]

        # add the bias term to the test set
        F_test = np.hstack((np.ones((F_test.shape[0], 1)), F_test))
        
        # apply the weights to the test features
        Yh[test_indices] = F_test @ B_train

        # calculate the accuracy metrics
        for i in range(n_data_vecs):
            r[test_fold-1, i] = np.corrcoef((Yh[test_indices, i], Y[test_indices, i]))[0, 1]
            mse[test_fold-1, i] = np.nanmean((Yh[test_indices, i]- Y[test_indices, i])**2)
        
        # save weights
        B[:, :, test_fold-1] = B_train

    # calculate the mean squared error
    overall_mse = np.mean((Yh - Y)**2, axis=0)

    return Yh, overall_mse, r, np.arange(n_folds)+1, mse, B

if __name__ == "__main__":
    N = 10000
    P = 100
    sig = 3
    F = np.random.randn(N, P)
    w = np.random.randn(P, 2)
    y = np.dot(F, w) 

    Yh, overall_mse, r, test_folds, mse, B = regress_predictions_from_3way_crossval(F, y, folds=4, method='ridge', std_feats=False, demean_feats=True)
    
    print(overall_mse)

    Yh, overall_mse, r, test_folds, mse, B = regress_predictions_from_3way_crossval(F, y, folds=4, method='least-squares', std_feats=False, demean_feats=True)
    print(overall_mse)
    pass