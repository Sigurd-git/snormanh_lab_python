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
        fold_indices = subdivide(n_samples, I["test_folds"])
    else:
        assert np.ndim(I["test_folds"]) == 1
        fold_indices = I["test_folds"]
    all_folds = np.unique(fold_indices)
    n_folds = len(all_folds)
    r = np.zeros((n_folds, n_data_vecs))
    mse = np.zeros((n_folds, n_data_vecs))
    B = np.zeros((n_features + 1, n_data_vecs, n_folds))
    # iterate through each fold
    for index,test_fold in enumerate(all_folds):
        # train and testing folds
        test_indices = fold_indices == test_fold
        train_indices = np.logical_not(test_indices)
        F_train = F[train_indices]
        Y_train = Y[train_indices]
        if np.isscalar(I["train_folds"]):
            train_fold_indices = I["train_folds"]
        elif np.ndim(I["train_folds"]) == 1:
            train_fold_indices = I["train_folds"][train_indices]
            assert len(train_fold_indices) == len(F_train)
            
        else:
            raise ValueError("train_folds must be a scalar or a 1D array")
        
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
            B_train, best_K, _, _, _, _ = regress_weights_from_2way_crossval(
                F_train, 
                Y_train, 
                folds=train_fold_indices, 
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

            r[index, i] = np.corrcoef((Yh[test_indices, i], Y[test_indices, i]))[0, 1]
            mse[index, i] = np.nanmean((Yh[test_indices, i]- Y[test_indices, i])**2)
        
        # save weights
        B[:, :, index] = B_train

    # calculate the mean squared error
    overall_mse = np.mean((Yh - Y)**2, axis=0)

    return Yh, overall_mse, r, np.arange(n_folds)+1, mse, B

if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # make some data using sklearn's make_regression function-- very handy!
    X, y = make_regression(n_samples=1000,
                        n_features=100, 
                        noise=10,
                        random_state=0,
                        effective_rank=20,
                        n_targets=2)

    groups = np.array([1]*200+[2]*200+[3]*200+[4]*200+[5]*200)

    Yh, overall_mse, r, test_folds, mse, B = regress_predictions_from_3way_crossval(X, y, train_folds=groups,test_folds=groups, method='ridge', std_feats=False, demean_feats=False)
    
    print(r.mean(axis=0))

    Yh, overall_mse, r, test_folds, mse, B = regress_predictions_from_3way_crossval(X, y, train_folds=groups,test_folds=groups, method='least-squares', std_feats=False, demean_feats=True)
    print(r.mean(axis=0))
    pass