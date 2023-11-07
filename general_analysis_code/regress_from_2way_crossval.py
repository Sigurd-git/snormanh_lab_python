import numpy as np
from himalaya.ridge import RidgeCV, Ridge
from himalaya.backend import set_backend
import numpy as np
from sklearn.model_selection import GroupKFold
from sam_code.regress_weights_from_2way_crossval import regress_weights_from_2way_crossval
import dask.array as da
alphas = np.logspace(-100, 100, 201, base=2)
import dask
import torch

#make_regression
from sklearn.datasets import make_regression

def wrapper_cv(X, y, groups):
    n_split = len(np.unique(groups))
    cv = GroupKFold(n_splits=n_split)
    for train_index, test_index in cv.split(X, y, groups):
        yield train_index, test_index


def regress_from_2way_crossval_himalaya(X, Y, groups, alphas=alphas,backend='torch',half=True):
    cv = wrapper_cv(X, Y, groups)
    set_backend(backend)
    if half:
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
    model = RidgeCV(alphas=alphas, cv=cv, fit_intercept=True,solver_params={'nsample_large':True})
    model.fit(X, Y)
    best_alphas = model.best_alphas_
    return model, best_alphas



def regress_from_2way_crossval_sam(X, Y, groups, alphas=alphas, refit=True):

    B, best_K, mse, r, _, _ = regress_weights_from_2way_crossval(X,
                                                                 Y,
                                                                 folds=groups,
                                                                 method='ridge',
                                                                 std_feats=False,
                                                                 demean_feats=True,
                                                                 regularization_metric="unnormalized-squared-error")
    if refit:
        refit_model = Ridge(alpha=alphas, fit_intercept=True)

        # set the weights
        refit_model.assign(B[1:],B[0],X)

        return refit_model, best_K


def regress_from_2way_crossval(X, Y, groups, alphas=alphas, refit=True, autocast=False, backend='himalaya'):
    if backend == 'himalaya':
        return regress_from_2way_crossval_himalaya(X, Y, groups, alphas=alphas, refit=refit, autocast=autocast)
    elif backend == 'sam':
        return regress_from_2way_crossval_sam(X, Y, groups, alphas=alphas, refit=refit)
    else:
        raise ValueError(f'backend {backend} is not supported')


if __name__ == '__main__':

    import time
    n_samples, n_features, n_targets = 90000, 200, 300
    X,Y = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, random_state=0, noise=1000, bias=3.0)
    groups = np.random.randint(0, 3, n_samples)
    
    start_time = time.time()
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    refit_model, best_K_h = regress_from_2way_crossval_himalaya(X, Y, groups,backend='numpy')
    Y_hat_h = refit_model.predict(X)
    rs_h = [np.corrcoef(Y_hat_h[:,i], Y[:,i])[0, 1] for i in range(Y.shape[1])]

    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f'himalaya elapsed time: {elapsed_time}')


    # set_backend('numpy')
    # start_time = time.time()
    # refit_model, best_K_s = regress_from_2way_crossval_sam(X, Y, groups)
    # Y_hat_s = refit_model.predict(X)
    # rs_s = [np.corrcoef(Y_hat_s[:,i], Y[:,i])[0, 1] for i in range(Y.shape[1])]
    # end_time = time.time()
    # elapsed_time2 = end_time-start_time
    # print(f'sam elapsed time: {elapsed_time2}')
    
    # print(f'Acceleration fold: {elapsed_time2/elapsed_time}')
    # #isclose(rs_h, rs_s)
    # print(np.where(np.isclose(rs_h, rs_s,atol=1e-3)==False))

    
    pass