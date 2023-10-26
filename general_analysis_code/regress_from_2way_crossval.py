import numpy as np
from himalaya.ridge import RidgeCV,Ridge
from himalaya.backend import set_backend
import numpy as np
from sklearn.model_selection import GroupKFold
from sam_code.regress_weights_from_2way_crossval import regress_weights_from_2way_crossval
alphas= np.logspace(-100, 100, 201,base=2)

def wrapper_cv(X, y, groups):
    n_split = len(np.unique(groups))
    cv = GroupKFold(n_splits=n_split)
    for train_index, test_index in cv.split(X, y, groups):
        yield train_index, test_index

def regress_from_2way_crossval_himalaya(X, Y, groups, alphas=alphas,refit=True,autocast=False):
    cv = wrapper_cv(X, Y, groups)
    if autocast:
        import torch
        set_backend('torch')
        with torch.cuda.amp.autocast(enabled=True):
            model = RidgeCV(alphas=alphas, cv=cv,fit_intercept=True)
            model.fit(X, Y)
    else:
        model = RidgeCV(alphas=alphas, cv=cv,fit_intercept=True)
        model.fit(X, Y)
    best_alphas = model.best_alphas_
    #alert if best_alpha is at the edge of the range
    for index,best_alpha in enumerate(best_alphas):
        if best_alpha == alphas[0] or best_alpha == alphas[-1]:
            print(f'Warning: best alpha for target{index} is {best_alpha}, which is at the edge of the range')
        
    
    if refit:
        if autocast:
            with torch.cuda.amp.autocast(enabled=True):
                refit_model = Ridge(alpha=best_alphas,fit_intercept=True)
                refit_model.fit(X, Y)
        else:
            refit_model = Ridge(alpha=best_alphas,fit_intercept=True)
            refit_model.fit(X, Y)

        return refit_model, best_alphas
    else:
        return best_alphas


def regress_from_2way_crossval_sam(X, Y, groups, alphas=alphas,refit=True):
    
    B, best_K, mse, r,_,_ = regress_weights_from_2way_crossval(X, 
                Y, 
                folds=groups, 
                method='ridge',
                std_feats=False,
                demean_feats=True,
                regularization_metric="unnormalized-squared-error")
    if refit:
        refit_model = Ridge(alpha=alphas,fit_intercept=True)
        
        # Here we construct simulated data just to enable the model to call the fit method. This step is to bypass certain checks.
        dummy_X = np.array(range(len(B[1:])))[np.newaxis,:]
        dummy_y = np.dot(dummy_X, B[1:]) + B[0]
        refit_model.fit(dummy_X, dummy_y)
        
        # set the weights
        refit_model.intercept_ = B[0]
        refit_model.coef_ = B[1:]

        return refit_model, best_K
def regress_from_2way_crossval(X, Y, groups, alphas=alphas,refit=True,autocast=False,backend='himalaya'):
    if backend == 'himalaya':
        return regress_from_2way_crossval_himalaya(X, Y, groups, alphas=alphas,refit=refit,autocast=autocast)
    elif backend == 'sam':
        return regress_from_2way_crossval_sam(X, Y, groups, alphas=alphas,refit=refit)
    else:
        raise ValueError(f'backend {backend} is not supported')
if __name__ == '__main__':
    n_samples, n_features, n_targets = 1000, 500, 4
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_targets)
    groups = np.random.randint(0, 3, n_samples)

    model = regress_from_2way_crossval_sam(X, Y, groups)
    
    