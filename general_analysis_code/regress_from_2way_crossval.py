import numpy as np
from himalaya.ridge import RidgeCV,Ridge
from himalaya.backend import set_backend
import numpy as np
from sklearn.model_selection import GroupKFold

alphas= np.logspace(-100, 100, 201,base=2)

def wrapper_cv(X, y, groups):
    n_split = len(np.unique(groups))
    cv = GroupKFold(n_splits=n_split)
    for train_index, test_index in cv.split(X, y, groups):
        yield train_index, test_index

def ridge_custom_groups(X, Y, groups, alphas=alphas,refit=True,autocast=False):
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


if __name__ == '__main__':
    n_samples, n_features, n_targets = 1000, 500, 4
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_targets)
    groups = np.random.randint(0, 3, n_samples)

    model = ridge_custom_groups(X, Y, groups)
    
    