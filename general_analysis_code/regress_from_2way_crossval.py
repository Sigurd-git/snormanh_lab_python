import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.datasets import make_regression
alphas = np.logspace(-100, 99, 200, base=2)




def wrapper_cv(X, y, groups):
    n_split = len(np.unique(groups))
    cv = GroupKFold(n_splits=n_split)
    for train_index, test_index in cv.split(X, y, groups):
        yield train_index, test_index


def regress_from_2way_crossval_himalaya(
    X,
    Y,
    groups,
    alphas=alphas,
    backend="torch_cuda",
    half=True,
    n_alphas_batch=10,
    n_targets_batch=None,
    fit_intercept=True,
):
    cv = wrapper_cv(X, Y, groups)
    set_backend(backend)
    if half:
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
    model = RidgeCV(
        alphas=alphas,
        cv=cv,
        fit_intercept=fit_intercept,
        solver_params={
            "nsample_large": True,
            "n_alphas_batch": n_alphas_batch,
            "n_targets_batch": n_targets_batch,
        },
    )
    model.fit(X, Y)
    best_alphas = model.best_alphas_
    
    return model, best_alphas



if __name__ == "__main__":

    n_samples, n_features, n_targets = 1000, 10, 10
    # n_samples, n_features, n_targets = 90000, 2000, 150
    X, Y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=0,
        noise=1000,
        bias=3.0,
    )
    groups = np.random.randint(0, 3, n_samples)



    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    refit_model, best_K_h = regress_from_2way_crossval_himalaya(
        X,
        Y,
        groups,
        backend="torch_cuda",
    )
    model = Ridge(alpha=best_K_h, fit_intercept=True)
    model.assign(refit_model.coef_, refit_model.intercept_, X)
    model.cv_scores_ = refit_model.cv_scores_
    Y_hat_h = model.predict(X)

    rs_1 = [np.corrcoef(Y_hat_h[:, i], Y[:, i])[0, 1] for i in range(Y.shape[1])]

    
    refit_model, best_K_h = regress_from_2way_crossval_himalaya(
        X,
        Y,
        groups,
        backend="torch",
    )
    model = Ridge(alpha=best_K_h, fit_intercept=True)
    model.assign(refit_model.coef_, refit_model.intercept_, X)
    model.cv_scores_ = refit_model.cv_scores_
    # import dill
    # #pickle an generater
    # with open('model.pkl', 'wb') as f:
    #     dill.dump(model, f)
    
    # #load the pickle
    # with open('model.pkl', 'rb') as f:
    #     model = dill.load(f)
    Y_hat_h = model.predict(X)
    # Y_hat_h = model.predict(X)
    rs_2 = [np.corrcoef(Y_hat_h[:, i], Y[:, i])[0, 1] for i in range(Y.shape[1])]
    pass
