import numpy as np

from general_analysis_code.regress_from_2way_crossval import (
    regress_from_2way_crossval_himalaya,
)


def correlate_columns(x, y):
    # Correlate each column of x with the corresponding column of y
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.T, yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, np.array(1.0)), np.array(-1.0)).diagonal()


def regress_from_3way_crossval(
    X,
    Y,
    groups,
    alphas=np.logspace(-100, 99, 200, base=2),
    backend="torch_cuda",
    half=False,
    n_alphas_batch=10,
    n_targets_batch=None,
):
    all_folds = np.unique(groups)
    Y_hat = np.zeros_like(Y)
    n_data_vecs = Y.shape[1]
    r = np.zeros((len(all_folds), n_data_vecs))
    coefs = []
    intercepts = []
    train_cv_scores = []
    for index, test_fold in enumerate(all_folds):
        # train and testing folds
        test_indices = groups == test_fold
        train_indices = np.logical_not(test_indices)
        F_train = X[train_indices]
        Y_train = Y[train_indices]
        train_fold_indices = groups[train_indices]
        model, best_alphas = regress_from_2way_crossval_himalaya(
            F_train,
            Y_train,
            train_fold_indices,
            alphas=alphas,
            backend=backend,
            half=half,
            n_alphas_batch=n_alphas_batch,
            n_targets_batch=n_targets_batch,
        )
        F_test = X[test_indices]

        # apply the weights to the test features
        Y_hat_fold = model.predict(F_test)
        Y_hat[test_indices] = Y_hat_fold

        # calculate the accuracy metrics
        r_fold = correlate_columns(Y_hat_fold, Y[test_indices])
        r[index] = r_fold

        coef = np.array(model.coef_.cpu())  # n_features x n_ycolumn
        intercept = np.array(model.intercept_.cpu())  # n_ycolumn
        cv_score = np.array(
            model.cv_scores_.cpu()
        )  # n_trainfolds x n_alphas x subject/electrode/rep
        coefs.append(coef)
        intercepts.append(intercept)
        train_cv_scores.append(cv_score)

    coefs = np.array(coefs)  # n_folds x n_features x n_ycolumn
    intercepts = np.array(intercepts)  # n_folds x n_ycolumn
    train_cv_scores = np.array(
        train_cv_scores
    )  # n_folds x n_trainfolds x n_alphas x subject/electrode/rep

    return Y_hat, r, coefs, intercepts, best_alphas, train_cv_scores


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # make some data using sklearn's make_regression function-- very handy!
    X, y = make_regression(
        n_samples=1000,
        n_features=100,
        noise=10,
        random_state=0,
        effective_rank=20,
        n_targets=2,
    )

    groups = np.array([1] * 200 + [2] * 200 + [3] * 200 + [4] * 200 + [5] * 200)

    Y_hat, r, models = regress_from_3way_crossval(X, y, groups)

    coefs = np.array(
        [model.coef_ for model in models]
    )  # n_folds x n_features x n_ycolumn
    intercepts = np.array([model.intercept_ for model in models])  # n_folds x n_ycolumn
    pass
