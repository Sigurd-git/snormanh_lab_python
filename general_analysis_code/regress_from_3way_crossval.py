import numpy as np
import torch
from general_analysis_code.regress_from_2way_crossval import (
    regress_from_2way_crossval_himalaya,
)


def correlate_columns(x, y):
    # Correlate each column of x with the corresponding column of y
    xv = x - torch.mean(x, dim=0)
    yv = y - torch.mean(y, dim=0)
    xvss = torch.sum((xv * xv), dim=0)
    yvss = torch.sum((yv * yv), dim=0)
    result = torch.matmul(xv.T, yv) / torch.sqrt(torch.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return torch.maximum(
        torch.minimum(result, torch.tensor(1.0)), torch.tensor(-1.0)
    ).diagonal()


def regress_from_3way_crossval(
    X,
    Y,
    groups,
    alphas=torch.logspace(-100, 99, 200, base=2),
    backend="torch_cuda",
    half=False,
    n_alphas_batch=10,
    n_targets_batch=None,
    train_groups=None,
):
    groups = torch.as_tensor(groups, dtype=torch.int64)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    X = torch.as_tensor(X, dtype=torch.float32)
    all_folds = torch.unique(groups)
    Y_hat = torch.zeros_like(Y)
    n_data_vecs = Y.shape[1]
    r = torch.zeros((len(all_folds), n_data_vecs))
    coefs = []
    intercepts = []
    train_cv_scores = []
    for index, test_fold in enumerate(all_folds):
        test_fold = int(test_fold)
        # train and testing folds
        test_indices = groups == test_fold
        train_indices = torch.logical_not(test_indices)
        F_train = X[train_indices]
        Y_train = Y[train_indices]
        if train_groups is None:
            train_fold_indices = groups[train_indices]
        else:
            assert len(train_groups) == len(all_folds)
            train_fold_indices = train_groups[test_fold]
            assert len(train_fold_indices) == len(F_train)
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

        coef = model.coef_.cpu()  # n_features x n_ycolumn
        intercept = model.intercept_.cpu()  # n_ycolumn
        cv_score = (
            model.cv_scores_.cpu()
        )  # n_trainfolds x n_alphas x subject/electrode/rep
        coefs.append(coef)
        intercepts.append(intercept)
        train_cv_scores.append(cv_score)

    coefs = torch.stack(coefs)  # n_folds x n_features x n_ycolumn
    intercepts = torch.stack(intercepts)  # n_folds x n_ycolumn
    train_cv_scores = torch.stack(train_cv_scores)
    # n_folds x n_trainfolds x n_alphas x subject/electrode/rep

    return Y_hat, r, coefs, intercepts, best_alphas, train_cv_scores


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    n_samples = 1000
    n_features = 100
    # make some data using sklearn's make_regression function-- very handy!
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=10,
        random_state=0,
        effective_rank=20,
        n_targets=2,
    )
    n_folds = 5
    groups = np.random.randint(0, n_folds, n_samples)
    train_groups = []
    unique_groups = np.unique(groups)
    for i in range(5):
        test_group_i = unique_groups[i]
        n_train_samples = groups[groups != test_group_i].shape[0]
        # create random train groups
        train_groups_i = np.random.randint(0, n_folds, n_train_samples)
        train_groups.append(train_groups_i)
    train_groups = np.array(train_groups, dtype=object)
    (
        Y_hat,
        r,
        coefs,
        intercepts,
        best_alphas,
        train_cv_scores,
    ) = regress_from_3way_crossval(
        X, y, groups, backend="torch_mps", train_groups=train_groups
    )

    print(r)
