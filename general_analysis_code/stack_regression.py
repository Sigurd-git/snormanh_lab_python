from himalaya.ridge import RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import make_regression
import numpy as np
from himalaya.backend import set_backend
from cvxopt import matrix, solvers

# Set option to not show progress in CVXOPT solver
solvers.options["show_progress"] = False
alphas = np.logspace(-100, 99, 200, base=2)


def compute_S(err):
    """
    Compute the stacking weights for each voxel.

    Parameters:
    - err (dict): A dictionary containing the prediction errors matrix of shape (n_features, n_voxels) for each feature space.

    Returns:
    - S (ndarray): The stacking weights matrix of shape (n_voxels, n_features).
    """
    n_features = len(err)
    n_voxels = err[0].shape[-1]
    P = np.zeros((n_voxels, n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            P[:, i, j] = np.mean(err[i] * err[j], 0)

    # solve the quadratic programming problem to obtain the weights for stacking
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        # solve for stacking weights for every voxel
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(n_features)
    return S


def stack_from_prefit(
    Xs,
    Y,
    base_models,
    backend="torch_cuda",
):
    """
    Xs: list of arrays of shape (n_samples, n_features)
    Y: array of shape (n_samples, n_targets)
    """
    set_backend(backend)
    # get predictions from base models
    pred = []
    for i, model in enumerate(base_models):
        pred.append(model.predict(Xs[i]))  # n_samples x n_targets

    err = dict()
    # compute the prediction errors
    for i in range(len(pred)):
        err[i] = np.array(pred[i] - Y)

    S = compute_S(err)  # n_voxels x n_features
    
    S = S.T  # n_features x n_voxels

    return S


if __name__ == "__main__":
    backend = "torch_cuda"
    half = True
    n_alphas_batch = 10
    n_targets_batch = None
    # 加载示例数据集
    n_samples, n_features, n_targets = 1000, 6, 5
    # n_samples, n_features, n_targets = 90000, 2000, 150
    X, Y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=0,
        noise=100,
        bias=3.0,
    )
    groups = np.random.randint(0, 3, n_samples)

    # 划分数据集
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, Y, groups, test_size=0.2, random_state=0
    )
    set_backend(backend)
    # 定义基础模型
    base_models = [RidgeCV(fit_intercept=True), RidgeCV(fit_intercept=True)]

    # 定义最终模型

    # 定义 KFold
    kfold = KFold(n_splits=5, shuffle=True)

    for model in base_models:
        # fit the model
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Model Score: {score}")
    model_error = RidgeCV(fit_intercept=True)
    X_error, Y_error = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=10,
        noise=100,
        bias=3.0,
    )
    model_error.fit(X_error, Y_error)
    score = model_error.score(X_test, y_test)
    print(f"Model Score: {score}")
    base_models.append(model_error)
    stacked_weights = stack_from_prefit(
        [X_test, X_test, X_test],
        y_test,
        base_models,
        groups=groups_test,
        alphas=alphas,
        backend=backend,
        half=half,
        n_alphas_batch=n_alphas_batch,
    )

    # save the weights
    np.save("stacked_weights.npy", stacked_weights)
