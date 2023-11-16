# from sklearn.linear_model import LinearRegression, RidgeCV
from himalaya.ridge import RidgeCV
from sklearn.model_selection import KFold, train_test_split
from general_analysis_code.regress_from_2way_crossval import regress_from_2way_crossval_himalaya
from sklearn.datasets import make_regression
import numpy as np
from himalaya.backend import set_backend
alphas = np.logspace(-100, 99, 200, base=2)

def stack_from_prefit(
    Xs,
    Y,
    base_models,
    groups=None,
    alphas=alphas,
    backend="torch_cuda",
    half=True,
    n_alphas_batch=10,
    n_targets_batch=None,
):
    set_backend(backend)
    base_regressors = [base_model.predict(X) for base_model,X in zip(base_models,Xs)]
    base_regressors = np.array(base_regressors).transpose(2, 1, 0) # n_targets x n_samples x n_models 
    stacked_weights = []
    for X_stack, Y_column in zip(base_regressors, Y.transpose()):
        
        stacked_model, best_alpha = regress_from_2way_crossval_himalaya(
        X_stack,
        Y_column,
        groups,
        alphas=alphas,
        backend=backend,
        half=half,
        n_alphas_batch=n_alphas_batch,
        n_targets_batch=n_targets_batch,
        fit_intercept=False,
    )
        stacked_weights.append(stacked_model.coef_)
    stacked_weights = np.array(stacked_weights).transpose() #n_models x n_targets
    return stacked_weights



if __name__ == "__main__":
    backend = "torch_cuda"
    half = True
    n_alphas_batch = 10
    n_targets_batch = None
    # 加载示例数据集
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

    # 划分数据集
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, Y, groups, test_size=0.2, random_state=0
    )


    # 定义基础模型
    base_models = [RidgeCV(), RidgeCV()]

    # 定义最终模型


    # 定义 KFold
    kfold = KFold(n_splits=5, shuffle=True)

    for model in base_models:
        # fit the model
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Model Score: {score}")
        stacked_weights = stack_from_prefit(
            X_test,
            y_test,
            base_models,
            groups=groups_test,
            alphas=alphas,
            backend=backend,
            half=half,
            n_alphas_batch=n_alphas_batch
        )

    # save the weights
    np.save('stacked_weights.npy', stacked_weights)

