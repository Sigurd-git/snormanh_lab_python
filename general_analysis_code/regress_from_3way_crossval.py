from general_analysis_code.regress_from_2way_crossval import regress_from_2way_crossval_himalaya
from sam_code.regress_predictions_from_3way_crossval import regress_predictions_from_3way_crossval
import numpy as np

def correlate_columns(x, y):
    # Correlate each column of x with the corresponding column of y
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.T, yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, np.array(1.0)), np.array(-1.0)).diagonal()

def regress_from_3way_crossval_himalaya(X, Y, groups,alphas= np.logspace(-100, 100, 201,base=2), save_parameters=False,half=False,return_best_alphas=False):

    all_folds = np.unique(groups)
    Y_hat = np.zeros_like(Y)
    n_data_vecs = Y.shape[1]
    r = np.zeros((len(all_folds), n_data_vecs))
    coefs = []
    intercepts = []
    for index,test_fold in enumerate(all_folds):
        # train and testing folds
        test_indices = groups == test_fold
        train_indices = np.logical_not(test_indices)
        F_train = X[train_indices]
        Y_train = Y[train_indices]
        train_fold_indices = groups[train_indices]
        model,best_alphas = regress_from_2way_crossval_himalaya(F_train, Y_train, train_fold_indices,alphas=alphas,refit=True,autocast=half)
        F_test = X[test_indices]
        
        # apply the weights to the test features
        Y_hat_fold = model.predict(F_test)
        Y_hat[test_indices] = Y_hat_fold

        # calculate the accuracy metrics
        r_fold = correlate_columns(Y_hat_fold, Y[test_indices])
        r[index] = r_fold

        coef = np.array(model.coef_) # n_features x n_ycolumn
        intercept = np.array(model.intercept_) # n_ycolumn
        if save_parameters:
            coefs.append(coef)
            intercepts.append(intercept)
            
    coefs = np.array(coefs) # n_folds x n_features x n_ycolumn
    intercepts = np.array(intercepts) # n_folds x n_ycolumn
    match save_parameters, return_best_alphas:
        case True,True:
            return Y_hat, r, coefs, intercepts, best_alphas
        case True,False:
            return Y_hat, r, coefs, intercepts
        case False,True:
            return Y_hat, r, best_alphas
        case False,False:
            return Y_hat, r

def regress_from_3way_crossval_sam(X, y, groups,alphas= np.logspace(-100, 100, 201,base=2), save_parameters=False):
    n_ycolumn = y.shape[1]
    n_folds = len(np.unique(groups))

    cv_score = np.full((n_folds, n_ycolumn), np.nan)
    cv_coef = np.full((n_folds, X.shape[-1],n_ycolumn), np.nan)
    cv_intercept = np.full((n_folds, n_ycolumn), np.nan)

    y_hat, _, r, _, _, B = regress_predictions_from_3way_crossval(
            X,
            y,
            test_folds=groups,
            train_folds=groups,
            method='ridge',
            demean_feats=True,
            std_feats=False
        )
    # B  n_features+1 x rep/electrode x n_folds


    for k in range(n_folds):
        cv_score[k, :] = correlate_columns(
                y_hat[groups == (k + 1), :], y[groups == (k + 1), :]
            )
        cv_coef[k, :, :] = B[1:, :, k]
        cv_intercept[k, :] = B[0, :, k]
       
    if save_parameters:
        return y_hat,cv_score,cv_coef,cv_intercept
    else:
        return y_hat,cv_score

def regress_from_3way_crossval(X, Y, groups,alphas= np.logspace(-100, 100, 201,base=2), save_parameters=True,backend='himalaya',half=False,return_best_alphas=False):
    if backend == 'himalaya':
        return regress_from_3way_crossval_himalaya(X, Y, groups,alphas=alphas, save_parameters=save_parameters,half=half,return_best_alphas=return_best_alphas)
    elif backend == 'sam':
        assert half == False, "sam backend does not support half precision"
        assert return_best_alphas == False, "sam backend does not support return_best_alphas"
        return regress_from_3way_crossval_sam(X, Y, groups,alphas=alphas, save_parameters=save_parameters)
        
        

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

    Y_hat, r, models = regress_from_3way_crossval(X, y, groups, save_models=True)
    
    coefs = np.array([model.coef_ for model in models]) # n_folds x n_features x n_ycolumn
    intercepts = np.array([model.intercept_ for model in models]) # n_folds x n_ycolumn
    pass