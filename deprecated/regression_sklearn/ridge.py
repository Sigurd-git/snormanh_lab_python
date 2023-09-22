import numpy as np
import sklearn
try:
    import torch
except ImportError:
    pass
from sklearn.metrics import make_scorer
from sklearn.linear_model import RidgeCV
def correlate_columns(x, y):
    # returns matrix with rows corresp. to columns of x, cols to columns of y
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.T, yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, np.array(1.0)), np.array(-1.0)).diagonal()
    
correlation_scorer = make_scorer(lambda x,y: correlate_columns(x,y))
class RidgeDoubleCV(RidgeCV):
    def __init__(self,alphas,alpha_per_target=True,outer_scoring = correlation_scorer, **kwargs):
        super().__init__(alphas=alphas,alpha_per_target=alpha_per_target,**kwargs)
        self.outer_scoring = outer_scoring
        
    def cv_prediction_scores(self, X, y, cv=sklearn.model_selection.KFold(10)):
        """Compute the prediction score for each fold of the cross-validation
        generator.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples, n_targets) n_targets can be 1
            Target values.
        cv : Int or cross-validation generator
            If Int, the number of folds to be created via KFlods. 
            Otherwise, a cross-validation generator.
        Returns
        -------
        scores : array, shape (n_splits,)
            Scores for each cross-validation split.
        """

        assert X.shape[0] == y.shape[0]
        if torch.is_tensor(X):
            X = X.numpy()
        if torch.is_tensor(y):
            y = y.numpy()

        good_columns = ~np.isnan(y).any(axis=0) # boolean array
        good_columns = np.logical_and(good_columns,~(y.std(axis=0)==0).astype(bool))
        y_preprocessed = y[:,good_columns]

        if isinstance(cv, int):
            cv = sklearn.model_selection.KFold(cv,shuffle=False)

        scores = np.zeros((cv.n_splits,y.shape[1]))
       
        for i,(train, test) in enumerate(cv.split(X)):
            X_train,y_train = X[train,:],y_preprocessed[train,:]
            X_test,y_test = X[test,:],y_preprocessed[test,:]

            self.fit(X_train,y_train)
            #self.fit(X[train,:], y[train,:])
            preds = self.predict(X_test)
            s = correlate_columns(preds, y_test)
            scores[i,good_columns] = s
        
        return scores

    def cv_prediction_scores(self, X, y, cv=sklearn.model_selection.KFold(10)):
        """Compute the prediction score for each fold of the cross-validation
        generator.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples, n_targets) n_targets can be 1
            Target values.
        cv : Int or cross-validation generator
            If Int, the number of folds to be created via KFlods. 
            Otherwise, a cross-validation generator.
        Returns
        -------
        scores : array, shape (n_splits,)
            Scores for each cross-validation split.
        """

        assert X.shape[0] == y.shape[0]
        if torch.is_tensor(X):
            X = X.numpy()
        if torch.is_tensor(y):
            y = y.numpy()

        good_columns = ~np.isnan(y).any(axis=0) # boolean array
        good_columns = np.logical_and(good_columns,~(y.std(axis=0)==0).astype(bool))
        y_preprocessed = y[:,good_columns]

        if isinstance(cv, int):
            cv = sklearn.model_selection.KFold(cv,shuffle=False)

        scores = np.zeros((cv.n_splits,y.shape[1]))
       
        for i,(train, test) in enumerate(cv.split(X)):
            X_train,y_train = X[train,:],y_preprocessed[train,:]
            X_test,y_test = X[test,:],y_preprocessed[test,:]

            self.fit(X_train,y_train)
            #self.fit(X[train,:], y[train,:])
            preds = self.predict(X_test)
            s = correlate_columns(preds, y_test)
            scores[i,good_columns] = s
        
        return scores

if __name__ == "__main__":
    from sklearn.datasets import make_regression


    np.random.seed(0)


    ########### Generate a regression dataset ###########

    # Number of samples and features
    n_samples, n_features = 100, 3
    X, y,coef = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1,coef=True)
    y = y.reshape(-1, 1)  # make y 2D array


    ########### Ridge regression ###########

    # Ridge alphas to be tested
    alphas = np.logspace(-2, 2, 5)

    # Initialize the model
    model = RidgeDoubleCV(alphas=alphas, alpha_per_target=True)

    # Compute the prediction scores for each fold
    scores = model.cv_prediction_scores(X, y, cv=5)

    #get prediction for each fold
    preds = model.cv_predict(X,y,cv=10)

    # Print the mean and std of the prediction scores
    print("Prediction scores: mean = {:.3f}, std = {:.3f}".format(scores.mean(), scores.std()))
