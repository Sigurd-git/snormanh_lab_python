from sklearn.model_selection import GridSearchCV, KFold, check_cv
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score,cross_val_predict, cross_validate
import numpy as np
from general_analysis_code_python.subdivide import subdivide
# Define a function to calculate the correlation between the columns of two matrices
def correlate_columns(x, y):
    """
    Function to calculate the Pearson correlation between the columns of two matrices.
    
    Inputs:
    x: numpy array of shape (n_samples, n_features_x)
    y: numpy array of shape (n_samples, n_features_y)
    
    Returns:
    A 1D numpy array of shape (n_features,) where each element is the correlation 
    between the corresponding columns of x and y.
    
    Example usage:
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[1, 2], [2, 3], [3, 4]])
    correlate_columns(x, y)
    """
    # Calculate centered values
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    
    # Calculate sum of squares
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    
    # Calculate correlation
    result = np.matmul(xv.T, yv) / np.sqrt(np.outer(xvss, yvss))
    
    # Bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, np.array(1.0)), np.array(-1.0)).diagonal()

# Make a scorer using our correlation function
correlation_scorer = make_scorer(lambda x,y: correlate_columns(x,y))

# Define a class to perform nested cross-validation
class Grouped_NestedCV:
    """
    Class to perform nested cross-validation. GridSearchCV is used to optimize hyperparameters 
    in the inner loop, and the outer loop is used for model validation.
    
    Example usage:
    model = Ridge()
    params = {'alpha': [0.1, 1, 10, 100]}
    nested_cv = NestedCV(model, params, outer_cv=5, inner_cv=5, scoring='accuracy', random_state=42)
    nested_cv.fit_predict(X, y)
    """
    def __init__(self, model, params, scoring='r2', random_state=42,groups=None):
        # Initialize the class with model, parameter grid, number of folds and scoring method
        self.model = model
        self.params = params
        self.scoring = scoring
        self.random_state = random_state
        self.inner_scores_ = []
        self.test_indices_ = []
        self.coefs_ = []
        self.intercept_ = []
        self.groups = groups

        if self.groups is not None:
            #outer_cv and inner_cv should be a GroupKFold object
            assert isinstance(self.outer_cv, GroupKFold), "outer_cv should be a GroupKFold object"
            assert isinstance(self.inner_cv, GroupKFold), "inner_cv should be a GroupKFold object"

    # Fit the model and predict
    def fit_predict(self, X, y):
        # If outer_cv is an integer, create a KFold object, otherwise use it as it is
        if isinstance(self.outer_cv, int):
            outer_cv = KFold(n_splits=self.outer_cv, shuffle=True, random_state=self.random_state)
        else:
            outer_cv = check_cv(self.outer_cv)

        # Prepare empty lists to hold true and predicted values
        outer_true = []
        outer_pred = []

        # Outer loop for cross-validation
        for train_index, test_index in outer_cv.split(X, y, self.groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_groups = self.groups[train_index]
            # Same for inner_cv
            if isinstance(self.inner_cv, int):
                inner_cv = KFold(n_splits=self.inner_cv, shuffle=True, random_state=self.random_state)
            else:
                inner_cv = check_cv(self.inner_cv)

            # obtain individual predictions
            cv_preds = cross_val_predict(model, X,y, cv=custom_cv_iterator(X,y))
            # or, assess average performance over all columns in a fold
            cv_scores = cross_val_score(model, X,y, cv=custom_cv_iterator(X,y), scoring=correlation_scorer)






if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    import numpy as np
    np.random.seed(0)


    ########### Generate a regression dataset ###########

    # Number of samples and features
    n_samples, n_features = 100, 3
    X, y,coef = make_regression(n_samples=n_samples, n_features=n_features, noise=100,coef=True)
    y = y.reshape(-1, 1)  # make y 2D array


    ########### Ridge regression ###########

    # set parameters for ridge regression
    ridge_params = {'alpha': [0.1, 1, 10, 100]}
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # use pearson correlation as scoring
    nested_cv = NestedCV(Ridge(), ridge_params, outer_cv=outer_cv, inner_cv=inner_cv, scoring='r2')
    nested_cv.fit_predict(X, y)

    # get the best score
    print(nested_cv.get_inner_cv_scores())

    # get the true values and predictions 
    y_true, y_pred = nested_cv.get_predictions()

    y_pred_ordered = nested_cv.restore_order()

    # get the correlation between true values and predictions
    print(correlate_columns(y, y_pred_ordered))

    #get coefficient
    coefs = nested_cv.get_parameters(bias=False)

    #get mean coefficient
    coef_pred = np.mean(coefs,axis=0).reshape(-1,1)

    #compare with true coefficient
    print(correlate_columns(coef,coef_pred))


    ########### Linear regression using GroupKfold###########
    from sklearn.linear_model import LinearRegression

    # set parameters for linear regression

    # Number of samples and features
    n_samples, n_features = 100, 3
    X, y,coef = make_regression(n_samples=n_samples, n_features=n_features, noise=100,coef=True)
    y = y.reshape(-1, 1)  # make y 2D array

    groups1 = subdivide(int(n_samples/2),5)
    groups2 = subdivide(int(n_samples/2),5)

    groups = np.concatenate((groups1,groups2))

    linear_params = {}
    outer_cv = GroupKFold(n_splits=5)
    inner_cv = GroupKFold(n_splits=4)

    # use pearson correlation as scoring
    nested_cv = NestedCV(LinearRegression(), linear_params, outer_cv=outer_cv, inner_cv=inner_cv, scoring='r2',groups=groups)
    nested_cv.fit_predict(X, y)

    # get the best scores
    print(nested_cv.get_inner_cv_scores())

    # get the true values and predictions
    y_true, y_pred = nested_cv.get_predictions()

    y_pred_ordered = nested_cv.restore_order()


    # get the correlation between true values and predictions
    print(correlate_columns(y, y_pred_ordered))

    #get coefficient
    coefs = nested_cv.get_parameters(bias=False)

    #get mean coefficient
    coef_pred = np.mean(coefs,axis=0).reshape(-1,1)

    #compare with true coefficient
    print(correlate_columns(coef,coef_pred))




