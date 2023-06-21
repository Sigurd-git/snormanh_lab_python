from sklearn.model_selection import GridSearchCV, KFold, check_cv
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import GroupKFold
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
class NestedCV:
    """
    Class to perform nested cross-validation. GridSearchCV is used to optimize hyperparameters 
    in the inner loop, and the outer loop is used for model validation.
    
    Example usage:
    model = Ridge()
    params = {'alpha': [0.1, 1, 10, 100]}
    nested_cv = NestedCV(model, params, outer_cv=5, inner_cv=5, scoring='accuracy', random_state=42)
    nested_cv.fit_predict(X, y)
    """
    def __init__(self, model, params, outer_cv=5, inner_cv=5, scoring='accuracy', random_state=42,groups=None):
        # Initialize the class with model, parameter grid, number of folds and scoring method
        self.model = model
        self.params = params
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
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

            # Same for inner_cv
            if isinstance(self.inner_cv, int):
                inner_cv = KFold(n_splits=self.inner_cv, shuffle=True, random_state=self.random_state)
            else:
                inner_cv = check_cv(self.inner_cv)

            #linear regression do not need hyperparameter optimization
            #empty dict
            if self.params == {}:
                # Fit the model on the training set
                self.model.fit(X_train, y_train)

                # Predict on the test set
                y_pred = self.model.predict(X_test)

                # Store true and predicted values
                outer_true.extend(y_test)
                outer_pred.extend(y_pred)

                self.coefs_.append(self.model.coef_)
                self.intercept_.append(self.model.intercept_)
                self.inner_scores_.append(self.model.score(X_test,y_test))

                # Store test indices
                self.test_indices_.extend(test_index)
            else:
                # Apply GridSearchCV for hyperparameter optimization
                grid_search = GridSearchCV(self.model, self.params, cv=inner_cv, scoring=self.scoring)

                if self.groups is None:
                    grid_search.fit(X_train, y_train)
                else:
                    grid_search.fit(X_train, y_train,groups=self.groups[train_index])

                # Store the best score obtained in the inner loop
                self.inner_scores_.append(grid_search.best_score_)

                # Predict on the test set
                y_pred = grid_search.predict(X_test)

                # Store true and predicted values
                outer_true.extend(y_test)
                outer_pred.extend(y_pred)

                self.coefs_.append(grid_search.best_estimator_.coef_)
                self.intercept_.append(grid_search.best_estimator_.intercept_)

                # Store test indices
                self.test_indices_.extend(test_index)

        # Store the outer loop's true and predicted values
        self.outer_true = np.array(outer_true)
        self.outer_pred = np.array(outer_pred)

    # Compute the score
    def get_score(self,scoring=correlate_columns):
        """
        Returns the score between true and predicted values using the specified scoring function.
        """
        return scoring(self.outer_true, self.outer_pred)

    # Get predictions
    def get_predictions(self):
        """
        Returns the true and predicted values.
        """
        return self.outer_true, self.outer_pred

    # Get inner CV scores
    def get_inner_cv_scores(self):
        """
        Returns the best score obtained in the inner loop of cross-validation for each fold of the outer loop.
        """
        return self.inner_scores_

    # Restore the order of predictions to match the original data order
    def restore_order(self):
        """
        Returns the predicted values in the original order of the data.
        """
        original_order_pred = np.empty_like(self.outer_pred)
        original_order_pred[self.test_indices_] = self.outer_pred
        return original_order_pred
    
    def get_parameters(self,bias=True):
        """
        bias: bool, default=True
        Returns the parameters that gave the best results in the inner loop.
        """
        if bias:
            return self.intercept_, self.coefs_
        else:
            return self.coefs_


# class SectionedKFold:
#     def __init__(self, n_splits, section_lengths):
#         """
#         Initialize a KFold object that can split data according to sections.

#         n_splits: int, The number of folds in the KFold cross-validation.
#         section_lengths: list of int, The length of each section in the data.

#         Example:
#         cv = SectionedKFold(5, [100, 150, 200])
#         """

#         self.n_splits = n_splits
#         self.section_lengths = section_lengths
#         self.kfold = KFold(n_splits)

#     def split(self, X,y,groups=None):
#         """
#         Generate indices to split data into training and test set.

#         X: array-like of shape (n_samples, n_features), Training data, where n_samples is the 
#            number of samples and n_features is the number of features.

#         y: array-like of shape (n_samples,n_targets), The target variable for supervised learning problems.

#         groups
        
#         Yields
#         -------
#         train : ndarray
#             The training set indices for that split.
#         test : ndarray
#             The testing set indices for that split.
        
#         Example:
#         cv = SectionedKFold(5, [100, 150, 200])
#         for train_index, test_index in cv.split(X):
#             print("TRAIN:", train_index, "TEST:", test_index)
#         """

#         assert len(X) == len(y)== sum(self.section_lengths), "The length of X and y must be equal to the sum of section lengths."

#         boundaries = np.cumsum([0] + self.section_lengths)
#         for test_fold_indices in self.kfold.split(range(self.n_splits)):
#             test_indices = []
#             train_indices = []
#             for i in range(len(boundaries) - 1):
#                 start, end = boundaries[i], boundaries[i + 1]
#                 section_indices = np.arange(start, end)
#                 section_size = len(section_indices)
#                 section_test_indices = section_indices[int(test_fold_indices[1]) * section_size // self.n_splits : (int(test_fold_indices[1]) + 1) * section_size // self.n_splits]
#                 section_train_indices = np.setdiff1d(section_indices, section_test_indices)
#                 test_indices.extend(section_test_indices)
#                 train_indices.extend(section_train_indices)
#             yield train_indices, test_indices

#     def get_n_splits(self, X, y=None, groups=None):
#         """
#         Returns the number of splitting iterations in the cross-validator

#         X : array-like of shape (n_samples, n_features)
#             Training data, where n_samples is the number of samples
#             and n_features is the number of features.
        
#         Returns
#         -------
#         n_splits : int
#             Returns the number of splitting iterations in the cross-validator.
        
#         Example:
#         cv = SectionedKFold(5, [100, 150, 200])
#         num_splits = cv.get_n_splits(X)
#         """
#         return self.n_splits



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




