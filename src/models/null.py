import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

class NullRegressor:
    """
    Class used as baseline model for regression problem
    ...

    Attributes
    ----------
    y : Numpy Array-like
        Target variable
    pred_value : Float
        Value to be used for prediction
    preds : Numpy Array
        Predicted array

    Methods
    -------
    fit(y)
        Store the input target variable and calculate the predicted value to be used
    predict(y)
        Generate the predictions
    fit_predict(y)
        Perform a fit followed by predict
    """


    def __init__(self):
        self.y = None
        self.pred_value = None
        self.preds = None

    def fit(self, y):
        self.y = y
        self.pred_value = y.mean()

    def predict(self, y):
        self.preds = np.full((len(y), 1), self.pred_value)
        return self.preds

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(self.y)

class NullClassifier:
    """
    Class used as baseline model for classification problem
    ...

    Attributes
    ----------
    y : Numpy Array-like
        Target variable
    pred_value : Any
        Value to be used for prediction (most frequent class)
    preds : Numpy Array
        Predicted array

    Methods
    -------
    fit(y)
        Store the input target variable and calculate the most frequent class
    predict(y)
        Generate the predictions
    fit_predict(y)
        Perform a fit followed by predict
    """

    def __init__(self):
        self.y = None
        self.pred_value = None
        self.preds = None

    def fit(self, y):
        self.y = y
        self.pred_value = y.mode()  # Find the most frequent class

    def predict(self, y):
        self.preds = np.full((len(y), 1), self.pred_value)
        return self.preds

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(self.y)


class SklearnNullRegressor(RegressorMixin, BaseEstimator):
    """
    A scikit-learn compatible NullRegressor, which only predicts the mean of 
    the train target.
    """
    def __init__(self):
        self.pred_value = None
        self.preds = None
    
    def fit(self, X, y):
        self.pred_value = y.mean()
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Perform prediction
        self.preds = np.full((len(X), 1), self.pred_value)
        return self.preds
