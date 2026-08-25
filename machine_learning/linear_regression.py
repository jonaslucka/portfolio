"""
Linear Regression implementation

This module provides an implementation of the Linear Regression algorithm
from scratch only using NumPy.

Author: Jonas Lucka
Date: 2026
"""

import numpy as np

class LinearRegression:
    """
    Represents a linear regresion model and allows for fitting and prediction. 
    This class and class methods are based on Chapter 3.4 in: 
    
        Dive into Deep Learning from Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola 

    The model predicts: y_hat = x @ w + b 

    where w is the vector of feature weights and b is the bias. 

    
    Parameters
    ---------- 
    None 
    

    Attributes 
    ---------- 
    params : ndarray or None
        Parameter vector containing the feature weights followed by the bias parameter.
        Has the shape: n_features + 1 after fitting. 

    n_features : int or None
        Number of features in the input data. Set during fitting. 

    loss_history : ndarray or list or None 
        History of the loss values recorded by the optimizer during training.
    """
    def __init__(self):
        self.params = None
        self.n_features = None
        self.loss_history = None

    def predict(self, x):
        """
        Predict target values for input samples.
    
        Parameters
        ----------
        x : arraylike
            Input matrix of shape (n_samples, n_features).
            
        Returns
        -------
        ndarray
            Predicted values.
        
        Raises
        ------
        ValueError
            If the Model is not fitted before the prediction.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before the prediction.")

        x = np.asarray(x)

        w = self.params[:-1]
        b = self.params[-1]

        return x @ w + b


    def loss(self, x, y):
        """
        Compute the loss as the mean squared error.
        
        Parameters
        ----------
        x : array_like
            Input feature matrix
        y : array_like
            Target values
        
        Returns
        -------
        float
            Mean squared error
        """
        y = np.asarray(y)

        predictions = self.predict(x)

        return np.mean((predictions - y) ** 2)
