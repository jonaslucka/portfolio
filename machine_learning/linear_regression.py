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
        Parameter vector containing the feature weights followed by the bias.
        Has the shape: n_features + 1 after fitting. 

    n_features : int or None
        Number of features of the input data. 

    loss_history : ndarray or list or None 
        History of the loss values recorded by the optimizer during training.
    """
    def __init__(self):
        self.params = None
        self.n_features = None
        self.loss_history = None

    def loss(self, params, x, y):
        """
        Compute the mean squared error for an arbitrary set of model parameters.
        This method is used for fitting the model paramters.

        Parameters
        ----------
        params : ndarray
            Parameter vector containing the feature weights followed by the bias

        x : array_like
            Input data in form of a matrix
        
        y : array_like
            Target values

        Returns
        -------
        float
            Mean squared error
        """
        w = params[:-1]
        b = params[-1]

        predictions = x @ w + b

        return np.mean((predictions - y) ** 2)

    def gradient(self, params, x, y):
        """
        Compute the gradient of the mean squared error
        with the respect to the model parameters

        Parameters
        ----------
        params : ndarray
            Parameter vector containing the feature weights followed by the bias
        
        x : array_like
            Input data in form of a matrix
       
        y : array_like
            Target values

        Returns
        -------
        ndarray
            Gradient vector.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        w = params[:-1]
        b = params[-1]

        predictions = x @ w + b
        errors = predictions - y

        grad_w = (2 / len(x)) * (x.T @ errors)
        grad_b = 2 * np.mean(errors)

        return np.concatenate([grad_w, [grad_b]])

    def fit(
        self,
        x,
        y,
        optimizer,
        learning_rate=0.01,
        epochs=100,
        batch_size=32
    ):
        """
        For a given optimizer it fits the linear regression model to the given data.

        Parameter
        ---------
        x : array_like
            Input data in form of a matrix
       
        y : array_like
            Target values

        optimizer : callable
            Optimization function used to update the model parameters.

            The optimizer must accept the input data, initial parameters,
            gradient, loss function, learning rate, number of
            epochs, and batch size.

        learning_rate : float, default=0.01
            Step size used by the optimizer when updating the parameters.

        epochs : int, default=100
            Number of times the optimizer iterates over the training data.

        batch_size : int, default=32
            Number of samples used in each minibatch during optimization.

        Returns
        -------
        LinearRegression
            The fitted model.

        Raises
        ------
        ValueError
            If the dimension of x and y are not correct
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim !=2:
            raise ValueError("x must be a 2-dimensional array.")
        if y.ndim !=1:
            y = y.reshape(-1)
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of samples")

        self.n_features = x.shape[1]

        initial_params = np.zeros(self.n_features + 1)

        result = optimizer(
            x=x,
            y=y,
            initial_params=initial_params,
            gradient_fn=self.gradient,
            loss_fn=self.loss,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        self.params, self.loss_history = result
        return self

    def predict(self, x):
        """
        Predict target values for input samples.

        Parameters
        ----------
        x : array_like
            Input data in form of a matrix

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
