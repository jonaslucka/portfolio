"""
Minibatch Stochastic Gradient Descent implementation

This module provides an implementation of the Minibatch Stochastic Gradient Descent algorithm 
that can be used both for general numerical optimization and for machine learning.

Author: Jonas Lucka
Date: 2026
"""


import numpy as np


def optimize_minibatch_sgd(
        x,
        y,
        initial_params,
        gradient_fn,
        loss_fn=None,
        learning_rate=0.01,
        epochs=100,
        batch_size=32
    ):
    """
    Minibatch stochastic gradient descent method based on:
    https://sebastianraschka.com/faq/docs/sgd-methods.html algorithm 5)


    Parameters
    ----------
    x : array_like
        Training samples.

    y : array_like
        Target values corresponding to x.

    initial_params : array_like
        Initial parameter vector.

    gradient_fn : callable
        Gradient of the objective function (gradient of the loss function)

    loss_fn : callable, optional
        Objective loss function.
        If given, return the whole loss history
    
    learning_rate : float, optional
        Step size used for each parameter update.

    epochs : int, optional
        Number of complete passes through the training dataset.

    batch_size : int, optional
        Number of samples used to compute each gradient update.
        If batch_size=1, the method corresponds to stochastic
        gradient descent. 
        If batch_size is equal to the number of
        training samples, it corresponds to full-batch gradient descent.


    Returns
    -------
    params : ndarray
        Parameter vector after the final update.

    loss_history : list of float, optional
        Loss evaluated on the full training dataset after each epoch.
        Returned only when `loss_fn` is provided.


    Raises
    ------

    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if epochs < 0:
        raise ValueError("epochs must be non-negative.")
    loss_history = []

    x = np.asarray(x)
    y = np.asarray(y)

    # Initialize w and b here it is params
    params = np.asarray(initial_params, dtype=float).copy()

    n_samples = x.shape[0]

    for _ in range(epochs):

        # Shuffle dataset
        indices = np.random.permutation(n_samples)

        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Iterate through mini-batches
        for start in range(0, n_samples, batch_size):

            end = start + batch_size

            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Calculate gradient
            gradient = gradient_fn(params, x_batch, y_batch)

            # Update parameters
            params -= learning_rate * gradient

        if loss_fn is not None:
            loss = loss_fn(params, x, y)
            loss_history.append(loss)

    if loss_fn is not None:
        return params, loss_history
    return params
