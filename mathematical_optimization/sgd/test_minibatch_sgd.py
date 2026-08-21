"""
Tests for mini-batch stochastic gradient descent.

This module provides pytest tests for the implementation of the
mini-batch SGD algorithm in minibatch_sgd_algorithm.py.

Author: Jonas Lucka
Date: 2026
"""

import numpy as np

from mathematical_optimization.sgd.minibatch_sgd_algorithm import optimize_minibatch_sgd

def test_minibatch_sgd_linear_regression():
    """
    Mini-batch SGD should converge to the minimum of a simple
    linear regression problem.

    The optimal parameter is theta = (2, -3)
    """

    rng = np.random.default_rng(42)

    x = rng.normal(size=(200, 2))

    true_params = np.array([2.0, -3.0])

    y = x @ true_params

    def gradient_fn(params, x_batch, y_batch):
        errors = x_batch @ params - y_batch
        return 2.0 * (x_batch.T @ errors) / len(x_batch)

    initial_params = np.zeros(2)

    params = optimize_minibatch_sgd(x, y, initial_params, gradient_fn)

    assert np.allclose(params, true_params, atol=1e-5)


def test_minibatch_sgd_reduces_loss():
    """
    Mini-batch SGD should reduce the loss during optimization.
    """

    x = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])

    y = 2.0 * x[:, 0]

    def loss_fn(params, x, y):
        predictions = x[:, 0] * params[0]

        return np.mean((predictions - y) ** 2)

    def gradient_fn(params, x_batch, y_batch):
        predictions = x_batch[:, 0] * params[0]
        errors = predictions - y_batch

        return np.array([
            2.0 * np.mean(x_batch[:, 0] * errors)
        ])

    initial_params = np.array([0.0])

    initial_loss = loss_fn(np.array([0.0]), x, y)

    _, loss_history = optimize_minibatch_sgd(
        x,
        y,
        initial_params,
        gradient_fn,
        loss_fn=loss_fn,
        batch_size=2
        )

    assert loss_history[-1] <= initial_loss

def test_minibatch_sgd_returns_loss_history():
    """
    If a loss function is provided, SGD should return the loss
    history containing one value per epoch.
    """

    x = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = 2.0 * x[:, 0]

    def loss_fn(params, x, y):
        predictions = x[:, 0] * params[0]

        return np.mean((predictions - y) ** 2)

    def gradient_fn(params, x_batch, y_batch):
        predictions = x_batch[:, 0] * params[0]
        errors = predictions - y_batch

        return np.array([
            2.0 * np.mean(x_batch[:, 0] * errors)
        ])

    initial_params = np.array([0.0])

    params, loss_history = optimize_minibatch_sgd(
        x,
        y,
        initial_params,
        gradient_fn,
        loss_fn=loss_fn,
        epochs=10,
        batch_size=2
        )

    assert isinstance(loss_history, list)
    assert len(loss_history) == 10
    assert np.isfinite(loss_history).all()
    assert np.isclose(loss_history[-1], loss_fn(params, x, y))

def test_minibatch_sgd_batch_size_one():
    """
    A batch size of one should perform stochastic gradient descent.

    The optimal parameter is theta = 2
    """

    x = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = 2.0 * x[:, 0]

    def gradient_fn(params, x_batch, y_batch):
        predictions = x_batch[:, 0] * params[0]
        errors = predictions - y_batch

        return np.array([
            2.0 * np.mean(x_batch[:, 0] * errors)
        ])

    initial_params = np.array([0.0])

    params = optimize_minibatch_sgd(x, y, initial_params, gradient_fn, batch_size=1)

    assert np.allclose(params, np.array([2.0]), atol=1e-5)

def test_minibatch_sgd_full_batch():
    """
    A batch size equal to the number of samples should perform
    full-batch gradient descent.

    The optimal parameter is theta = 2
    """

    x = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = 2.0 * x[:, 0]

    def gradient_fn(params, x_batch, y_batch):
        predictions = x_batch[:, 0] * params[0]
        errors = predictions - y_batch

        return np.array([
            2.0 * np.mean(x_batch[:, 0] * errors)
        ])

    initial_params = np.array([0.0])

    params = optimize_minibatch_sgd(x, y, initial_params, gradient_fn, batch_size=len(x))

    assert np.allclose(params, np.array([2.0]), atol=1e-5)

def test_minibatch_sgd_incomplete_final_batch():
    """
    SGD should correctly handle a final mini-batch containing fewer
    samples than batch_size.

    The optimal parameter is theta = 2
    """

    x = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])

    y = 2.0 * x[:, 0]

    def gradient_fn(params, x_batch, y_batch):
        predictions = x_batch[:, 0] * params[0]
        errors = predictions - y_batch

        return np.array([
            2.0 * np.mean(x_batch[:, 0] * errors)
        ])

    initial_params = np.array([0.0])

    params = optimize_minibatch_sgd(x, y, initial_params, gradient_fn, batch_size=2)

    assert np.allclose(params, np.array([2.0]), atol=1e-5)
