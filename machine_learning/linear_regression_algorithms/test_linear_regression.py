"""
Tests for the Linear Regression implementation from scratch.

This module provides pytest tests for the implementation of the
class LinearRegression in linear_regression.py.

Author: Jonas Lucka
Date: 2026
"""

import numpy as np
import pytest

from machine_learning.linear_regression_algorithms import LinearRegression
from mathematical_optimization import optimize_minibatch_sgd


class TestLinearRegression:
    """Tests for LinearRegression."""

    def test_loss(self):
        """
        The method loss should find the loss of a linear regression model.
        
        The model is defined as y_hat = 2x + 1.
        The input data is x = (1, 2, 3).
        The target values are y = (3, 5, 8).
        The Error is: (0, 0, -1).
        The Mean squared Error should be: 1/3.
        """
        model = LinearRegression()

        params = np.array([2.0, 1.0])

        x = np.array([
            [1.0],
            [2.0],
            [3.0]
        ])

        y = np.array([
            3.0,
            5.0,
            8.0
        ])

        expected = 1.0 / 3.0

        result = model.loss(params, x, y)

        assert np.allclose(result, expected, atol=1e-5)

    def test_gradient(self):
        """
        The method gradient should find the gradietn of a linear regression model.

        The model is defined as y_hat = 2x + 1. 
        The input data is x = (1, 2, 3). 
        The target values are y = (3, 5, 10). 
        The actual gradient is: (-6, -2)
        """
        model = LinearRegression()

        params = np.array([2.0, 1.0])

        x = np.array([
            [1.0],
            [2.0],
            [3.0]
        ])

        y = np.array([
            3.0,
            5.0,
            10.0
        ])

        expected_gradient = np.array([-6.0, -2.0])

        gradient = model.gradient(params, x, y)

        assert np.allclose(expected_gradient, gradient, atol=1e-5)

    def test_fit_learns_simple_linear_relationship(self):
        """
        The method fit should find the right parameter for a simple linear relationship.

        The training data follows the relationship y = 3x + 2.
        The weight is w = 3
        The bias is b = 2.
        """
        model = LinearRegression()

        x = np.arange(1, 11, dtype=float).reshape(-1, 1)
        y = 3.0 * x[:, 0] + 2.0

        model.fit(
            x,
            y,
            optimizer=optimize_minibatch_sgd,
            learning_rate=0.01,
            epochs=2000,
            batch_size=4
        )

        assert np.allclose(model.params, np.array([3.0, 2.0]), atol=1e-5)

    def test_fit_multiple_features(self):
        """
        The method fit should find the rigth parameters for a regression with multiple features.

        The training data follows the relationship y = 2*x1 - 3*x2 + 5.
        The weights are w = (2, -3)
        The bias is b = 5.
        """
        model = LinearRegression()

        x = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 2.0],
            [2.0, 5.0],
            [6.0, 1.0],
            [1.0, 6.0]
        ])

        y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 5.0

        model.fit(
            x,
            y,
            optimizer=optimize_minibatch_sgd,
            learning_rate=0.01,
            epochs=3000,
            batch_size=4
        )

        expected_params = np.array([
            2.0,
            -3.0,
            5.0
        ])

        assert np.allclose(model.params, expected_params, atol=1e-2)

    def test_fit_stores_loss_history(self):
        """
        The method fit should store the optimizer's loss history.
        """
        model = LinearRegression()

        x = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0]
        ])

        y = np.array([
            3.0,
            5.0,
            7.0,
            9.0
        ])

        model.fit(
            x,
            y,
            optimizer=optimize_minibatch_sgd,
            learning_rate=0.01,
            epochs=10,
            batch_size=2
        )

        assert model.loss_history is not None
        assert len(model.loss_history) > 0
        assert np.isfinite(model.loss_history).all()

    def test_fit_invalid_x_dimension(self):
        """
        The method fit should reject one dimensional x.
        """
        model = LinearRegression()

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        with pytest.raises(
            ValueError,
            match="x must be a 2-dimensional array",
        ):
            model.fit(
                x,
                y,
                optimizer=optimize_minibatch_sgd,
            )

    def test_fit_mismatched_sample_count(self):
        """
        The method fit should reject x and y with different sample counts.
        """
        model = LinearRegression()

        x = np.array([
            [1.0],
            [2.0],
            [3.0]
        ])

        y = np.array([
            2.0,
            4.0
        ])

        with pytest.raises(
            ValueError,
            match="same number of samples",
        ):
            model.fit(
                x,
                y,
                optimizer=optimize_minibatch_sgd
            )

    def test_fit_flattens_y(self):
        """
        The method fit should accept a column vector for y.
        """
        model = LinearRegression()

        x = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0]
        ])

        y = np.array([
            [3.0],
            [5.0],
            [7.0],
            [9.0]
        ])

        model.fit(
            x,
            y,
            optimizer=optimize_minibatch_sgd,
            learning_rate=0.01,
            epochs=1000,
            batch_size=2
        )

        assert model.params.shape == (2,)
        assert np.allclose(model.params, np.array([2.0, 1.0]), atol=1e-2)

    def test_predict_before_fit_raises_error(self):
        """Prediction should fail if the model has not been fitted."""
        model = LinearRegression()

        x = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(x)

    def test_predict(self):
        """
        The method prediction should have the rigth predictions.
        
        The model is defined as y_hat = 2*x1 + 3*x2 + 5.
        The expected predictions are (10, 12, 13)
        """
        model = LinearRegression()

        model.params = np.array([2.0, 3.0, 5.0])

        x = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0]
        ])

        expected = np.array([
            10.0,
            12.0,
            13.0
        ])

        predictions = model.predict(x)

        np.testing.assert_allclose(predictions, expected)
