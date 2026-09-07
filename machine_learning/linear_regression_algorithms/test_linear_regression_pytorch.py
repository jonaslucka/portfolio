"""
Tests for the PyTorch Linear Regression implementation.

This module provides pytest tests for the implementation of the
class LinearRegression in linear_regression.py.

Author: Jonas Lucka
Date: 2026
"""

import numpy as np
import pytest
import torch

from machine_learning.linear_regression_algorithms import LinearRegressionPytorch


class TestLinearRegression:
    """Tests for LinearRegression."""

    def test_forward(self):
        """
        The method forward should compute the correct predictions.

        The model is defined as y_hat = 2x + 1.
        The input data is x = (1, 2, 3).
        """
        model = LinearRegressionPytorch(num_features=1)

        with torch.no_grad():
            model.linear.weight.fill_(2.0)
            model.linear.bias.fill_(1.0)

        x = torch.tensor([
            [1.0],
            [2.0],
            [3.0]
        ])

        expected = torch.tensor([
            [3.0],
            [5.0],
            [7.0]
        ])

        predictions = model(x)

        assert torch.allclose(predictions, expected)

    def test_fit_learns_linear_relationship(self):
        """
        The method fit should find the right parameter for a simple linear relationship.

        The training data follows the relationship y = 3x + 2.
        The weight is w = 3
        The bias is b = 2.
        """
        model = LinearRegressionPytorch(
            num_features=1,
            learning_rate=0.01
        )

        x = torch.arange(1.0, 11.0).reshape(-1, 1)
        y = 3.0 * x + 2.0

        model.fit(
            x,
            y,
            epochs=2000,
            batch_size=4
        )

        weights = model.linear.weight.detach()
        bias = model.linear.bias.detach()

        assert torch.allclose(
            weights,
            torch.tensor([[3.0]]),
            atol=1e-5
        )

        assert torch.allclose(
            bias,
            torch.tensor([2.0]),
            atol=1e-5
        )

    def test_fit_multiple_features(self):
        """
        The method fit should find the rigth parameters for a regression with multiple features.

        The training data follows the relationship y = 2*x1 - 3*x2 + 5.
        The weights are w = (2, -3)
        The bias is b = 5.
        """
        model = LinearRegressionPytorch(
            num_features=2,
            learning_rate=0.01
        )

        x = torch.tensor([
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 2.0],
            [2.0, 5.0],
            [6.0, 1.0],
            [1.0, 6.0]
        ])

        y = 2.0 * x[:, 0:1] - 3.0 * x[:, 1:2] + 5.0

        model.fit(
            x,
            y,
            epochs=3000,
            batch_size=4
        )

        weights = model.linear.weight.detach()
        bias = model.linear.bias.detach()

        expected_weights = torch.tensor([[2.0, -3.0]])
        expected_bias = torch.tensor([5.0])

        assert torch.allclose(
            weights,
            expected_weights,
            atol=1e-2
        )

        assert torch.allclose(
            bias,
            expected_bias,
            atol=1e-2
        )

    def test_fit_stores_loss_history(self):
        """
        The method fit should store the optimizer's loss history.
        """
        model = LinearRegressionPytorch(num_features=1)

        x = torch.tensor([
            [1.0],
            [2.0],
            [3.0],
            [4.0]
        ])

        y = 2.0 * x + 1.0

        model.fit(
            x,
            y,
            epochs=10,
            batch_size=2
        )

        assert len(model.loss_history) == 10
        assert np.isfinite(model.loss_history).all()

    def test_fit_invalid_x_dimension(self):
        """
        The method fit should reject one dimensional x.
        """
        model = LinearRegressionPytorch(num_features=1)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([2.0, 4.0, 6.0])

        with pytest.raises(
            ValueError,
            match="x must be a 2-dimensional tensor"
        ):
            model.fit(x, y)
