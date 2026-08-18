"""
Tests for BFGS

This module provides pytest for the implementation of the BFGS algorithm in bfgs_algorithm.py.

Author: Jonas Lucka
Date: 2026
"""


import numpy as np

from mathematical_optimization.bfgs import optimize_bfgs

def test_bfgs_quadratic():
    """
    BFGS should find the minimum of a simple quadratic function.

    f(x) = (x1 - 1)^2 + 2(x2 + 2)^2

    The initial point is x_0 = (5, 5)

    The minimum is at x = (1, -2).
    """

    def f(x):
        return (x[0] - 1)**2 + 2 * (x[1] + 2)**2

    def grad_f(x):
        return np.array([
            2 * (x[0] - 1),
            4 * (x[1] + 2)
        ])

    x0 = np.array([5.0, 5.0])

    x = optimize_bfgs(f, grad_f, x0)

    assert np.allclose(x, np.array([1.0, -2.0]), atol=1e-5)


def test_bfgs_rosenbrock():
    """
    BFGS should converge to the minimum of the
    Rosenbrock function f.

    The initial point is x_0 = (-1.2, 1)

    The minimum is at x = (1, 1).
    """

    def f(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_f(x):
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])

    x0 = np.array([-1.2, 1.0])

    x = optimize_bfgs(f, grad_f, x0)

    assert np.allclose(x, np.array([1.0, 1.0]), atol=1e-4)


def test_bfgs_reduces_function_value():
    """
    BFGS should return a point with a lower or equal
    objective value than the initial point.

    f(x) = (x1 - 3)^2 + 3(x2 + 1)^2.

    The initial point is x_0 = (10, 10).
    """

    def f(x):
        return (x[0] - 3)**2 + 3 * (x[1] + 1)**2

    def grad_f(x):
        return np.array([
            2 * (x[0] - 3),
            6 * (x[1] + 1)
        ])

    x0 = np.array([10.0, 10.0])

    x = optimize_bfgs(f, grad_f, x0)

    assert f(x) <= f(x0)


def test_bfgs_gradient_is_small_at_solution():
    """
    The gradient should be approximately zero at the solution.

    f(x) = (x1 - 5)^2 + (x2 - 4)^2.
    
    The initial point is x_0 = (0,0).
    """

    def f(x):
        return (x[0] - 5)**2 + (x[1] - 4)**2

    def grad_f(x):
        return np.array([
            2 * (x[0] - 5),
            2 * (x[1] - 4)
        ])

    x0 = np.array([0.0, 0.0])

    x = optimize_bfgs(f, grad_f, x0)

    assert np.linalg.norm(grad_f(x)) < 1e-5
