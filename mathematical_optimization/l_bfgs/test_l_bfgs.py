"""
Tests for L-BFGS

This module provides pytest for the implementation of the L-BFGS algorithm in l_bfgs_algorithm.py.

Author: Jonas Lucka
Date: 2026
"""


import numpy as np

from mathematical_optimization.l_bfgs.l_bfgs_algorithm import (optimize_lbfgs, two_loop_recursion)

def test_lbfgs_quadratic():
    """
    L-BFGS should find the minimum of a simple quadratic function.

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

    x = optimize_lbfgs(f, grad_f, x0)

    assert np.allclose(x, np.array([1.0, -2.0]), atol=1e-5)


def test_lbfgs_rosenbrock():
    """
    L-BFGS should converge to the minimum of the
    Rosenbrock function f.

    The initial point is x_0 = (-1.2, 1)

    The minimum is at x = (1, 1).

    The chosen initial point and line-search configuration
    make a larger alpha_max useful for this test
    """

    def f(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_f(x):
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])

    x0 = np.array([-1.2, 1.0])

    x = optimize_lbfgs(f, grad_f, x0, alpha_max=250)

    assert np.allclose(x, np.array([1.0, 1.0]), atol=1e-5)


def test_lbfgs_reduces_function_value():
    """
    L-BFGS should return a point with a lower or equal
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

    x = optimize_lbfgs(f, grad_f, x0)

    assert f(x) <= f(x0)


def test_lbfgs_gradient_is_small_at_solution():
    """
    The gradient should be approximately zero at the solution.

    f(x) = (x1 - 5)^2 + (x2 - 4)^2.
    
    The initial point is x_0 = (0,0).

    The minimum is at x = (5, 4).
    """

    def f(x):
        return (x[0] - 5)**2 + (x[1] - 4)**2

    def grad_f(x):
        return np.array([
            2 * (x[0] - 5),
            2 * (x[1] - 4)
        ])

    x0 = np.array([0.0, 0.0])

    x = optimize_lbfgs(f, grad_f, x0)

    assert np.linalg.norm(grad_f(x)) < 1e-5


def test_two_loop_recursion_empty_history():
    """
    With no correction pairs, the initial inverse Hessian
    approximation is the identity matrix.
    """

    grad = np.array([1.0, 2.0, 3.0])

    result = two_loop_recursion(grad, [], [])

    assert np.allclose(result, grad)

def test_lbfgs_with_one_memory_pair():
    """
    L-BFGS should converge when only one correction pair is stored.

    f(x) = (x1 - 1)^2 + 2*(x2 + 2)^2

    The initial point is x_0 = (5, 5).
    
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

    x  = optimize_lbfgs(f, grad_f, x0, m=1)

    assert np.allclose(x, np.array([1.0, -2.0]), atol=1e-5)

def test_lbfgs_high_dimensional_quadratic():
    """
    L-BFGS should solve a higher-dimensional quadratic problem.

    The objective function is

        f(x) = 1/2 * sum_{i=1}^n i * (x_i - 1)^2,

    where n = 100.

    The initial point is x_0 = (0, ..., 0).

    The minimum is at x = (1, ..., 1).
    """

    n = 100

    diagonal = np.arange(1.0, n + 1.0)
    x_star = np.ones(n)

    def f(x):
        return 0.5 * np.sum(diagonal * (x - x_star)**2)

    def grad_f(x):
        return diagonal * (x - x_star)

    x0 = np.zeros(n)

    x = optimize_lbfgs(f, grad_f, x0, m=10)

    assert np.allclose(x, x_star, atol=1e-5)

def test_lbfgs_return_history():
    """
    L-BFGS should optionally return the optimization history.

    f(x) = (x1 - 1)^2 + (x2 + 2)^2

    The initial point is x_0 = (5, 5).

    The minimum is at x = (1, -2).
    """

    def f(x):
        return (x[0] - 1)**2 + (x[1] + 2)**2

    def grad_f(x):
        return np.array([
            2 * (x[0] - 1),
            2 * (x[1] + 2)
        ])

    x0 = np.array([5.0, 5.0])

    x, history = optimize_lbfgs(f, grad_f, x0, return_history=True)

    assert np.allclose(x, np.array([1.0, -2.0]), atol=1e-5)
    assert isinstance(history, list)
    assert np.allclose(history[0], x0)
    assert np.allclose(history[-1], x)
