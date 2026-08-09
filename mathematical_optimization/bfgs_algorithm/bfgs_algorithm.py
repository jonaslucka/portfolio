"""
BFGS implementation

This module provides an implementation of the gradient descent algorithm.

Author: Jonas Lucka
Date: 2026

"""

import numpy as np

def BFGS(f, grad_f, x0, H0=None, tol=1e-5, max_it=1000):
    """
    BFGS method which is based on Nocedal and Wright: Numerical Optimization

    Parameters
    ----------
    f : callable
        Objective function.

    grad_f : callable
        Gradient of f.

    x0 : array_like
        Starting point.

    H0 : ndarray, optional
        Initial inverse Hessian approximation.
        If None, the identity matrix is used.

    tol : float, optional
        Convergence tolerance.

    max_iter : int, optional
        Maximum number of iterations.


    Returns
    -------
    x : ndarray
        Approximate minimizer.
    """

    x = x0
    grad = grad_f(x)
    k = 0

    if H0 is None:
        H = np.eye(len(x))
    else:
        H = H0

    while np.linalg.norm(grad) > tol and k < max_it:
        p = -H @ grad
        alpha = wolfe_line_search(f, grad_f, x, p)
        x_new = x + alpha * p
        grad_new = grad_f(x_new)
        s = x_new - x
        y = grad_new - grad

        # Update H
        rho = 1.0 / (y @ s)
        I = np.eye(len(x))
        H = (
            (I - rho * np.outer(s, y))
            @ H
            @ (I - rho * np.outer(y, s))
            + rho * np.outer(s, s)
        )

        # Prepare for the next iteration
        x = x_new
        grad = grad_new
        k += 1
    return x
    