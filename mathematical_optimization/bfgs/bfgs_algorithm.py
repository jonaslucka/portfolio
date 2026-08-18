"""
BFGS implementation

This module provides an implementation of the BFGS algorithm.

Author: Jonas Lucka
Date: 2026
"""

import numpy as np

from ..line_search import wolfe_line_search


def optimize_bfgs(f, grad_f, x0, H0=None, tol=1e-5, max_it=1000):
    """
    BFGS method which is based on Algorithm 6.1 in
    Nocedal and Wright: Numerical Optimization 2nd edition
    The step length is obtained from a strong Wolfe line search.

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

    max_it : int, optional
        Maximum number of iterations.


    Returns
    -------
    x : ndarray
        Approximate minimizer.


    Raises
    ------
    RuntimeError
        If no acceptable step length is found within max_it iterations.
    """

    x = x0
    grad = grad_f(x)
    k = 0

    # Check if H0 is given
    if H0 is None:
        H = np.eye(len(x))
    else:
        H = H0

    while np.linalg.norm(grad) > tol and k < max_it:

        # Calculation of descent direction p and take a step with stepsize alpha
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

    # Check why the algorithm stopped
    if np.linalg.norm(grad) <= tol:
        return x

    raise RuntimeError(
        "BFGS failed to converge within the maximum "
        "number of iterations."
    )
