"""
Line Search implementation

This module provides an implementation of a line search algorithm
which fullfills the wolfe conditions.

Author: Jonas Lucka
Date: 2026
"""

def zoom(f, grad_f, x, p, alpha_lo, alpha_hi, c1=1e-5, c2=0.9, max_it=100):
    """
    Zoom phase of the strong Wolfe line search.

    Implements Algorithm 3.6 from Nocedal and Wright: Numerical Optimization 2nd edition

    This function searches for a step length alpha_star in the
    interval [alpha_lo, alpha_hi].
    The trial point is chosen by bisection.

    Parameters
    ----------
    f : callable
        Objective function.

    grad_f : callable
        Gradient of f.

    x : ndarray
        Current point.

    p : ndarray
        Search direction. It is assumed to be a descent direction.

    alpha_lo : float
        Lower endpoint of the current interval.

    alpha_hi : float
        Upper endpoint of the current interval.

    c1 : float, optional
        Parameter of the sufficient decrease condition.

    c2 : float, optional
        Parameter of the strong Wolfe curvature condition.

    max_it : int, optional
        Maximum number of iterations.

        
    Returns
    -------
    alpha_j : float
              A step length satisfying the strong Wolfe conditions.


    Raises
    ------
    ValueError
        If input parameters are invalid.

    RuntimeError
        If no acceptable step length is found within max_iter iterations.
    """
    # Check if alpha_lo and alpha_hi fullfills their assumptions
    if alpha_lo >= alpha_hi:
        raise ValueError(
            "alpha_lo and alpha_hi must satisfy alpha_lo < alpha_hi."
            )

    phi_0 = f(x)
    grad_0 = grad_f(x)
    dphi_0 = grad_0 @ p
    phi_lo = f(x + alpha_lo * p)

    for _ in range(max_it):

        # Interpolate between alpha_lo and alpha_hi. We use bisection.
        alpha_j = 0.5 * (alpha_lo + alpha_hi)

        phi_j = f(x + alpha_j * p)

        # Check if sufficient decrease condition is not fullfilled or whether
        # if the function value has increased
        if (phi_j > phi_0 + c1 * alpha_j * dphi_0 or phi_j >= phi_lo):
            alpha_hi = alpha_j

        else:
            grad_j = grad_f(x + alpha_j * p)
            dphi_j = grad_j @ p

            # Check if Strong Wolfe curvature condition is fullfilled
            if abs(dphi_j) <= -c2 * dphi_0:
                return alpha_j

            if dphi_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo

            # Prepare for the next iteration
            alpha_lo = alpha_j
            phi_lo = phi_j

    raise RuntimeError(
        "Zoom failed to find a step length satisfying "
        "the strong Wolfe conditions within"
        f"max_it={max_it} iterations."
    )


def wolfe_line_search(f, grad_f, x, p, alpha_max=10.0, alpha_1=1.0, c1=1e-5, c2=0.9, max_it=100):
    """
    Line search satisfying the strong Wolfe conditions.

    Implements Algorithm 3.5 from Nocedal and Wright: Numerical Optimization 2nd edition

    Parameters
    ----------
    f : callable
        Objective function.

    grad_f : callable
        Gradient of f.

    x : ndarray
        Current point.

    p : ndarray
        Descent direction.

    alpha_max : float, optional
        Maximum allowed step length.

    alpha_1 : float, optional
        Initial trial step length. Must satisfy
        0 < alpha_1 < alpha_max.

    c1 : float, optional
        Parameter for the sufficient decrease condition.

    c2 : float, optional
        Parameter for the curvature condition.

    max_it : int, optional
        Maximum number of line search iterations.

        
    Returns
    -------
    float
        Step length satisfying the strong Wolfe conditions.


    Raises
    ------
    ValueError
        If the input parameters are invalid or p is not a descent direction.

    RuntimeError
        If no acceptable step length is found within max_it iterations.
    """
    # Check if alpha_1 and parameters c1 and c2 fullfills their assumptions
    if not 0 < alpha_1 < alpha_max:
        raise ValueError(
            "alpha_1 must satisfy 0 < alpha_1 < alpha_max."
        )

    if not 0 < c1 < c2 < 1:
        raise ValueError(
            "Parameters must satisfy 0 < c1 < c2 < 1."
        )

    phi_0 = f(x)
    grad_0 = grad_f(x)
    dphi_0 = grad_0 @ p

    # Check if p is descent direction
    if dphi_0 >= 0:
        raise ValueError(
            "p must be a descent direction."
        )

    alpha_0 = 0.0
    phi_prev = phi_0

    alpha = alpha_1

    for i in range(1, max_it + 1):

        phi = f(x + alpha * p)

        # Check if sufficient decrease condition is not fullfilled or whether
        # if the function value has increased
        if (phi > phi_0 + c1 * alpha * dphi_0
            or (phi >= phi_prev and i > 1)):
            return zoom(f, grad_f, x, p, alpha_0, alpha, c1=c1, c2=c2, max_it=max_it)

        grad = grad_f(x + alpha * p)
        dphi = grad @ p

        # Check if Strong Wolfe curvature condition is fullfilled
        if abs(dphi) <= -c2 * dphi_0:
            return alpha

        if dphi >= 0:
            return zoom(f, grad_f, x, p, alpha_0, alpha, c1=c1, c2=c2, max_it=max_it)

        # Choose alpha_next in the intervall (alpha_i, alpha_max)
        alpha_next = min(2.0 * alpha, alpha_max)

        # Prepare for next iteration
        alpha_0 = alpha
        phi_prev = phi
        alpha = alpha_next

    raise RuntimeError(
        "Wolfe line search failed to find a step length "
        "satisfying the strong Wolfe conditions within"
        f"max_it={max_it} iterations."
    )
