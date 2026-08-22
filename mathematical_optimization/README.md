# Mathematical Optimization

This directory contains implementations of numerical optimization algorithms from scratch using Python and NumPy.

## Algorithms

### BFGS

BFGS is a quasi-Newton optimization algorithm that apporixmates the inverse Hessian matrix of the objective function.

See [`bfgs/`](bfgs/) for the implementation and tests.

### L-BFGS

L-BFGS is a memory-efficient variant of BFGS. Instead of storing the full inverse Hessian approximation, it stores a limited number of correction pairs and computes the search direction using the two-loop recursion.

See [`l_bfgs/`](l_bfgs/) for the implementation and tests

### Minibatch SGD

Minibatch stochastic gradient descent is a first-order optimization method that updates the parameters using the gradient computed on small subset (minibatches) of the dataset.

See [`sgd/`](sgd/) for the implementation and tests.

## Line Search

The BFGS and L-BFGS implementations use a strong Wolfe line search to determine an appropiate step size.

The line search consists of two phases:

1. **Bracketing phase** — searches for an interval containing a step length satisfying the Wolfe conditions.
2. **Zoom phase** — repeatedly reduces the interval until a suitable step length is found.

The implementation is based on Algorithm 3.5 and 3.6 from Nocedal and Wright, *Numerical Optimization*, 2nd edition.

This implementation uses bisection during the zoom phase.

See [`line_search.py`](line_search.py) for the implementation.

## Tests

Each algorithm has an associated pytest test module.

## Comparison

| Algortihm | Derivatives | Memory
|---|---|---|
| BFGS | Gradient | O(n^2)
| L-BFGS | Gradient | O(mn) |
| Minibatch SGD | Gradient | O(n) |

Here, `n` is the number of optimization parameters and `m` is the number of stored correction pairs in L-BFGS.