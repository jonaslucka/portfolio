# Mathematical Optimization

This directory contains implementations of numerical opitmization algorithms from scratch using Python and NumPy.

## Algorithms

### BFGS

BFGS is a quasi-Newton optimization algorithm that apporixmates the inverse Hessian matrix.

### L-BFGS

L-BFGS is a memory-efficient variant of BFGS that stores only a limited number of correction pairs.

### Minibatch SGD
Minibatch stochastic gradient descetn is a first-order optimization method that updates the parameters using the gradient computed on small subset of the dataset.

## Tests

Each algorithm has an associated pytest test module.

## Comparison

| Algortihm | Derivatives | Memory
|---|---|---|
| BFGS | Gradient | O(n^2)
| L-BFGS | Gradient | O(mn) |
| Minibatch SGD | Gradient | O(n) |