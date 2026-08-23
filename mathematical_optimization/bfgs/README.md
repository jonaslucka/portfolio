# BFGS

BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton algorithm for unconstrained minimization.

Instead of computing the Hessian of the objective function explicitly, BFGS builds and iteratively updates an approximation of the inverse Hessian using gradient information.

Here the implementation is based on Nocedal and Wright: Numerical Optimization

## Mathematical Background

Given an objective function

$$
f(x),
$$

BFGS uses the gradient 

$$
\nabla f(x)
$$

to construct a approximation $H_k$ of the inverse Hessian

$$\nabla^2 f(x_k)^{-1}.$$

The search direction is computed as

$$
p_k = -H_k \nabla f(x_k).
$$

A line search is then used to determine a suitable step size $\alpha_k$.

In this implementation, the line search uses the strong Wolfe conditions for the construction.

The new iterate is

$$
x_{k+1} = x_k + \alpha_k p_k.
$$

Calculate the change in parameters

$$
s_k = x_{k+1} - x_k
$$

and the change in the gradient

$$
y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
$$

to update the inverse Hessian approximation.

The BFGS update is

$$H_{k+1}=(I - \rho_k s_k y_k^T)H_k(I - \rho_k y_k s_k^T)+\rho_k s_k s_k^T$$

where

$$
\rho_k = \frac{1}{y_k^T s_k}.
$$

The initial approximation is typically chosen as

$$
H_0 = I
$$

## Algorithm

Starting from $x_0$, BFGS repeatedly performs the following steps:

1. Compute the gradient $\nabla f(x_k)$
2. Compute the search direction

$$
p_k = -H_k \nabla f(x_k).
$$

3. Use a strong Wolfe line search to determine $\alpha_k$
4. Take a step

$$
x_{k+1} = x_k + \alpha_k p_k
$$

5. Compute $s_k$ and $y_k$
6. Update the inverse Hessian approximation $H_k$
7. Repeat until the gradient norm is below the specified tolerance

## Implementation

```python
optimize_bfgs(f, grad_f, x0, H0, tol, max_it)
```

where the inputs are:

- **f**: is the objective function.
- **grad_f**:  is the gradient of the objective function.
- **x0**: is the initial vector.
- **H0**: Initial inverse Hessian approximation. If omitted then the identity matrix is used
- **tol**: Convergence tolerance. Default 1e-5
- **max_it**: Maximum number of BFGS iterations. Default: 100

The function returns the minimizer.

## Computational Cost and Properties

BFGS stores the full inverse Hessian approximation $H_k$, which is an $n \times n$ matrix.
The storage requirement is therefore $O(n^2).$

The matrix updates require $ O(n^2) $ arihtmetic operations per iteraion, in addition to the cost of evaluating the objective function, evaluating the gradient and the cost of the line search.

Further the algorithms rate of convergence is superlinear and the algorithm is generally robust.

## Tests

The implementation is tested using pytest.

The tests cover:

- convergence on quadratic functions
- convergence on the Rosenbrock function
- reduction of the objective fucntion value
- convergence of the gradient towards zero

## References

The implementation follows the BFGS algorithms described in:

- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
  Springer, 2006, Algorithms 6.1.