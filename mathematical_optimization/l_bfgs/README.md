# L-BFGS

L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton method for unconstrained optimization.

L-BFGS is a limited-memory variant of BFGS. Instead of explicitly storing the full inverse Hessian approximation, L-BFGS stores a limited number of the most recent correction pairs.
This significantly reduces the memory requirements compared with BFGS.

Here the implementation is based on Nocedal and Wright: Numerical Optimization

## Mathematical Background

L-BFGS uses an approximation $H_k$ of the inverse Hessian of the objective function similar as in the BFGS algorithm.

But L-BFGS does not explicitly construct or store $H_k$.

Instead it stores the most recent $m$ pairs

$$
(s_i, y_i)
$$

where

$$
s_i = x_{i+1} - x_i
$$

and

$$
y_i = \nabla f(x_{i+1}) - \nabla f(x_i).
$$

The search direction is

$$
p_k = -H_k \nabla f(x_k).
$$

Typical values of $m$ are relatively small compared to $n$, the dimension of the optimization problem.

### Two-loop recursion

The product

$$
H_k \nabla f(x_k)
$$

is computed using the L-BFGS two-loop recursion.

First calculate

$$
q = \nabla f(x_k).
$$

The first loop processes the stored correction pairs in reverse order:

$$
\alpha_i = \rho_i s_i^T q,
$$

$$
q = q - \alpha_i y_i,
$$

where

$$
\rho_i = \frac{1}{y_i^T s_i}.
$$

Calculate with the initial inverse Hessian approximation:

$$
r = H_k^0 q.
$$

In this implementation, $H_k^0 = \gamma_k I$

where the scaling factor $\gamma_k$ is computed from the most recent correction
pair as

$$
\gamma_k =
\frac{s_{k-1}^T y_{k-1}}
     {y_{k-1}^T y_{k-1}}.
$$

Finally, the second loop in forward order:

$$
\beta_i = \rho_i y_i^T r,
$$

$$
r = r + s_i(\alpha_i - \beta_i).
$$

The resulting vector is

$$
r = H_k \nabla f(x_k).
$$

The search direction is

$$
p_k = -r.
$$

With that it is not necessary to store the approximated inverse Hessian.

## Algorithm

Starting from $x_0$, L-BFGS repeatedly performs the following steps:

1. Compute the gradient $\nabla f(x_k)$
2. Via the two loop recursion compute

$$
H_k \nabla f(x_k)
$$

3. Compute the search direction

$$
p_k = -H_k \nabla f(x_k).
$$

4. Use a strong Wolfe line search to determine the step size $\alpha_k$.

5. Take a step

$$
x_{k+1} = x_k + \alpha_k p_k.
$$

6. Compute$$s_k = x_{k+1} - x_k$$
and

$$
y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
$$

7. If the curvature condition $$s_k^T y_k > 0$$

   is satisfied, store the pair $(s_k, y_k)$.

8. If more than $m$ correction pairs are stored, discard the oldest pair.

9. Repeat until convergence.

## Implementation

```python
optimize_lbfgs(
    f,
    grad_f,
    x0,
    m=10,
    alpha_max=250,
    tol=1e-5,
    max_it=100,
    return_history=False
)
```

where the inputs are:

- **f**: is the objective function.
- **grad_f**: is the gradient of the objective function.
- **x0**: is the initial vector.
- **m**: Maximum number of correction pairs stored in memory.
- **alpha_max**: Maximum step length considered by the line search.
- **tol**: Convergence tolerance. Default 1e-5
- **max_it**: Maximum number of BFGS iterations. Default: 100
- **return_history**: If True, the sequence of iterates is also returned.

The function returns the minimizer and if return_history is True then the sequence of iterates is also returned.

The two loop recursion is implemented separately as:

```python
two_loop_recursion(grad_fk, s_history, y_history)
```

## Computational Cost and Properties

L-BFGS avoids the $O(n^2)$ storage which is required in the BFGS algorithm.

With $m$ stored correction pairs:

Memory: $O(mn)$
Two-loop recursion: $O(mn)$ per iteration

With $m=10$ the computational and memory requirements scale approximately linearly for big numbers of variables.

This makes L-BFGS particularly useful for large-scale unconstrained
optimization problems.

## Test

The implementation is tested using pytest.

The tests cover:

- convergence on quadratic functions
- convergence on the Rosenbrock function
- reduction of the objective fucntion value
- convergence of the gradient towards zero
- behavior of the two loop recursion with no stored correction pairs
- convergence with only one stored correction pair
- convergence on a higher dimension quadratic problem
- returning the optimization history

The Rosenbrock test uses a larger alpha_max for the line search because the default step-length limit was too restrictive from the chosen starting point

## References

The implementation follows the L-BFGS algorithm described in:

- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
  Springer, 2006, Algorithms 7.4 and 7.5.