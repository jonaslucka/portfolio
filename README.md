# Portfolio
A portfolio of projects involving Python, SQL, optimization methods and machine learning.

## Project Structure

```text
portfolio/
│
├── optimization/
│   ├── bfgs/
│   │   ├── bfgs_algorithm.py
│   │   └── test_bfgs.py
│   │
│   ├── lbfgs/
│   │   ├── l_bfgs_algorithm.py
│   │   └── test_l_bfgs.py
│   │
│   └── sgd/
│       ├── minibatch_sgd_algorithm.py
│       └── test_minibatch_sgd.py
│
├── machine_learning/
│   └── linear_regression/
│
└── sql/
```

Each Python package contains an __init__.py file where appropriate.

## References

- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*
  (2nd ed.). Springer.
  https://doi.org/10.1007/978-0-387-40065-5

- Raschka, S. *How is stochastic gradient descent implemented
  in the context of machine learning and deep learning?*
  https://sebastianraschka.com/faq/docs/sgd-methods.html