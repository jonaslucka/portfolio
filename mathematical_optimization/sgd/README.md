# Minibatch Stochastic Gradient Descent

Minibatch Stochastic Gradient Descent (minibatch SGD) is an optimization algorithm often used for machine learning.

Minibatch SGD computes the gradient using a randomly selected subset, called a minibatch, of the training data.

Here the implementation is based on Sebastian Raschka:

https://sebastianraschka.com/faq/docs/sgd-methods.html

This implementation is written from scratch using NumPy and does not depend on frameworks like PyTorch or TensorFlow.

## Mathematical Background

Consider an objective function $L$ that is defined as the average loss over a dataset containing $n$ samples:

$$
L(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell(\theta; x_i, y_i),
$$

where $\theta$ is the parameter vector and $\ell$ is the loss for an
individual training example.

The gradient is

$$
\nabla L(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla_\theta \ell(\theta; x_i,y_i).
$$

Fullbatch gradient descent updates the parameters according to

$$
\theta_{k+1}
=
\theta_k
-
\eta \nabla L(\theta_k),
$$

where $\eta > 0$ is the learning rate.

Stochastic gradient descent estimates the gradient using a subset of the training data.

For a minibatch $B_k$ containing $b$ samples, the minibatch gradient is

$$
g_k
=
\frac{1}{b}
\sum_{i \in B_k}
\nabla_\theta \ell(\theta_k;x_i,y_i).
$$

The parameter update then becomes

$$
\theta_{k+1}
=
\theta_k
-
\eta g_k.
$$

The minibatch gradient is an unbiased estimator of the full gradient when the minibatch is sampled uniformly from the training set. 

Then each training example has probability $\frac{b}{n}$ of being included in the minibatch:
$$
\mathbb{E}[g_k]
= 
\frac{1}{b}\sum^n_{i=1}\frac{b}{n}\nabla_\theta \ell(\theta_k;x_i,y_i) 
= 
\frac{1}{n}\sum^n_{i=1}\nabla_\theta \ell(\theta_k;x_i,y_i)
=
\nabla L(\theta_k).
$$

The minibatch gradient generally differs from the full gradient for an individual update, but it has the correct expectation under uniform sampling.

## Algorithm

Given training data

$$
D = \{(x_i,y_i)\}_{i=1}^n
$$

and an initial parameter vector $\theta$, minibatch SGD repeatedly updates the parameters using the gradient estimated from a minibatch.

For each epoch:

1. Shuffle the training data.
2. Split the shuffled data into minibatches $B$.
3. Compute the gradient of the loss on each minibatch.
4. Update the parameters:$$\theta_{k+1}=\theta_k-\eta \nabla L_B(\theta_k),$$
where
$\nabla L_B(\theta_k)$ is the gradient estimated using the minibatch B.

5. Optionally evaluate the loss on the complete training dataset.


The optimizer can be used with different models and loss function.

## Minibatch Size

The `batch_size` parameter determines how many samples are used for each parameter update.

### Stochastic Gradient Descent

If batch_size = 1 each update uses a single training example.

This corresponds to stochastic gradient descent.

### Minibatch SGD

If 1 < batch_size < n each update uses a subset of the training data.

This is the usual minibatch SGD setting.

### Fullbatch Gradient Descent

If batch_size = n the entire training dataset is used for each update.

This corresponds to the fullbatch gradient descent.

## Implementation

```python
optimize_minibatch_sgd(
        x,
        y,
        initial_params,
        gradient_fn,
        loss_fn=None,
        learning_rate=0.01,
        epochs=100,
        batch_size=32
    )
```

where the inputs are:

- **x**: Training samples
- **y**: Target values corresponding to the training samples.
- **initial_params**: Initial parameter vector
- **gradient_fn**: Function that computes the gradient of the loss with respect to the parameters for a given minibatch.

It must have the form:
```python
gradient_fn(params, x_batch, y_batch)
```
- **loss_fn**: Loss function. If given it is used to record the loss after each epoch.

It must have the form:
```python
loss_fn(params, x, y)
```
- **learning_rate**: Step size used for each parameter update.
- **epochs**: 
Number of complete passes through the training dataset.
- **batch_size**: Number of samples used for each gradient update.

## Computational Cost and Properties

The computational cost of an update depends on the model and on how the gradient is computed. If evaluating the gradient for one sample requires $$O(n)$$ operations, where n is the number of parameters, then a minibatch with b samples requires approximately $$O(bn)$$ operations.

The optimizer itself requires $$O(n)$$ additional memory for the parameters and gradient. The training data must also be stored in memory, requiring $$O(Nd)$$ memory for N samples with d features, with some additional overhead due to shuffling and minibatch construction.

## Test

The implementation is tested using pytest.

The tests cover:

- convergence on a linear regression problem
- reduction of the loss
- correct loss_history behaviour
- batch_size=1 behaves as stochastic gradient descent
- fullbatch training works correctly
- incomplete final minibatches are handled correctly

## References

The implementation follows the Minibatch SGD v2 algorithm from:
https://sebastianraschka.com/faq/docs/sgd-methods.html, Algorithm 5)