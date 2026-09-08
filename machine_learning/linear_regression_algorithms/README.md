# Linear Regression
Linear regression is a supervised learning method used to model the relationship between a dependent variable and one or more independent variables.

The objective is to find a linear function that minimizes the difference between the observed target values and the predicted values.

This folder contains two implementations of linear regression:
* NumPy: implementation from scratch using an external optimizer.
* PyTorch: implementation using torch.nn.Linear, automatic differentiation and SGD.

Here the implementations are based on Dive into Deep Learning.

## Mathematical Background

Consider a supervised regression dataset $D$ containing $n$ observations and $d$ features:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n}
$$

where:

* $x_i \in \mathbb{R}^{d}$ is the feature vector for observation $i$.
* $y_i \in \mathbb{R}$ is the target value for observation $i$.

The input data are collected into a feature matrix:
$$
X =
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1d}\\
x_{21} & x_{22} & \cdots & x_{2d}\\
\vdots & \vdots & \ddots & \vdots\\
x_{n1} & x_{n2} & \cdots & x_{nd}
\end{bmatrix}
\in \mathbb{R}^{n\times d}.
$$

The targed values are collected into a vector:

$$
y =
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}
\in \mathbb{R}^{n}..
$$

Linear regression learns:

* a weight vector $w$
* a scalar bias $b$

The weight vector contains one parameter for every input feature:

$$
w =
\begin{bmatrix}
w_1\\
w_2\\
\vdots\\
w_d
\end{bmatrix}
\in \mathbb{R}^{d}.
$$

The bias is a scalar:

$$
b \in \mathbb{R}.
$$

For all $n$ observations the model predicts:

$$
\hat y = Xw + b.
$$

The model is trained by minimizing the Mean Squared Error:
$$
L(w,b)
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2.
$$

For the NumPy implementation the gradients with respect to the weights and bias are computed explicitly.

The gradient with respect to the weights is

$$
\nabla_w L
=
\frac{2}{n}X^T(Xw+b-y)
$$

and the gradient with respect to the bias is

$$
\nabla_b L
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i).
$$

These gradients are passed to the optimization algorithm to update the model parameters.

## Algorithm

The algorithm is:

1. Initialize the weights $w$ and the bias $b$.
2. Select a minibatch of observations.
3. Compute the predictions:

$$
\hat y=Xw+b.
$$

4. Compute the Mean Square Error loss.
5. Compute the gradients.
6. Update the parameters using the optimizer.
7. Repeat for the specified number of epochs.

## Implementations

### NumPy
```python
class LinearRegression:
    def __init__(self):
        self.params = None
        self.num_features = None
        self.loss_history = None
```

The model attributes are:

- **params**: are the model parameters. It contains the weights w and the bias b
- **num_features**: is the number of features
- **loss_history**: is the history of loss  

&nbsp;

```python
loss(self, params, x, y)
```

The inputs are:

- **params**: are the model parameters. It contains the weights w and the bias b
- **x**: is the input data in form of a feature matrix
- **y**: are the target values

The method returns the Mean Squared Error of the model.

&nbsp;

```python
gradient(self, params, x, y)
```

The inputs are:

- **params**: are the model parameters. It contains the weights w and the bias b
- **x**: is the input data in form of a feature matrix
- **y**: are the target values

The method returns the gradient of the Loss fucnction

&nbsp;

This is the main method of this class. This method trains the model.

```python
model.fit(
    self,
    x,
    y,
    optimizer,
    learning_rate=0.01,
    epochs=100,
    batch_size=32
)
```

The inputs are:

- **x**: is the input data in form of a feature matrix.
- **y**: are the target values.
- **optimizer**: Optimization function used to update the model parameters.
- **learning_rate**: is the step size used by the optimizer when updating the parameters. Default: 0.01
- **epochs**: is the number of complete passes through the training dataset. Default: 100
- **batch_size**: is the number of samples used to compute each gradient update. Default: 32

This method returns the fitted model

&nbsp;

```python
predict(self, x):
```

The input is:

- **x**: is the input data in form of a feature matrix.

This method predict values for given input.

&nbsp;

### PyTorch
```python
class LinearRegression(nn.Module):
    def __init__(self, num_features, learning_rate=0.01):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=learning_rate
        )
        self.loss_history = []
```
The inputs are:

- **num_features**: is the number of features of the input data.
- **learning_rate**: is the step size used by the optimizer when updating the parameters. Default=0.01
            
The class attributes are:
- **linear**: PyTorchs linear layer containing the model weights and bias.
- **loss_fn**: is the Mean squared error loss function.
- **optimizer**:is the Stochastic gradient descent optimizer. 
- **loss_history** : is the history of the loss values.

&nbsp;

```python
forward(self, x):
```

The input is:

- **x**: is the input data in form of a feature matrix.

This methods gives the predicted values of the given model.

&nbsp;

```python
fit(self, x, y, epochs=10, batch_size=32)
```

The inputs are:

- **x**: is the input data in form of a feature matrix.
- **y**: are the target values.
- **epochs**: is the number of complete passes through the training dataset. Default: 10
- **batch_size**: is the number of samples used to compute each gradient update. Default: 32

This method returns the fitted model

&nbsp;

```python
predict(self, x):
```

The input is:

- **x**: is the input data in form of a feature matrix.

This method predict values for given input.

## Tests
Both implementations are tested using pytest.

The tests cover:

- Correct predictions
- Learning a simple linear relationship
- Learning a relationship wiht multiple features
- Loss calculation
- Gradient calculation
- Loss history
- Input validation
- Mismatched sample counts

## References

The implementations follows the Linear Regression algorithms described in:

Zhang, A., & Lipton, Z. C., & Li M. and Smola A. J. (2023) *Dive into Deep Learning*. Cambridge University Press. https://D2L.ai
Chapters 3.4 and 3.5