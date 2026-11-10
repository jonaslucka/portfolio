"""
MNIST Neural Network Implementation

This module provides an implementation of a feedforward neural network
for handwritten digit classification using PyTorch.

The network is trained on the MNIST dataset and uses stochastic gradient
descent for optimization.

The network architecture consists of:
    - Input layer: 784 neurons
    - Hidden layer: 128 neurons with ReLU activation
    - Output layer: 10 neurons (digits 0-9)

The model is trained using the cross-entropy loss function.

Author: Jonas Lucka
Date: 2026
"""

import torch
import torch.nn.functional as F


def init_mnist_params():
    """
    Initialize the parameters of the MNIST neural network.

    Weights are initialized by using a normal distribution with a standard
    deviation of 0.01 and Biases are initialized to be zero.

    Returns
    -------
    dict
        Dictionary containing the network parameters:

        W1 : torch.Tensor
            Weight matrix of the first layer with shape (784, 128).

        b1 : torch.Tensor
            Bias vector of the first layer with shape (128,).

        W2 : torch.Tensor
            Weight matrix of the second layer with shape (128, 10).

        b2 : torch.Tensor
            Bias vector of the second layer with shape (10,).
    """
    return {
        "W1": (torch.randn(784, 128) * 0.01).requires_grad_(),
        "b1": torch.zeros(128, requires_grad=True),
        "W2": (torch.randn(128, 10) * 0.01).requires_grad_(),
        "b2": torch.zeros(10, requires_grad=True),
    }


def mnist_forward(params, x):
    """
    Perform the forward pass through the neural network.

    The input data is passed through two fully connected layers
    where a ReLU activation function is applied after the first layer.

    The output consists of 10 logits, representing the model
    unnormalized scores for each digit.

    The model predicts:

        y_hat = ReLU(x @ W1 + b1) @ W2 + b2

    Parameters
    ----------
    params : dict
        Dictionary containing the network weights and biases.

    x : torch.Tensor
        Input data containing flattened MNIST images.

    Returns
    -------
    torch.Tensor
        Output logits with shape (n_samples, 10).
    """
    h = torch.relu(x @ params["W1"] + params["b1"])
    logits = h @ params["W2"] + params["b2"]
    return logits


def mnist_loss(params, x, y):
    """
    Compute the cross-entropy loss of the neural network.

    The loss is calculated by comparing the predicted class scores
    with the true class labels.

    Parameters
    ----------
    params : dict
        Dictionary containing the network weights and biases.

    x : torch.Tensor
        Input data with shape (n_samples, 784).

    y : torch.Tensor
        Integer class labels with shape (n_samples,).
        Labels must be integers between 0 and 9.

    Returns
    -------
    torch.Tensor
        Cross-entropy loss.
    """
    logits = mnist_forward(params, x)

    return F.cross_entropy(logits, y)
