"""
MNIST Model Training

This module provides functionality to train the MNIST neural network
using a minibatch gradient descent.

The training process uses:
    - Stochastic Gradient Descent optimizer.
    - Cross-entropy loss function.
    - Minibatch gradient descent.
    - Validation loss monitoring.

Author: Jonas Lucka
Date: 2026
"""

import torch

from projects.MNIST import mnist_loss

def train_model(
    params,
    train_loader,
    val_loader,
    learning_rate=0.1,
    epochs=10
):
    """
    Train the neural network using minibatch gradient descent.

    For each epoch the model is trained on minibatches from the
    training dataset. After each epoch the model is evaluated on
    the validation dataset to monitor its  performance

    Parameters
    ----------
    params : dict
        Dictionary containing the initial network weights and biases.

    train_loader : DataLoader
        DataLoader containing the training datsaet.

    val_loade : DataLoader
        DataLoader containing the validation dataset.

    learning_rate : float, default=0.1
        Learning rate used by the SGD optimizer.

    epochs : int, default=10
        Number of training epochs

    Returns
    -------
    params : dict
        Updated network parameters after training.

    history : dict
        Dictionary containing the average training and validation
        losses recorded after each epoch.

    Raises
    ------
    ValueError
        If learning_rate or epochs is not positive
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    optimizer = torch.optim.SGD(
        params.values(),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):

        # Training phase
        total_train_loss = 0.0

        for images, labels in train_loader:

            optimizer.zero_grad()

            loss = mnist_loss(params, images, labels)

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item() * images.size(0)

        average_train_loss = (
            total_train_loss / len(train_loader.dataset)
        )

        # Validation phase
        total_val_loss = 0.0

        with torch.no_grad():

            for images, labels in val_loader:

                loss = mnist_loss(params, images, labels)

                total_val_loss += loss.item() * images.size(0)

        average_val_loss = (
            total_val_loss / len(val_loader.dataset)
        )

        # Store training history
        history["train_loss"].append(average_train_loss)
        history["val_loss"].append(average_val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {average_train_loss:.4f} "
            f"Validation Loss: {average_val_loss:.4f}"
        )

    return params, history

