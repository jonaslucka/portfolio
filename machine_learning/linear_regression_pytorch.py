"""
Linear Regression implementation with PyTorch

This module provides an implementation of the Linear Regression algorithm using PyTorch.

Author: Jonas Lucka
Date: 2026
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class LinearRegression(nn.Module):
    """
    Represents a linear regresion model and allows for fitting and prediction. 
    This class and methods are based on Chapter 3.4 and Chapter 3.5 in: 
        
        Dive into Deep Learning from Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola
    
        The model predicts: y_hat = x @ w + b 
    
        where w is the vector weights and b is the bias. 
    
        
        Parameters
        ----------
        num_features : int
            Number of features of the input data. 

        learning_rate : float, default=0.01
            Step size used by the optimizer when updating the parameters. 

        Attributes
        ----------
        linear : torch.nn.Linear
            PyTorch linear layer containing the model weights and bias.

        loss_fn : torch.nn.MSELoss
            Mean squared error loss function.

        optimizer : torch.optim.Optimizer
            Stochastic gradient descent optimizer. 

        loss_history : list 
            History of the loss values recorded by the optimizer during training.
    """
    def __init__(self, num_features, learning_rate=0.01):
        super().__init__()

        self.linear = nn.Linear(num_features, 1)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=learning_rate
        )
        self.loss_history = []

    def forward(self, x):
        """
        Compute predictions for the given input data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data in form of a matrix

        Returns
        ------- 
        torch.Tensor
            Predicted target values.
        """
        return self.linear(x)

    def fit(self, x, y, epochs=10, batch_size=32):
        """
        Fit the model to the given training data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data in form of a matrix.

        y : torch.Tensor
            Target values.

        epochs : int, default=10
            Number of complete passes through the training dataset.

        batch_size : int, default=32
            Number of samples used to compute each gradient update.
            
        Returns
        -------
        LinearRegression
            The fitted model.

        Raises
        ------
        ValueError
            If x is not a 2-dimensional tensor.
        """
        if x.ndim != 2:
            raise ValueError("x must be a 2-dimensional tensor")

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.loss_history = []
        self.train()

        for _ in range(epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in dataloader:
                predictions = self.forward(x_batch)
                loss = self.loss_fn(predictions, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.detach().item()
            average_epoch_loss = epoch_loss / len(dataloader)
            self.loss_history.append(average_epoch_loss)
        return self

    def predict(self, x):
        """
        Predict target values for the given input data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data in form of a matrix.

        Returns
        -------
        torch.Tensor
            Predicted target values.
        """
        self.eval()
        with torch.no_grad():
            return self(x)
