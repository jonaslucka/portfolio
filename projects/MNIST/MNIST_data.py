"""
MNIST data loading and preprocessing

This module provides functionality to download, preprocess, and load
the MNIST dataset using PyTorch.

Author: Jonas Lucka
Date: 2026
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_mnist_data(
    data_dir="data",
    batch_size=64,
    validation_size=10000,
    random_seed=32
):
    """
    Load and preprocess the MNIST dataset and create DataLoaders
    for training, validation and testing.

    The original training dataset is split into training and
    validation subsets using a reproducible random split.

    Images are converted to tensors, normalized to [0,1],
    and flattened into vectors of length 784

    Parameters
    ----------
    data_dir : str, default="data"
        Directory where the MNIST dataset is stored.

    batch_size : int, default=64
        Number of images per batch.

    validation_size : int, default=10000
        Number of training images reserved for validation.

    random_seed : int, default=32
        Random seed for reproducible dataset splitting
        and training data shuffling.

    Returns
    -------
    train_loader : DataLoader
        DataLoader containing the training subset.

    val_loader : DataLoader
        DataLoader containing the validation subset.

    test_loader : DataLoader
        DataLoader containing the test dataset.

    Raises
    ------
    ValueError
        If batch_size is not positive or validation_size
        is outside the valid range.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not 0 < validation_size < 60000:
        raise ValueError(
        "validation_size must be between 1 and 59999."
        )

    # Convert images to tensors and normalize pixel values to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Download the training dataset
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download the test dataset
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Split training data into training and validation sets
    train_size = len(train_dataset) - validation_size

    generator = torch.Generator().manual_seed(random_seed)

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, validation_size],
        generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
