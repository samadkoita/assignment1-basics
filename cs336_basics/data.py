from __future__ import annotations
import time
import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable

import einops
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
import numpy as np

from einops import repeat

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    rd_idx = np.random.choice(len(dataset) - context_length, batch_size, replace=False)
    indices = rd_idx[:, None] + np.arange(context_length)
    ineff = dataset[indices]
    labels = dataset[indices + 1]
    # ineff = np.stack([dataset[r:r + context_length] for r in rd_idx])
    # labels = np.stack([dataset[r+1:r + 1 + context_length] for r in rd_idx])

    

    data = torch.tensor(ineff, device=device)
    labels = torch.tensor(labels, device=device)
    assert data.shape == labels.shape
    return data, labels
np.memmap