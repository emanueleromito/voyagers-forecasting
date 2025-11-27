# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from .base import TimeSeriesGenerator


class Multivariatizer(abc.ABC):
    """
    Base class for Multivariatizers.
    Takes a list of base generators and produces a multivariate time series.
    """
    def __init__(self, base_generators: List[TimeSeriesGenerator]):
        self.base_generators = base_generators

    @abc.abstractmethod
    def generate(self, length: int, n_variates: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate a multivariate time series.

        Parameters
        ----------
        length
            Length of the time series.
        n_variates
            Number of variates (dimensions) to generate.
        random_state
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Multivariate time series of shape (n_variates, length).
        """
        pass


class LinearMixupMultivariatizer(Multivariatizer):
    """
    Generates multivariate series by linear combinations of independent base series.
    Y = A * X, where X are independent base series and A is a mixing matrix.
    """
    def generate(self, length: int, n_variates: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        
        # Generate n_variates independent base series
        # We randomly pick a generator for each latent dimension
        base_series_list = []
        for i in range(n_variates):
            gen = rng.choice(self.base_generators)
            # Use deterministic sub-seed if random_state is provided
            sub_seed = random_state + i if random_state is not None else None
            base_series_list.append(gen.generate(length, random_state=sub_seed))
        
        X = np.stack(base_series_list) # (n_variates, length)
        
        # Random mixing matrix
        A = rng.randn(n_variates, n_variates)
        
        # Mix
        Y = A @ X
        
        return Y


class NonLinearMixupMultivariatizer(Multivariatizer):
    """
    Generates multivariate series by passing independent base series through a random MLP.
    """
    def generate(self, length: int, n_variates: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        
        # Generate n_variates independent base series
        base_series_list = []
        for i in range(n_variates):
            gen = rng.choice(self.base_generators)
            sub_seed = random_state + i if random_state is not None else None
            base_series_list.append(gen.generate(length, random_state=sub_seed))
        
        X = np.stack(base_series_list).T # (length, n_variates)
        X_torch = torch.from_numpy(X).float()
        
        # Random MLP with deterministic initialization
        if random_state is not None:
            torch.manual_seed(random_state)
        
        mlp = nn.Sequential(
            nn.Linear(n_variates, n_variates * 2),
            nn.ReLU(),
            nn.Linear(n_variates * 2, n_variates)
        )
        
        with torch.no_grad():
            Y_torch = mlp(X_torch)
            
        return Y_torch.numpy().T # (n_variates, length)


class SequentialMultivariatizer(Multivariatizer):
    """
    Generates multivariate series with lead-lag relationships.
    """
    def generate(self, length: int, n_variates: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        
        # Generate 1 base series
        gen = rng.choice(self.base_generators)
        base_series = gen.generate(length + n_variates, random_state=random_state) # Generate extra for shifting
        
        Y = []
        for i in range(n_variates):
            # Shift the base series
            lag = i
            Y.append(base_series[lag : lag + length])
            
        Y = np.stack(Y)
        
        # Add some independent noise to each variate to make it not perfectly correlated
        noise = rng.normal(0, 0.1 * np.std(Y), Y.shape)
        Y += noise
        
        return Y
