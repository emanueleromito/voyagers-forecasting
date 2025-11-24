# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import List, Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)

from .base import TimeSeriesGenerator


class KernelSynthGenerator(TimeSeriesGenerator):
    """
    Generates time series using Gaussian Processes with random kernels.
    """

    def __init__(
        self,
        max_kernels: int = 5,
        periodicities: List[int] = [24, 48, 96, 168, 336, 720, 1440, 8760, 17520],
        length_scales: List[float] = [0.1, 1.0, 10.0],
        data_length: int = 1024, # Used for kernel scaling
    ):
        self.max_kernels = max_kernels
        self.periodicities = periodicities
        self.length_scales = length_scales
        self.data_length = data_length
        
        self.kernel_bank = [
            *[ExpSineSquared(periodicity=p / data_length) for p in periodicities],
            *[RBF(length_scale=l) for l in length_scales],
            DotProduct(sigma_0=0.0),
            DotProduct(sigma_0=1.0),
            RationalQuadratic(alpha=0.1),
            RationalQuadratic(alpha=1.0),
            WhiteKernel(noise_level=0.1),
            WhiteKernel(noise_level=1.0),
            ConstantKernel(),
        ]

    def _random_binary_map(self, a: Kernel, b: Kernel):
        binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
        return np.random.choice(binary_maps)(a, b)

    def generate(self, length: int) -> np.ndarray:
        # Note: We use the configured data_length for X scaling to match the kernel parameters
        # But we sample 'length' points.
        X = np.linspace(0, 1, length)[:, None]
        
        selected_kernels = np.random.choice(
            self.kernel_bank, np.random.randint(1, self.max_kernels + 1), replace=True
        )
        kernel = functools.reduce(self._random_binary_map, selected_kernels)
        
        try:
            gpr = GaussianProcessRegressor(kernel=kernel)
            ts = gpr.sample_y(X, n_samples=1)
            return ts.squeeze()
        except Exception:
            # Fallback in case of numerical instability
            return np.random.randn(length)


class ARGenerator(TimeSeriesGenerator):
    """
    Generates time series using an Autoregressive (AR) process.
    """
    def __init__(self, order: int = 1, noise_std: float = 1.0):
        self.order = order
        self.noise_std = noise_std

    def generate(self, length: int) -> np.ndarray:
        # Randomly sample AR coefficients
        # We want a stable process, so we keep roots inside unit circle
        # A simple heuristic is to normalize coefficients to sum to < 1
        coeffs = np.random.uniform(-1, 1, self.order)
        coeffs /= (np.abs(coeffs).sum() + 0.1) 
        
        ts = np.zeros(length)
        # Initialize with noise
        ts[:self.order] = np.random.normal(0, self.noise_std, self.order)
        
        noise = np.random.normal(0, self.noise_std, length)
        
        for t in range(self.order, length):
            # AR(p) process: X_t = c + sum(phi_i * X_{t-i}) + epsilon_t
            # We assume c=0 for simplicity
            ts[t] = np.dot(coeffs, ts[t-self.order:t][::-1]) + noise[t]
            
        return ts


class TrendSeasonalityGenerator(TimeSeriesGenerator):
    """
    Generates time series by summing trend, seasonality, and noise components.
    """
    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std

    def generate(self, length: int) -> np.ndarray:
        t = np.arange(length)
        
        # 1. Trend
        trend_type = np.random.choice(['linear', 'quadratic', 'sigmoid'])
        if trend_type == 'linear':
            slope = np.random.uniform(-0.01, 0.01)
            trend = slope * t
        elif trend_type == 'quadratic':
            a = np.random.uniform(-0.0001, 0.0001)
            b = np.random.uniform(-0.01, 0.01)
            trend = a * t**2 + b * t
        else: # sigmoid
            k = np.random.uniform(0.01, 0.1)
            x0 = np.random.uniform(0, length)
            trend = 1 / (1 + np.exp(-k * (t - x0)))

        # 2. Seasonality
        seasonality = np.zeros(length)
        num_seasonal_components = np.random.randint(1, 4)
        for _ in range(num_seasonal_components):
            period = np.random.choice([12, 24, 168, 365]) # Common periods
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.1, 5.0)
            seasonality += amplitude * np.sin(2 * np.pi * t / period + phase)

        # 3. Noise
        noise = np.random.normal(0, self.noise_std, length)
        
        return trend + seasonality + noise
