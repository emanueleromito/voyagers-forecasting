# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict, Union, Optional
import numpy as np
import torch

from .base import TimeSeriesGenerator
from .multivariate import Multivariatizer, LinearMixupMultivariatizer, NonLinearMixupMultivariatizer, SequentialMultivariatizer

class TaskSampler:
    """
    Samples heterogeneous forecasting tasks (Univariate, Multivariate, Covariate-Informed).
    """
    def __init__(
        self,
        base_generators: List[TimeSeriesGenerator],
        multivariatizers: Optional[List[Multivariatizer]] = None,
        univariate_prob: float = 0.4,
        multivariate_prob: float = 0.3,
        covariate_prob: float = 0.3,
        min_variates: int = 2,
        max_variates: int = 5,
    ):
        self.base_generators = base_generators
        if multivariatizers is None:
            self.multivariatizers = [
                LinearMixupMultivariatizer(base_generators),
                NonLinearMixupMultivariatizer(base_generators),
                SequentialMultivariatizer(base_generators),
            ]
        else:
            self.multivariatizers = multivariatizers
            
        self.probs = [univariate_prob, multivariate_prob, covariate_prob]
        assert sum(self.probs) == 1.0
        
        self.min_variates = min_variates
        self.max_variates = max_variates

    def sample(self, length: int, random_state: Optional[int] = None) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        
        task_type = rng.choice(['univariate', 'multivariate', 'covariate'], p=self.probs)
        
        if task_type == 'univariate':
            return self._sample_univariate(length, rng, random_state)
        elif task_type == 'multivariate':
            return self._sample_multivariate(length, rng, random_state)
        else:
            return self._sample_covariate(length, rng, random_state)

    def _sample_univariate(self, length: int, rng, random_state: Optional[int] = None) -> Dict:
        gen = rng.choice(self.base_generators)
        ts = gen.generate(length, random_state=random_state)
        return {"target": ts}

    def _sample_multivariate(self, length: int, rng, random_state: Optional[int] = None) -> Dict:
        n_variates = rng.randint(self.min_variates, self.max_variates + 1)
        multivariatizer = rng.choice(self.multivariatizers)
        ts = multivariatizer.generate(length, n_variates, random_state=random_state)
        return {"target": ts}

    def _sample_covariate(self, length: int, rng, random_state: Optional[int] = None) -> Dict:
        n_variates = rng.randint(self.min_variates, self.max_variates + 1)
        multivariatizer = rng.choice(self.multivariatizers)
        ts = multivariatizer.generate(length, n_variates, random_state=random_state) # (n_variates, length)
        
        # Partition variates into target, past_covariates, future_covariates
        # We need at least 1 target
        n_targets = rng.randint(1, n_variates)
        n_covariates = n_variates - n_targets
        
        # Shuffle indices to randomly assign roles
        indices = np.arange(n_variates)
        rng.shuffle(indices)
        
        target_indices = indices[:n_targets]
        covariate_indices = indices[n_targets:]
        
        target = ts[target_indices]
        
        past_covariates = {}
        future_covariates = {}
        
        for i, idx in enumerate(covariate_indices):
            name = f"cov_{i}"
            cov_data = ts[idx]
            
            # Randomly decide if it's known in future or past-only
            is_known = rng.rand() > 0.5
            
            past_covariates[name] = cov_data
            if is_known:
                # For training data, we just need to indicate WHICH covariates are known
                # The actual values are taken from 'past_covariates' (which contains the full series)
                # by the dataset slicing logic.
                # We set it to None so Chronos2Dataset fills it with NaNs of correct length (prediction_length)
                # or we can just pass a dummy array of correct length if we knew prediction_length here.
                # But passing None is safer as the dataset handles it.
                future_covariates[name] = None
                
        return {
            "target": target,
            "past_covariates": past_covariates,
            "future_covariates": future_covariates
        }
