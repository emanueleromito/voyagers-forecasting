# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import numpy as np
from typing import Optional

class TimeSeriesGenerator(abc.ABC):
    """
    Abstract base class for time series generators.
    """
    
    @abc.abstractmethod
    def generate(self, length: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate a time series of the specified length.
        
        Parameters
        ----------
        length
            The length of the time series to generate.
        random_state
            Random seed for reproducibility.
            
        Returns
        -------
        np.ndarray
            The generated time series of shape (length,).
        """
        pass
