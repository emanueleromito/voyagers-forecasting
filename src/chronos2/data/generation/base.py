# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import numpy as np

class TimeSeriesGenerator(abc.ABC):
    """
    Abstract base class for time series generators.
    """
    
    @abc.abstractmethod
    def generate(self, length: int) -> np.ndarray:
        """
        Generate a time series of the specified length.
        
        Parameters
        ----------
        length
            The length of the time series to generate.
            
        Returns
        -------
        np.ndarray
            The generated time series of shape (length,).
        """
        pass
