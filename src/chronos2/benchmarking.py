"""
Benchmarking utilities for Chronos-2 models.

This module provides utilities for evaluating Chronos-2 models on various benchmarks
including GIFT-Eval and Monash datasets.
"""

import itertools
import logging
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality
from tqdm.auto import tqdm

from .model import Chronos2Model

logger = logging.getLogger(__name__)


class Chronos2Predictor:
    """GluonTS-style predictor wrapper for Chronos-2 model."""
    
    def __init__(
        self,
        model: Chronos2Model,
        prediction_length: int,
        batch_size: int = 32,
        quantile_levels: Optional[List[float]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor.
        
        Parameters
        ----------
        model : Chronos2Model
            The Chronos-2 model to use for predictions
        prediction_length : int
            Number of time steps to forecast
        batch_size : int, optional
            Batch size for inference, by default 32
        quantile_levels : List[float], optional
            Quantile levels to predict, by default uses model's configured quantiles
        device : str, optional
            Device to use ('cuda' or 'cpu'), by default auto-detect
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model.to(device).eval()
        self.device = device
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        
        if quantile_levels is None:
            self.quantile_levels = model.chronos_config.quantiles
        else:
            self.quantile_levels = quantile_levels
        
    def predict(self, test_data_input) -> List[Forecast]:
        """
        Generate forecasts for test data.
        
        Parameters
        ----------
        test_data_input : Iterable
            Test data in GluonTS format (dicts with 'target' and 'start' keys)
            
        Returns
        -------
        List[Forecast]
            List of GluonTS Forecast objects
        """
        test_data_list = list(test_data_input)
        forecasts = []
        
        # Process in batches
        for i in tqdm(range(0, len(test_data_list), self.batch_size), desc="Predicting", leave=False):
            batch_data = test_data_list[i:i + self.batch_size]
            
            # Prepare batch
            contexts = []
            for item in batch_data:
                target = torch.tensor(item["target"], dtype=torch.float32)
                contexts.append(target)
            
            # Pad to same length
            max_len = max(len(c) for c in contexts)
            padded_contexts = []
            for c in contexts:
                if len(c) < max_len:
                    pad = torch.full((max_len - len(c),), float('nan'))
                    c = torch.cat([pad, c])
                padded_contexts.append(c)
            
            context_batch = torch.stack(padded_contexts).to(self.device)
            
            with torch.no_grad():
                # Calculate num_output_patches
                output_patch_size = self.model.chronos_config.output_patch_size
                num_output_patches = (self.prediction_length + output_patch_size - 1) // output_patch_size
                
                # Forward pass
                output = self.model(
                    context=context_batch,
                    num_output_patches=num_output_patches
                )
                
                # Shape: (batch, quantiles, horizon)
                quantile_preds = output.quantile_preds.cpu().numpy()
                
                # Slice to exact prediction length
                quantile_preds = quantile_preds[:, :, :self.prediction_length]
            
            # Convert to GluonTS Forecast objects
            for j, (preds, ts) in enumerate(zip(quantile_preds, batch_data)):
                forecast_start_date = ts["start"] + len(ts["target"])
                
                forecast = QuantileForecast(
                    forecast_arrays=preds,
                    forecast_keys=list(map(str, self.quantile_levels)),
                    start_date=forecast_start_date,
                )
                forecasts.append(forecast)
        
        return forecasts


def evaluate_on_monash_dataset(
    predictor: Chronos2Predictor,
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on a Monash dataset from GluonTS repository.
    
    Parameters
    ----------
    predictor : Chronos2Predictor
        The predictor to evaluate
    dataset_name : str
        Name of the dataset (e.g., 'electricity', 'traffic', 'm4_hourly')
    max_samples : int, optional
        Maximum number of samples to evaluate (for quick testing)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metrics
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = get_dataset(dataset_name)
        test_data = list(dataset.test)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return None
    
    if max_samples and len(test_data) > max_samples:
        import random
        random.seed(42)
        test_data = random.sample(test_data, max_samples)
        logger.info(f"Subsampled to {max_samples} series")
    
    logger.info(f"Evaluating on {len(test_data)} series...")
    
    # Generate forecasts
    forecasts = predictor.predict(test_data)
    
    # Convert test data to time series for evaluation
    tss = []
    for entry in test_data:
        ts = pd.Series(
            entry["target"],
            index=pd.period_range(
                start=entry["start"],
                periods=len(entry["target"]),
                freq=entry["start"].freq
            )
        )
        tss.append(ts)
    
    # Evaluate
    evaluator = Evaluator(quantiles=predictor.quantile_levels)
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
    
    return agg_metrics


def evaluate_on_gift_eval_dataset(
    predictor: Chronos2Predictor,
    dataset_name: str,
    term: str = "short",
    use_multivariate: bool = True,
    metrics: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on a GIFT-Eval dataset.
    
    Parameters
    ----------
    predictor : Chronos2Predictor
        The predictor to evaluate
    dataset_name : str
        Name of the dataset (e.g., 'm4_weekly', 'electricity')
    term : str, optional
        Forecast term ('short', 'medium', 'long'), by default 'short'
    use_multivariate : bool, optional
        Whether to use multivariate data if available, by default True
    metrics : List, optional
        List of GluonTS metrics to compute
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metrics
    """
    try:
        from gift_eval.data import Dataset
        from gluonts.model import evaluate_forecasts
    except ImportError:
        raise ImportError(
            "gift_eval package is required. Install with: pip install gift-eval"
        )
    
    if metrics is None:
        from gluonts.ev.metrics import (
            MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
            MeanWeightedSumQuantileLoss,
        )
        metrics = [
            MSE(forecast_type="mean"),
            MSE(forecast_type=0.5),
            MAE(),
            MASE(),
            MAPE(),
            SMAPE(),
            MSIS(),
            RMSE(),
            NRMSE(),
            ND(),
            MeanWeightedSumQuantileLoss(quantile_levels=predictor.quantile_levels),
        ]
    
    # Load dataset
    is_multivariate_source = (
        Dataset(name=dataset_name, term=term, to_univariate=False).target_dim > 1
    )
    
    dataset = Dataset(
        name=dataset_name,
        term=term,
        to_univariate=is_multivariate_source and not use_multivariate,
    )
    
    logger.info(f"Dataset: {dataset_name}/{term}, Size: {len(dataset.test_data)}")
    
    # Avoid cross-batch leakage with rolling window evaluation
    forecast_windows = []
    n_windows = dataset.test_data.windows
    
    for window_idx in range(n_windows):
        entries_window_k = list(
            itertools.islice(dataset.test_data.input, window_idx, None, n_windows)
        )
        forecasts_window_k = list(predictor.predict(entries_window_k))
        forecast_windows.append(forecasts_window_k)
    
    # Interleave results
    forecasts = [item for items in zip(*forecast_windows) for item in items]
    
    # Evaluate
    season_length = get_seasonality(dataset.freq)
    
    return evaluate_forecasts(
        forecasts,
        test_data=dataset.test_data,
        metrics=metrics,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )


def run_benchmark(
    model: Chronos2Model,
    datasets: List[Dict[str, Any]],
    batch_size: int = 32,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run benchmark on multiple datasets.
    
    Parameters
    ----------
    model : Chronos2Model
        The model to benchmark
    datasets : List[Dict[str, Any]]
        List of dataset configurations, each dict should have:
        - 'name': dataset name
        - 'type': 'monash' or 'gift-eval'
        - 'prediction_length': forecast horizon
        - 'term': (for GIFT-Eval only) 'short', 'medium', or 'long'
        - 'max_samples': (optional) max samples to evaluate
    batch_size : int, optional
        Batch size for inference, by default 32
    device : str, optional
        Device to use, by default auto-detect
        
    Returns
    -------
    pd.DataFrame
        DataFrame with benchmark results
    """
    results = []
    
    for ds_config in datasets:
        name = ds_config['name']
        ds_type = ds_config.get('type', 'monash')
        pred_len = ds_config['prediction_length']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on {name} ({ds_type})")
        logger.info(f"{'='*60}")
        
        try:
            predictor = Chronos2Predictor(
                model=model,
                prediction_length=pred_len,
                batch_size=batch_size,
                device=device,
            )
            
            if ds_type == 'monash':
                metrics = evaluate_on_monash_dataset(
                    predictor=predictor,
                    dataset_name=name,
                    max_samples=ds_config.get('max_samples'),
                )
            elif ds_type == 'gift-eval':
                metrics = evaluate_on_gift_eval_dataset(
                    predictor=predictor,
                    dataset_name=name,
                    term=ds_config.get('term', 'short'),
                )
            else:
                logger.error(f"Unknown dataset type: {ds_type}")
                continue
            
            if metrics:
                metrics['dataset'] = name
                metrics['type'] = ds_type
                results.append(metrics)
                
        except Exception as e:
            logger.error(f"Failed to evaluate on {name}: {e}")
            continue
    
    return pd.DataFrame(results)
