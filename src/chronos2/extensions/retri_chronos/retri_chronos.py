# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Iterable, List, Tuple

import faiss
import numpy as np
import torch
from tqdm import tqdm

from chronos2.model import Chronos2Model
from chronos2.pipeline import Chronos2Pipeline

logger = logging.getLogger(__name__)


class TimeSeriesKnowledgeBase:
    """
    Knowledge Base for Time Series using FAISS for similarity search.
    Uses a frozen Chronos-2 encoder to generate embeddings.
    """

    def __init__(self, model: Chronos2Model, dimension: int = 768, index_type: str = "FlatIP"):
        """
        Initialize the Knowledge Base.

        Parameters
        ----------
        model : Chronos2Model
            The Chronos-2 model to use for encoding.
        dimension : int
            Dimension of the embeddings (default: 768 for Chronos-Base).
        index_type : str
            FAISS index type. "FlatIP" for Inner Product (cosine similarity if normalized),
            "FlatL2" for Euclidean distance.
        """
        self.model = model
        self.dimension = dimension
        
        # Initialize FAISS index
        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")
            
        self.stored_series = []  # To store the actual time series values

    @torch.no_grad()
    def encode(self, context: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of time series into dense vectors.
        Arguments are expected to be on the same device as the model.

        Parameters
        ----------
        context : torch.Tensor
            Batch of time series context (batch_size, context_length).
        batch_size : int
             Batch size for processing if context is huge (not fully used here, assuming caller batches).
             
        Returns
        -------
        np.ndarray
            Embeddings of shape (batch_size, dimension).
        """
        # Ensure model is in eval mode
        was_training = self.model.training
        self.model.eval()

        # Get device
        device = next(self.model.parameters()).device
        context = context.to(device)

        # Forward pass through encoder
        # model.encode returns (encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches)
        encoder_outputs, _, _, _ = self.model.encode(context=context)
        
        # last_hidden_state: (batch_size, num_patches + special_tokens, d_model)
        last_hidden_state = encoder_outputs.last_hidden_state
        
        # Mean pooling over the sequence dimension to get a single vector per series
        # Note: We should exclude [REG] token if it exists, but for simplicity assuming 
        # standard pooling over all output tokens is fine for now.
        
        embeddings = last_hidden_state.mean(dim=1)
        
        # Normalize embeddings for cosine similarity (IndexFlatIP)
        if isinstance(self.index, faiss.IndexFlatIP):
             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
             
        embeddings_np = embeddings.cpu().numpy().astype('float32')

        if was_training:
            self.model.train()
            
        return embeddings_np

    def build_index(self, dataset: Iterable[torch.Tensor], batch_size: int = 64):
        """
        Build the FAISS index from a dataset of time series.

        Parameters
        ----------
        dataset : Iterable[torch.Tensor]
            Iterable of time series tensors (1D or 2D).
            We assume each item is a single time series or a batch to be indexed.
            For simplicity, let's assume dataset yields batches or we batch them.
        batch_size : int
            Batch size for encoding.
        """
        logger.info("Building Knowledge Base Index...")
        
        buffer = []
        for series in tqdm(dataset):
            # Ensure series is 1D or 2D handled correctly. 
            # Assuming dataset yields (context_length,) tensors or similar.
            if isinstance(series, torch.Tensor):
                if series.ndim == 1:
                    series = series.unsqueeze(0) # (1, T)
                
                buffer.append(series)
                
                if len(buffer) >= batch_size:
                    batch = torch.cat(buffer, dim=0) # (B, T)
                    self._process_batch(batch)
                    buffer = []
        
        if buffer:
            batch = torch.cat(buffer, dim=0)
            self._process_batch(batch)
            
        logger.info(f"Index built with {self.index.ntotal} vectors.")

    def _process_batch(self, batch: torch.Tensor):
        # Store original series (CPU) for retrieval
        # In a real large-scale system, we'd store indices/IDs and look up efficiently.
        # Here we store values in memory as requested/planned.
        self.stored_series.extend([s.cpu() for s in batch])
        
        emb = self.encode(batch)
        self.index.add(emb)

    def retrieve(self, query: torch.Tensor, k: int = 5) -> Tuple[np.ndarray, List[torch.Tensor]]:
        """
        Retrieve k nearest neighbors for each query series.

        Parameters
        ----------
        query : torch.Tensor
            Query time series batch (B, T).
        k : int
            Number of neighbors to retrieve.

        Returns
        -------
        distances : np.ndarray
            Distances/Scores of shape (B, k).
        neighbors : List[torch.Tensor]
            List of length B, where each element is a Tensor of shape (k, T_ref) 
            containing the retrieved series.
        """
        emb = self.encode(query)
        D, I = self.index.search(emb, k)
        
        neighbors_batch = []
        for i in range(len(query)):
            indices = I[i]
            # Retrieve stored series
            retrieved = [self.stored_series[idx] for idx in indices if idx != -1]
            # Stack them
            if retrieved:
                # Assuming all stored series have same length or handled downstream
                # If lengths differ, we might need padding here.
                # For Chronos, usually we fix context length.
                neighbors_batch.append(torch.stack(retrieved)) 
            else:
                # Fallback if index empty or something weird
                neighbors_batch.append(torch.zeros(k, query.shape[-1]))

        return D, neighbors_batch


def augment_group_input(
    target_context: torch.Tensor, neighbor_contexts: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Restructure input batch to inject retrieved series into Group Attention.
    
    Parameters
    ----------
    target_context : torch.Tensor
        Batch of target series (batch_size, context_length).
    neighbor_contexts : List[torch.Tensor]
        List of length batch_size, each containing (k, context_length) tensors of neighbors.
        
    Returns
    -------
    augmented_context : torch.Tensor
        New batch with targets and neighbors flattened (batch_size * (1+k), context_length).
    group_ids : torch.Tensor
        Group IDs of shape (batch_size * (1+k),).
    """
    batch_size = target_context.shape[0]
    k = neighbor_contexts[0].shape[0] if neighbor_contexts else 0
    device = target_context.device
    
    augmented_list = []
    group_ids_list = []
    
    for i in range(batch_size):
        # Add target
        augmented_list.append(target_context[i])
        group_ids_list.append(i)
        
        # Add neighbors
        if k > 0:
            # neighbor_contexts[i] is (k, T). We assume it is on the same device or move it.
            neighbors = neighbor_contexts[i].to(device)
            for neighbor in neighbors:
                augmented_list.append(neighbor)
                group_ids_list.append(i)
                
    augmented_context = torch.stack(augmented_list)
    group_ids = torch.tensor(group_ids_list, device=device, dtype=torch.long)
    
    return augmented_context, group_ids


class RetriChronosPipeline(Chronos2Pipeline):
    """
    Retri-Chronos Pipeline that performs Retrieval-Augmented Forecasting.
    """
    
    def __init__(self, model: Chronos2Model, kb: TimeSeriesKnowledgeBase):
        super().__init__(model)
        self.kb = kb
        self._current_k = 0
        
    def predict(
        self,
        inputs,
        prediction_length: int | None = None,
        batch_size: int = 256,
        context_length: int | None = None,
        k: int = 2,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Predict with retrieval augmentation.
        
        Parameters
        ----------
        k : int
            Number of neighbors to retrieve.
        """
        self._current_k = k
        return super().predict(
             inputs, 
             prediction_length=prediction_length, 
             batch_size=batch_size, 
             context_length=context_length, 
             **kwargs
        )

    def _predict_batch(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor,
        unrolled_quantiles_tensor: torch.Tensor,
        prediction_length: int,
        max_output_patches: int,
        target_idx_ranges: list[tuple[int, int]],
    ) -> List[torch.Tensor]:
        k = getattr(self, '_current_k', 0)
        
        if k == 0:
            return super()._predict_batch(
                context, group_ids, future_covariates, unrolled_quantiles_tensor,
                prediction_length, max_output_patches, target_idx_ranges
            )
            
        # 1. RETRIEVAL
        # Retrieve neighbors for the current batch context
        # context is (B, T)
        # We perform retrieval on the same device where context currently resides (should be model device)
        _, neighbors = self.kb.retrieve(context, k=k)
        
        # 2. AUGMENTATION
        # Expand batch: B -> B*(k+1)
        aug_context, aug_group_ids = augment_group_input(context, neighbors)
        
        # We also need to augment future_covariates if they exist
        if future_covariates is not None:
             B_new = aug_context.shape[0]
             F_len = future_covariates.shape[-1]
             aug_covariates = torch.full((B_new, F_len), float('nan'), device=future_covariates.device)
             
             # Fill in the targets (every (k+1)-th row)
             indices = torch.arange(0, B_new, k+1, device=future_covariates.device)
             aug_covariates[indices] = future_covariates
        else:
            aug_covariates = None
            
        # 3. CALL PREDICTION LOGIC ON AUGMENTED BATCH
        # We essentially duplicate the logic from _predict_batch but using aug_context
        
        context_aug = aug_context.to(device=self.model.device, dtype=torch.float32)
        group_ids_aug = aug_group_ids.to(device=self.model.device)
        if aug_covariates is not None:
             future_covariates_aug = aug_covariates.to(device=self.model.device, dtype=torch.float32)
        else:
             future_covariates_aug = None

        def get_num_output_patches(remaining_horizon: int):
            num_output_patches = math.ceil(remaining_horizon / self.model_output_patch_size)
            num_output_patches = min(num_output_patches, max_output_patches)
            return num_output_patches

        predictions = []
        remaining = prediction_length

        # predict first set of patches
        prediction = self._predict_step(
            context=context_aug,
            group_ids=group_ids_aug,
            future_covariates=future_covariates_aug,
            num_output_patches=get_num_output_patches(remaining),
        )
        predictions.append(prediction)
        remaining -= prediction.shape[-1]

        # prepare inputs for long horizon prediction
        if remaining > 0:
            context_aug, group_ids_aug, future_covariates_aug, unrolled_sample_weights = (
                self._prepare_inputs_for_long_horizon_unrolling(
                    context=context_aug,
                    group_ids=group_ids_aug,
                    future_covariates=future_covariates_aug,
                    unrolled_quantiles=unrolled_quantiles_tensor,
                )
            )

        # long horizon heuristic
        while remaining > 0:
            prediction, context_aug, future_covariates_aug = self._autoregressive_unroll_for_long_horizon(
                context=context_aug,
                group_ids=group_ids_aug,
                future_covariates=future_covariates_aug,
                prediction=prediction,
                unrolled_quantiles=unrolled_quantiles_tensor,
                unrolled_sample_weights=unrolled_sample_weights,
                num_output_patches=get_num_output_patches(remaining),
            )
            predictions.append(prediction)
            remaining -= prediction.shape[-1]

        batch_prediction = torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        )
        
        # 4. FILTERING
        # batch_prediction shape: (B_aug, Quantiles, Time)
        indices_to_keep = torch.arange(0, batch_prediction.shape[0], k+1)
        batch_prediction = batch_prediction[indices_to_keep]
        
        return [batch_prediction[start:end] for (start, end) in target_idx_ranges]
