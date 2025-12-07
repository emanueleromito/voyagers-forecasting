import math
import warnings
from typing import Optional, Sequence, Mapping, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chronos2.model import Chronos2Model
from chronos2.layers import TimeSelfAttention
from chronos2.pipeline import Chronos2Pipeline
from chronos2.dataset import Chronos2Dataset, DatasetMode, TensorOrArray


class LoRALayer(nn.Module):
    """
    Wraps a linear layer to add Low-Rank Adaptation (LoRA).
    W' = W + alpha * (B @ A)
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        # A: (rank, in_dim) - initialized with Kaiming Uniform
        # B: (out_dim, rank) - initialized to zeros
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        
        # Use same dtype and device as original layer
        dtype = original_layer.weight.dtype
        device = original_layer.weight.device
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank, dtype=dtype, device=device))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with Kaiming Uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B to zeros (so adaptation starts as identity)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_out = self.original_layer(x)
        
        # LoRA path: x @ A.T @ B.T * scaling
        # (batch, ..., in) @ (in, rank) @ (rank, out)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_out + lora_out


class ChronosPETSAWrapper(nn.Module):
    """
    Wraps a Chronos2Model to implement Parameter-Efficient Test-Time Adaptation (PETSA).
    Injects LoRA into TimeSelfAttention layers and provides an adapt_and_forecast method.
    """
    def __init__(self, model: Chronos2Model, lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Freeze the entire base model first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Inject LoRA layers
        self._inject_lora()
        
    def _inject_lora(self):
        """
        Iterate through the model and replace q, k, v projections in TimeSelfAttention
        with LoRALayer wrappers.
        """
        # Chronos2EncoderBlock contains TimeSelfAttention at layer[0]
        for block in self.model.encoder.block:
            # Check if layer[0] is indeed TimeSelfAttention
            if isinstance(block.layer[0], TimeSelfAttention):
                time_attn = block.layer[0].self_attention
                
                # Wrap q, k, v
                time_attn.q = LoRALayer(time_attn.q, rank=self.lora_rank, alpha=self.lora_alpha)
                time_attn.k = LoRALayer(time_attn.k, rank=self.lora_rank, alpha=self.lora_alpha)
                time_attn.v = LoRALayer(time_attn.v, rank=self.lora_rank, alpha=self.lora_alpha)
                
    def reset_lora(self):
        """
        Reset LoRA weights to their initial state (B=0), effectively removing adaptation effects.
        """
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.reset_parameters()

    def adapt_and_forecast(
        self,
        context: torch.Tensor,
        prediction_length: int,
        n_gradient_steps: int = 5,
        learning_rate: float = 1e-3,
        mask_ratio: float = 0.25,
        context_mask: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        future_covariates_mask: Optional[torch.Tensor] = None,
        sparse_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Performs PETSA:
        1. Masks the end of the context.
        2. Adapts LoRA weights to reconstruct the masked part.
        3. Forecasts the future using adapted weights.
        4. Resets LoRA weights.
        
        Args:
            context: (batch_size, context_length)
            prediction_length: int
            n_gradient_steps: int
            learning_rate: float
            mask_ratio: float, fraction of context to mask for adaptation
            context_mask: Optional mask for context
            future_covariates: Optional (batch_size, prediction_length)
            future_covariates_mask: Optional mask for future_covariates
            
        Returns:
            forecast: (batch_size, num_quantiles, prediction_length)
        """
        self.model.eval() 
        
        # Identify trainable parameters (only LoRA A and B)
        lora_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(lora_params, lr=learning_rate)
        
        # --- Step 1: Self-Supervised Masking (The Pretext Task) ---
        batch_size, context_len = context.shape
        
        # Determine mask length
        mask_len = int(context_len * mask_ratio)
        if mask_len < 1:
            mask_len = 1
            
        # Create masked context for adaptation
        adaptation_context = context[:, :-mask_len]
        adaptation_target = context[:, -mask_len:]
        
        if context_mask is not None:
            adaptation_context_mask = context_mask[:, :-mask_len]
            adaptation_target_mask = context_mask[:, -mask_len:]
        else:
            adaptation_context_mask = None
            adaptation_target_mask = None
            
        patch_size = self.model.chronos_config.output_patch_size
        num_adaptation_patches = math.ceil(mask_len / patch_size)
        
        # --- Step 2: Adaptation Loop ---
        with torch.enable_grad():
            for step in range(n_gradient_steps):
                optimizer.zero_grad()
                
                # Forward pass to compute loss on the masked portion
                output = self.model(
                    context=adaptation_context,
                    context_mask=adaptation_context_mask,
                    future_target=adaptation_target,
                    future_target_mask=adaptation_target_mask,
                    num_output_patches=num_adaptation_patches
                )
                
                loss = output.loss
                loss.backward()
                
                # Sparse Update Logic (Fisher Approximation Probe)
                if step == 0 and sparse_ratio > 0.0 and n_gradient_steps > 1:
                    # Calculate gradient norms for all trainable LoRA parameters
                    param_grads = []
                    for name, param in self.named_parameters():
                        if param.requires_grad and param.grad is not None and "lora" in name:
                            grad_norm = param.grad.norm().item()
                            param_grads.append((name, param, grad_norm))
                    
                    # Sort by gradient norm (descending)
                    param_grads.sort(key=lambda x: x[2], reverse=True)
                    
                    # Determine cutoff for top parameters
                    num_params_to_keep = max(1, int(len(param_grads) * (1.0 - sparse_ratio)))
                    params_to_keep = set([x[0] for x in param_grads[:num_params_to_keep]])
                    
                    # Freeze parameters not in the top set
                    for name, param, _ in param_grads:
                        if name not in params_to_keep:
                            param.requires_grad = False
                            param.grad = None # Create space for optimization
                
                optimizer.step()
            
        # --- Step 3: Forecasting ---
        with torch.no_grad():
            # Calculate required output patches for the requested prediction length
            num_forecast_patches = math.ceil(prediction_length / patch_size)
            
            forecast_output = self.model(
                context=context,
                context_mask=context_mask,
                num_output_patches=num_forecast_patches,
                future_covariates=future_covariates,
                future_covariates_mask=future_covariates_mask
            )
            
            # Extract predictions and slice to exact prediction length
            # shape: (batch, quantiles, patches * patch_size)
            forecast = forecast_output.quantile_preds
            forecast = forecast[..., :prediction_length]
            
        # --- Step 4: Restoration ---
        self.reset_lora()
        
        return forecast


class ChronosPETSAPipeline(Chronos2Pipeline):
    def __init__(self, model: ChronosPETSAWrapper):
        super().__init__(model=model.model)
        self.wrapper = model
        
    def predict(
        self,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]]],
        prediction_length: int | None = None,
        batch_size: int = 256,
        context_length: int | None = None,
        predict_batches_jointly: bool = False,
        limit_prediction_length: bool = False,
        **kwargs,
    ) -> list[torch.Tensor]:
        """
        Generate forecasts for the given time series.
        Overridden to remove @torch.no_grad() decorator for PETSA.
        """
        model_prediction_length = self.model_prediction_length
        if prediction_length is None:
            prediction_length = model_prediction_length

        max_output_patches = kwargs.pop("max_output_patches", self.max_output_patches)
        unrolled_quantiles = kwargs.pop("unrolled_quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Capture sparse_ratio for use in _predict_step
        self._sparse_ratio = kwargs.pop("sparse_ratio", 0.0)

        if len(kwargs) > 0:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}.")

        if not set(unrolled_quantiles).issubset(self.quantiles):
            raise ValueError(
                f"Unrolled quantiles must be a subset of the model's quantiles. "
                f"Found: {unrolled_quantiles=}, model_quantiles={self.quantiles}"
            )
        unrolled_quantiles_tensor = torch.tensor(unrolled_quantiles)

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        if context_length is None:
            context_length = self.model_context_length

        if context_length > self.model_context_length:
            warnings.warn(
                f"The specified context_length {context_length} is greater than the model's default context length {self.model_context_length}. "
                f"Resetting context_length to {self.model_context_length}."
            )
            context_length = self.model_context_length

        test_dataset = Chronos2Dataset.convert_inputs(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            mode=DatasetMode.TEST,
        )
        test_loader = DataLoader(test_dataset, batch_size=None, pin_memory=True, shuffle=False, drop_last=False)

        all_predictions: list[torch.Tensor] = []
        for batch in test_loader:
            assert batch["future_target"] is None
            batch_context = batch["context"]
            batch_group_ids = batch["group_ids"]
            batch_future_covariates = batch["future_covariates"]
            batch_target_idx_ranges = batch["target_idx_ranges"]

            if predict_batches_jointly:
                batch_group_ids = torch.zeros_like(batch_group_ids)

            batch_prediction = self._predict_batch(
                context=batch_context,
                group_ids=batch_group_ids,
                future_covariates=batch_future_covariates,
                unrolled_quantiles_tensor=unrolled_quantiles_tensor,
                prediction_length=prediction_length,
                max_output_patches=max_output_patches,
                target_idx_ranges=batch_target_idx_ranges,
            )
            all_predictions.extend(batch_prediction)

        return all_predictions

    def _predict_step(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor | None,
        num_output_patches: int,
    ) -> torch.Tensor:
        """
        Override _predict_step to use PETSA adaptation and forecasting.
        """
        # Calculate prediction length from patches
        prediction_length = num_output_patches * self.model_output_patch_size
        
        # Use the wrapper to adapt and forecast
        forecast = self.wrapper.adapt_and_forecast(
            context=context,
            prediction_length=prediction_length,
            future_covariates=future_covariates,
            sparse_ratio=getattr(self, "_sparse_ratio", 0.0),
        )
        
        return forecast
