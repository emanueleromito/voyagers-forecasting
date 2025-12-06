import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from chronos2.model import Chronos2Model
from chronos2.layers import TimeSelfAttention
from chronos2.pipeline import Chronos2Pipeline


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
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
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
        for _ in range(n_gradient_steps):
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
        )
        
        return forecast
