# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class TimeLoraLayer(BaseTunerLayer):
    """
    Time-dependent LoRA layer that adds time embeddings to standard LoRA.
    
    The time embeddings are dynamically computed from a timestep input via a time encoder,
    then concatenated to the LoRA A and B matrices, effectively increasing the rank from r to (r + time_embedding_dim).
    
    Forward accepts a `timestep` argument (scalar or tensor) that encodes the current time.
    """
    
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "timelora_A",
        "timelora_B",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "lora_alpha",
        "scaling",
        "lora_dropout",
        "time_embedding_dim",
        "timelora_time_encoder_A",
        "timelora_time_encoder_B",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.time_embedding_dim = {}
        
        # Standard LoRA matrices
        self.timelora_A = nn.ParameterDict({})
        self.timelora_B = nn.ParameterDict({})
        
        # Time encoders: map scalar t -> embedding vector
        # timelora_time_encoder_A: (1) -> (time_embedding_dim, in_features)
        # timelora_time_encoder_B: (1) -> (out_features, time_embedding_dim)
        self.timelora_time_encoder_A = nn.ModuleDict({})
        self.timelora_time_encoder_B = nn.ModuleDict({})
        
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs
        # Store reference to parent model for timestep access
        self._timelora_model = None

        base_layer_mod = self.get_base_layer()
        if isinstance(base_layer_mod, nn.Linear):
            self.in_features, self.out_features = base_layer_mod.in_features, base_layer_mod.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer_mod)}")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        time_embedding_dim: int,
        init_lora_weights: bool,
        inference_mode: bool = False,
        dtype: str = "bf16",
        **kwargs: Any,
    ) -> None:
        """Internal function to create timelora adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            lora_alpha (`int`): Alpha for the added adapter.
            lora_dropout (`float`): The dropout probability for LoRA layers.
            time_embedding_dim (`int`): Dimension of time embeddings.
            init_lora_weights (`bool`): Whether to initialize weights.
            inference_mode (`bool`): Whether to set the adapter in inference mode (frozen).
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.time_embedding_dim[adapter_name] = time_embedding_dim
        
        # Scaling factor
        self.scaling[adapter_name] = lora_alpha / r
        
        # Dropout
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        
        # Standard LoRA matrices: A is (r, in_features), B is (out_features, r)
        self.timelora_A[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))
        self.timelora_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        
        # Time encoders: lightweight MLPs to map t (scalar) to embedding matrices
        # For A: t -> (time_embedding_dim * in_features)
        time_hidden_dim = 64  # Hidden dimension for time MLP
        dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.timelora_time_encoder_A[adapter_name] = nn.Sequential(
            nn.Linear(1, time_hidden_dim, dtype=dtype),
            nn.SiLU(),  # Smooth activation like in diffusion models
            nn.Linear(time_hidden_dim, time_embedding_dim * self.in_features, dtype=dtype),
        )
        
        # For B: t -> (out_features * time_embedding_dim)
        self.timelora_time_encoder_B[adapter_name] = nn.Sequential(
            nn.Linear(1, time_hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, self.out_features * time_embedding_dim, dtype=dtype),
        )
        
        # Initialize weights
        self.reset_timelora_parameters(adapter_name, init_lora_weights)
        
        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_timelora_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        """Initialize the weights of TimeLora adapter"""
        if adapter_name not in self.timelora_A.keys():
            return

        if init_lora_weights:
            # Initialize A with Kaiming uniform (like standard LoRA)
            nn.init.kaiming_uniform_(self.timelora_A[adapter_name], a=math.sqrt(5))
            # Initialize B with zeros (like standard LoRA)
            nn.init.zeros_(self.timelora_B[adapter_name])
            
            # Initialize time encoders with small weights (start with minimal time influence)
            for module in self.timelora_time_encoder_A[adapter_name]:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.01)
                    nn.init.zeros_(module.bias)
            
            for module in self.timelora_time_encoder_B[adapter_name]:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.01)
                    nn.init.zeros_(module.bias)
        else:
            # Random initialization for debugging
            nn.init.kaiming_uniform_(self.timelora_A[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.timelora_B[adapter_name], a=math.sqrt(5))
            
            for module in self.timelora_time_encoder_A[adapter_name]:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            
            for module in self.timelora_time_encoder_B[adapter_name]:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True) unless
        inference_mode is True.

        Args:
            adapter_names (`str` or `list[str]`):
                 The name(s) of the adapter(s) to set as active.
            inference_mode (bool, optional):
                 Whether the activated adapter should be frozen (i.e. `requires_grad=False`). Default is False.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter (if not in inference mode)
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if (key in adapter_names) and (not inference_mode):
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        # Handle time encoders separately (they are nn.Sequential modules, not parameters)
        for encoder_name in ["timelora_time_encoder_A", "timelora_time_encoder_B"]:
            if hasattr(self, encoder_name):
                encoder_dict = getattr(self, encoder_name)
                for key, encoder in encoder_dict.items():
                    if (key in adapter_names) and (not inference_mode):
                        # Enable gradients for all parameters in the encoder
                        for param in encoder.parameters():
                            param.requires_grad = True
                    else:
                        # Disable gradients for all parameters in the encoder
                        for param in encoder.parameters():
                            param.requires_grad = False

        self._active_adapter = adapter_names


class TimeLoraLinear(nn.Module, TimeLoraLayer):
    """TimeLora implemented in a dense layer"""
    
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        time_embedding_dim: int,
        init_lora_weights: bool = True,
        dtype: str = "bf16",
        **kwargs,
    ) -> None:
        super().__init__()
        TimeLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, time_embedding_dim, init_lora_weights,dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None, merge_timestep: Optional[float] = None) -> None:
        """
        Merge the active adapter weights into the base weights.
        
        IMPORTANT: TimeLora is time-dependent, so merging behavior depends on `merge_timestep`:
        - If `merge_timestep` is None: Only merges the static LoRA part (A @ B), keeps time encoders active.
          This allows the merged model to still respond to different timesteps dynamically.
        - If `merge_timestep` is provided: Merges the full weights at that specific timestep (A_full @ B_full).
          This creates a static merged model optimized for that timestep, losing time-dependency.

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
            merge_timestep (`float`, *optional*):
                The timestep to use for merging. If None, only static LoRA weights are merged and time encoders
                remain active. If provided, the full time-dependent weights at this timestep are merged.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.timelora_A.keys():
                base_layer = self.get_base_layer()
                
                if merge_timestep is None:
                    # Mode 1: Merge only static LoRA part, keep time encoders active
                    # This preserves time-dependency after merge
                    A = self.timelora_A[active_adapter]
                    B = self.timelora_B[active_adapter]
                    delta_weight = B @ A * self.scaling[active_adapter]
                    
                    warnings.warn(
                        f"TimeLora adapter '{active_adapter}' merged in time-preserving mode. "
                        f"The time encoders remain active and the model will still respond to timestep inputs. "
                        f"To merge for a specific timestep, use merge(merge_timestep=<value>)."
                    )
                else:
                    # Mode 2: Merge full weights at specific timestep (loses time-dependency)
                    timestep_tensor = torch.tensor(merge_timestep, device=base_layer.weight.device, dtype=base_layer.weight.dtype)
                    delta_weight = self.get_delta_weight(active_adapter, timestep=timestep_tensor)
                    
                    warnings.warn(
                        f"TimeLora adapter '{active_adapter}' merged at timestep={merge_timestep}. "
                        f"The merged model is now static and optimized for t={merge_timestep}. "
                        f"Time-dependency is lost after this merge."
                    )
                
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + delta_weight
                    
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + delta_weight
                
                self.merged_adapters.append(active_adapter)
                # Store merge info for proper unmerge
                if not hasattr(self, '_merge_info'):
                    self._merge_info = {}
                self._merge_info[active_adapter] = {'merge_timestep': merge_timestep}

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers from the base weights."""
        if not self.merged_adapters:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.timelora_A.keys():
                # Retrieve merge info to unmerge correctly
                merge_info = getattr(self, '_merge_info', {}).get(active_adapter, {})
                merge_timestep = merge_info.get('merge_timestep', None)
                
                if merge_timestep is None:
                    # Was merged in time-preserving mode (only static LoRA)
                    A = self.timelora_A[active_adapter]
                    B = self.timelora_B[active_adapter]
                    delta_weight = B @ A * self.scaling[active_adapter]
                else:
                    # Was merged at specific timestep (full weights)
                    timestep_tensor = torch.tensor(merge_timestep, device=self.get_base_layer().weight.device, dtype=self.get_base_layer().weight.dtype)
                    delta_weight = self.get_delta_weight(active_adapter, timestep=timestep_tensor)
                
                self.get_base_layer().weight.data = self.get_base_layer().weight.data - delta_weight
                
                # Clean up merge info
                if hasattr(self, '_merge_info') and active_adapter in self._merge_info:
                    del self._merge_info[active_adapter]

    def get_delta_weight(self, adapter: str, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.
        
        This concatenates the time embeddings (computed from timestep) to the LoRA matrices:
        A_full = [timelora_A; time_embedding_A(t)]  # Shape: (r+time_embedding_dim, in_features)
        B_full = [timelora_B, time_embedding_B(t)]  # Shape: (out_features, r+time_embedding_dim)
        
        Delta = B_full @ A_full * scaling
        
        Args:
            adapter: Name of the adapter
            timestep: Time value(s), shape (batch_size, 1) or scalar. If None, uses t=0.
        """
        # Get the matrices
        A = self.timelora_A[adapter]  # (r, in_features)
        B = self.timelora_B[adapter]  # (out_features, r)
        
        # Handle timestep
        if timestep is None:
            # Default to t=0
            timestep = torch.ones(1, 1, device=A.device, dtype=A.dtype)
        elif timestep.dim() == 0:
            # Scalar -> (1, 1)
            timestep = timestep.view(1, 1).to(A.device, dtype=A.dtype)
        elif timestep.dim() == 1:
            # (batch_size,) -> (batch_size, 1)
            timestep = timestep.unsqueeze(-1).to(A.device, dtype=A.dtype)
        
        # Encode time: for merge, we use the first timestep (or mean)
        t_input = timestep[:1]  # (1, 1)
        
        # Generate time embeddings
        time_emb_A = self.timelora_time_encoder_A[adapter](t_input)  # (1, time_embedding_dim * in_features)
        time_emb_A = time_emb_A.view(self.time_embedding_dim[adapter], self.in_features)  # (time_embedding_dim, in_features)
        
        time_emb_B = self.timelora_time_encoder_B[adapter](t_input)  # (1, out_features * time_embedding_dim)
        time_emb_B = time_emb_B.view(self.out_features, self.time_embedding_dim[adapter])  # (out_features, time_embedding_dim)
        
        # Concatenate time embeddings
        A_full = torch.cat([A, time_emb_A], dim=0)  # (r+time_embedding_dim, in_features)
        B_full = torch.cat([B, time_emb_B], dim=1)  # (out_features, r+time_embedding_dim)
        
        # Compute delta
        delta_weight = B_full @ A_full * self.scaling[adapter]
        
        return delta_weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass with time-dependent LoRA.
        
        Args:
            x: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments. Can include 'timestep' for time-dependent adaptation.
                     timestep can be:
                     - scalar (float): same time for all samples
                     - tensor of shape (batch_size,) or (batch_size, 1): per-sample timestep
        
        Returns:
            Output tensor with TimeLora adaptation applied
        
        Note:
            - If adapters are merged in time-preserving mode (merge_timestep=None), time encoders
              still work and the model responds to timestep inputs.
            - If merged at specific timestep, the static merged weights are used and time encoders
              have no additional effect (but still compute, which is wasteful - consider unmerging).
        """
        # Extract timestep: Priority kwargs > model storage > None
        timestep = kwargs.pop('timestep', None)
        if timestep is None and hasattr(self, '_timelora_model') and self._timelora_model is not None:
            timestep = getattr(self._timelora_model, '_timestep_storage', None)
        
        print("[DEBUG] timestep=", timestep)
        breakpoint()
        result = self.base_layer(x, *args, **kwargs)
        
        if self.disable_adapters:
            return result
        
        print("[DEBUG] Processing TimeLora with timestep:", timestep)
        
        # Apply TimeLora adaptations
        for active_adapter in self.active_adapters:
            if active_adapter not in self.timelora_A.keys():
                continue
            
            # Get matrices
            A = self.timelora_A[active_adapter]  # (r, in_features)
            B = self.timelora_B[active_adapter]  # (out_features, r)
            
            # Handle timestep
            if timestep is None:
                # Default to t=0 for all samples
                batch_size = x.shape[0]
                t = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
            elif isinstance(timestep, (int, float)):
                # Scalar: broadcast to all samples
                batch_size = x.shape[0]
                t = torch.full((batch_size, 1), timestep, device=x.device, dtype=x.dtype)
            else:
                # Tensor: ensure shape (batch_size, 1)
                t = timestep
                if t.dim() == 0:
                    t = t.view(1, 1).expand(x.shape[0], 1)
                elif t.dim() == 1:
                    t = t.unsqueeze(-1)
                t = t.to(x.device, dtype=x.dtype)
            # Generate time embeddings for the batch
            time_emb_A = self.timelora_time_encoder_A[active_adapter](t).to(x.dtype)  # (batch_size, time_embedding_dim * in_features)
            time_emb_A = time_emb_A.view(t.shape[0], self.time_embedding_dim[active_adapter], self.in_features)  # (batch_size, time_embedding_dim, in_features)
            
            time_emb_B = self.timelora_time_encoder_B[active_adapter](t).to(x.dtype)  # (batch_size, out_features * time_embedding_dim)
            time_emb_B = time_emb_B.view(t.shape[0], self.out_features, self.time_embedding_dim[active_adapter])  # (batch_size, out_features, time_embedding_dim)
            
            # Expand A and B for batch dimension
            A_expanded = A.unsqueeze(0).expand(t.shape[0], -1, -1)  # (batch_size, r, in_features)
            B_expanded = B.unsqueeze(0).expand(t.shape[0], -1, -1)  # (batch_size, out_features, r)
            
            # Concatenate to form full matrices per sample
            A_full = torch.cat([A_expanded, time_emb_A], dim=1)  # (batch_size, r+time_embedding_dim, in_features)
            B_full = torch.cat([B_expanded, time_emb_B], dim=2)  # (batch_size, out_features, r+time_embedding_dim)
            
            # Apply dropout
            x_dropped = self.lora_dropout[active_adapter](x)  # (batch_size, ..., in_features)
            
            # Reshape for batch matrix multiplication
            original_shape = x_dropped.shape
            if x_dropped.dim() > 2:
                # Flatten all dimensions except last
                x_dropped = x_dropped.view(-1, self.in_features)  # (batch_size * seq_len, in_features)
                # Expand A_full and B_full to match
                A_full = A_full.repeat_interleave(original_shape[1] if len(original_shape) > 2 else 1, dim=0)
                B_full = B_full.repeat_interleave(original_shape[1] if len(original_shape) > 2 else 1, dim=0)
            
            # Compute: (x @ A_full.T @ B_full.T) * scaling
            # (batch_size * seq_len, in_features) @ (batch_size * seq_len, in_features, r+time_embedding_dim)
            lora_out = torch.bmm(
                x_dropped.unsqueeze(1),  # (batch_size * seq_len, 1, in_features)
                A_full.transpose(1, 2)   # (batch_size * seq_len, in_features, r+time_embedding_dim)
            )  # (batch_size * seq_len, 1, r+time_embedding_dim)
            
            lora_out = torch.bmm(
                lora_out,  # (batch_size * seq_len, 1, r+time_embedding_dim)
                B_full.transpose(1, 2)  # (batch_size * seq_len, r+time_embedding_dim, out_features)
            ).squeeze(1)  # (batch_size * seq_len, out_features)
            
            # Reshape back if needed
            if len(original_shape) > 2:
                lora_out = lora_out.view(*original_shape[:-1], self.out_features)
            
            result = result + lora_out * self.scaling[active_adapter]
        
        return result
