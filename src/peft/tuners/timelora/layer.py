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
    
    The time embeddings are concatenated to the LoRA A and B matrices, effectively
    increasing the rank from r to (r + time_embedding_dim).
    """
    
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "timelora_A",
        "timelora_B",
        "time_embedding_A",
        "time_embedding_B",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "lora_alpha",
        "scaling",
        "lora_dropout",
        "time_embedding_dim",
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
        
        # Time embedding vectors
        self.time_embedding_A = nn.ParameterDict({})
        self.time_embedding_B = nn.ParameterDict({})
        
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

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
        
        # Time embedding vectors: will be concatenated to form (time_embedding_dim, in_features) and (out_features, time_embedding_dim)
        self.time_embedding_A[adapter_name] = nn.Parameter(torch.empty(time_embedding_dim, self.in_features))
        self.time_embedding_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, time_embedding_dim))
        
        # Initialize weights
        self.reset_timelora_parameters(adapter_name, init_lora_weights)
        
        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_timelora_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        """Initialize the weights of TimeLora adapter"""
        if adapter_name not in self.timelora_A.keys():
            return

        if init_lora_weights:
            # Initialize A with Kaiming uniform (like standard LoRA)
            nn.init.kaiming_uniform_(self.timelora_A[adapter_name], a=math.sqrt(5))
            # Initialize B with zeros (like standard LoRA)
            nn.init.zeros_(self.timelora_B[adapter_name])
            
            # Initialize time embeddings
            nn.init.kaiming_uniform_(self.time_embedding_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.time_embedding_B[adapter_name])
        else:
            # Random initialization for debugging
            nn.init.kaiming_uniform_(self.timelora_A[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.timelora_B[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.time_embedding_A[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.time_embedding_B[adapter_name], a=math.sqrt(5))


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
        **kwargs,
    ) -> None:
        super().__init__()
        TimeLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, time_embedding_dim, init_lora_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.timelora_A.keys():
                base_layer = self.get_base_layer()
                
                # Get delta weight with time embeddings
                delta_weight = self.get_delta_weight(active_adapter)
                
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

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers from the base weights."""
        if not self.merged_adapters:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.timelora_A.keys():
                delta_weight = self.get_delta_weight(active_adapter)
                self.get_base_layer().weight.data = self.get_base_layer().weight.data - delta_weight

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.
        
        This concatenates the time embeddings to the LoRA matrices:
        A_full = [timelora_A; time_embedding_A]  # Shape: (r+time_embedding_dim, in_features)
        B_full = [timelora_B, time_embedding_B]  # Shape: (out_features, r+time_embedding_dim)
        
        Delta = B_full @ A_full * scaling
        """
        # Get the matrices
        A = self.timelora_A[adapter]  # (r, in_features)
        B = self.timelora_B[adapter]  # (out_features, r)
        time_A = self.time_embedding_A[adapter]  # (time_embedding_dim, in_features)
        time_B = self.time_embedding_B[adapter]  # (out_features, time_embedding_dim)
        
        # Concatenate time embeddings
        A_full = torch.cat([A, time_A], dim=0)  # (r+time_embedding_dim, in_features)
        B_full = torch.cat([B, time_B], dim=1)  # (out_features, r+time_embedding_dim)
        
        # Compute delta
        delta_weight = B_full @ A_full * self.scaling[adapter]
        
        return delta_weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass with time-dependent LoRA"""
        result = self.base_layer(x, *args, **kwargs)
        
        if self.disable_adapters:
            return result
        
        # Apply TimeLora adaptations
        for active_adapter in self.active_adapters:
            if active_adapter not in self.timelora_A.keys():
                continue
            
            # Get matrices
            A = self.timelora_A[active_adapter]
            B = self.timelora_B[active_adapter]
            time_A = self.time_embedding_A[active_adapter]
            time_B = self.time_embedding_B[active_adapter]
            
            # Concatenate to form full matrices
            A_full = torch.cat([A, time_A], dim=0)  # (r+time_embedding_dim, in_features)
            B_full = torch.cat([B, time_B], dim=1)  # (out_features, r+time_embedding_dim)
            
            # Apply dropout
            x_dropped = self.lora_dropout[active_adapter](x)
            
            # Compute: result + (x @ A_full.T @ B_full.T) * scaling
            result = result + (x_dropped @ A_full.T @ B_full.T) * self.scaling[active_adapter]
        
        return result
