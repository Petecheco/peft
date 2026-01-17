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

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_TIMELORA_TARGET_MODULES_MAPPING,
)

from .config import TimeLoraConfig
from .layer import TimeLoraLayer, TimeLoraLinear


class TimeLoraModel(BaseTuner):
    """
    Creates Time-dependent LoRA model from a pretrained transformers model.
    
    Time-dependent LoRA extends standard LoRA by adding lightweight time embeddings
    that are concatenated to the LoRA matrices, allowing the adapter to be conditioned
    on time information.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`TimeLoraConfig`]): The configuration of the TimeLora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The TimeLora model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import TimeLoraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = TimeLoraConfig(r=8, target_modules=["q_proj", "v_proj"], time_embedding_dim=1)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`TimeLoraConfig`]): The configuration of the TimeLora model.
    """

    prefix: str = "timelora_"
    tuner_layer_cls = TimeLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_TIMELORA_TARGET_MODULES_MAPPING

    def _check_new_adapter_config(self, config: TimeLoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.
        """
        super()._check_new_adapter_config(config)

    def _create_and_replace(
        self,
        timelora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = timelora_config.r
        lora_alpha = timelora_config.lora_alpha
        lora_dropout = timelora_config.lora_dropout
        time_embedding_dim = timelora_config.time_embedding_dim
        init_lora_weights = timelora_config.init_lora_weights

        kwargs = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "time_embedding_dim": time_embedding_dim,
            "init_lora_weights": init_lora_weights,
        }

        if isinstance(target, TimeLoraLinear):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(timelora_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(timelora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = TimeLoraLinear(target, adapter_name, **kwargs)
        else:
            raise ValueError(f"Unsupported layer type {type(target_base_layer)}")

        return new_module
