#!/usr/bin/env python3
# Simple test for TimeLora

import sys
print("Python version:", sys.version)

# Test 1: Import
print("\n=== Test 1: Import TimeLora ===")
try:
    from peft.tuners.timelora import TimeLoraConfig, TimeLoraModel
    print("SUCCESS: TimeLora imported")
except Exception as e:
    print("FAILED:", e)
    sys.exit(1)

# Test 2: Create config
print("\n=== Test 2: Create Config ===")
try:
    config = TimeLoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        time_embedding_dim=1,
        task_type="CAUSAL_LM"
    )
    print("SUCCESS: Config created")
    print("  - r:", config.r)
    print("  - time_embedding_dim:", config.time_embedding_dim)
    print("  - Effective rank:", config.r + config.time_embedding_dim)
except Exception as e:
    print("FAILED:", e)
    sys.exit(1)

# Test 3: Test layer creation
print("\n=== Test 3: Create TimeLora Layer ===")
try:
    import torch
    import torch.nn as nn
    from peft.tuners.timelora import TimeLoraLinear
    
    base_layer = nn.Linear(10, 20)
    timelora_layer = TimeLoraLinear(
        base_layer=base_layer,
        adapter_name="default",
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        time_embedding_dim=1,
        init_lora_weights=True
    )
    print("SUCCESS: Layer created")
    print("  - Input features:", timelora_layer.in_features)
    print("  - Output features:", timelora_layer.out_features)
except Exception as e:
    print("FAILED:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Forward pass
print("\n=== Test 4: Forward Pass ===")
try:
    x = torch.randn(2, 10)
    output = timelora_layer(x)
    print("SUCCESS: Forward pass")
    print("  - Input shape:", x.shape)
    print("  - Output shape:", output.shape)
except Exception as e:
    print("FAILED:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check parameters
print("\n=== Test 5: Check Parameters ===")
try:
    params = dict(timelora_layer.named_parameters())
    print("SUCCESS: Parameters")
    for name, param in params.items():
        if "timelora" in name or "time_embedding" in name:
            print("  - " + name + ": " + str(tuple(param.shape)))
except Exception as e:
    print("FAILED:", e)
    sys.exit(1)

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
