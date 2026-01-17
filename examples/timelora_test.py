"""
Simple test script for Time-dependent LoRA

This script demonstrates how to use TimeLora with a small model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import TimeLora
from peft import get_peft_model
from peft.tuners.timelora import TimeLoraConfig


def test_timelora_basic():
    """Test basic TimeLora functionality"""
    print("=" * 60)
    print("Testing Time-dependent LoRA")
    print("=" * 60)
    
    # Load a small model for testing
    model_name = "facebook/opt-125m"
    print(f"\n1. Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"   Original model parameters: {original_params:,}")
    
    # Create TimeLora config
    print("\n2. Creating TimeLora config")
    config = TimeLoraConfig(
        r=8,                      # LoRA rank
        lora_alpha=16,            # LoRA alpha
        target_modules=["q_proj", "v_proj"],  # Apply to attention Q and V projections
        lora_dropout=0.1,
        time_embedding_dim=1,     # Add 1 dimension for time embedding
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"   Config: r={config.r}, time_embedding_dim={config.time_embedding_dim}")
    print(f"   Effective rank: {config.r + config.time_embedding_dim}")
    
    # Apply TimeLora
    print("\n3. Applying TimeLora to model")
    model = get_peft_model(model, config)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    # Print model structure
    print("\n4. Model structure (first adapter layer):")
    model.print_trainable_parameters()
    
    # Test forward pass
    print("\n5. Testing forward pass")
    test_text = "Hello, this is a test for time-dependent LoRA."
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"   Input shape: {inputs['input_ids'].shape}")
        print(f"   Output shape: {outputs.logits.shape}")
        print("   ✓ Forward pass successful!")
    
    # Test parameter structure
    print("\n6. Checking TimeLora parameters")
    for name, param in model.named_parameters():
        if "timelora" in name or "time_embedding" in name:
            print(f"   {name}: {param.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_timelora_basic()
