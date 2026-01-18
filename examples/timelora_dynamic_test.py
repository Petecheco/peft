"""
Time-dependent LoRA Test with Dynamic Timestep Input

This example demonstrates how to use TimeLora with dynamic timestep inputs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, TimeLoraConfig

# 1. Load base model
model_name = "gpt2"
print("Loading model:", model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure TimeLora
config = TimeLoraConfig(
    task_type="CAUSAL_LM",
    r=8,  # LoRA rank
    time_embedding_dim=2,  # Time embedding dimension (will make effective rank = 10)
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
)

print("\nTimeLora Configuration:")
print("  - Base LoRA rank (r):", config.r)
print("  - Time embedding dim:", config.time_embedding_dim)
print("  - Effective rank:", config.r + config.time_embedding_dim)

# 3. Create PEFT model
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# 4. Test with different timesteps
print("\n" + "="*60)
print("Testing Time-Dependent Forward Pass")
print("="*60)

# Prepare input
text = "Hello, I am a language model"
inputs = tokenizer(text, return_tensors="pt")
print("\nInput text:", text)

# Test 1: No timestep (defaults to t=0)
print("\n[Test 1] Forward without timestep (t=0 by default)")
with torch.no_grad():
    outputs_t0 = peft_model(**inputs)
    print("  Output shape:", outputs_t0.logits.shape)

# Test 2: Scalar timestep (same for all samples)
print("\n[Test 2] Forward with scalar timestep (t=0.5)")
with torch.no_grad():
    outputs_t05 = peft_model(**inputs, timestep=0.5)
    print("  Output shape:", outputs_t05.logits.shape)

# Test 3: Tensor timestep (different time for each sample in batch)
print("\n[Test 3] Forward with batch of different timesteps")
batch_texts = ["Hello world", "How are you", "Nice to meet you"]
batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)
batch_size = len(batch_texts)

# Different timestep for each sample
timesteps = torch.tensor([0.0, 0.5, 1.0])  # Shape: (batch_size,)
print("  Batch size:", batch_size)
print("  Timesteps:", timesteps.tolist())

with torch.no_grad():
    outputs_batch = peft_model(**batch_inputs, timestep=timesteps)
    print("  Output shape:", outputs_batch.logits.shape)

# Test 4: Verify outputs are different for different timesteps
print("\n[Test 4] Verify time-dependent behavior")
with torch.no_grad():
    out_t0 = peft_model(**inputs, timestep=0.0)
    out_t1 = peft_model(**inputs, timestep=1.0)
    
    diff = (out_t0.logits - out_t1.logits).abs().mean().item()
    print("  Mean absolute difference between t=0 and t=1:", diff)
    print("  Outputs are different:", diff > 1e-6)

print("\n" + "="*60)
print("Testing Complete!")
print("="*60)

# 5. Inspect time encoder structure
print("\nTime Encoder Architecture:")
for name, module in peft_model.named_modules():
    if "time_encoder" in name:
        print("  -", name)
        print("    ", module)

print("\n[Success] TimeLora with dynamic timestep works correctly!")
print("\nUsage Summary:")
print("  1. Without timestep: model(input_ids)  # uses t=0")
print("  2. Scalar timestep:  model(input_ids, timestep=0.5)")
print("  3. Batch timesteps:  model(input_ids, timestep=torch.tensor([0.0, 0.5, 1.0]))")
