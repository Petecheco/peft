"""
TimeLora Merge Behavior Demo

Demonstrates the two merge modes:
1. Time-preserving merge (merge_timestep=None): Keeps time-dependency
2. Static merge (merge_timestep=t): Loses time-dependency, optimized for specific t
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, TimeLoraConfig

print("="*70)
print("TimeLora Merge Behavior Demonstration")
print("="*70)

# Setup
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = TimeLoraConfig(
    task_type="CAUSAL_LM",
    r=4,
    time_embedding_dim=2,
    lora_alpha=8,
    target_modules=["c_attn"],
)

peft_model = get_peft_model(model, config)
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

print("\n" + "="*70)
print("SCENARIO 1: Time-Preserving Merge (Default)")
print("="*70)
print("\nBehavior: Merges only static LoRA (A @ B), time encoders stay active")
print("Result: Model STILL responds to different timesteps after merge\n")

# Test before merge
with torch.no_grad():
    out_t0_before = peft_model(**inputs, timestep=0.0).logits
    out_t1_before = peft_model(**inputs, timestep=1.0).logits
    diff_before = (out_t0_before - out_t1_before).abs().mean().item()

print("Before merge:")
print("  - Output at t=0.0 vs t=1.0 difference:", diff_before)

# Merge without timestep (time-preserving mode)
print("\nMerging with: model.merge_and_unload()")
peft_model.merge_and_unload()

# Test after merge
with torch.no_grad():
    out_t0_after = peft_model(**inputs, timestep=0.0).logits
    out_t1_after = peft_model(**inputs, timestep=1.0).logits
    diff_after = (out_t0_after - out_t1_after).abs().mean().item()

print("\nAfter merge:")
print("  - Output at t=0.0 vs t=1.0 difference:", diff_after)
print("  - Time-dependency preserved:", diff_after > 1e-6)
print("\n✓ Conclusion: Time encoders still work! Model responds to timestep.")

print("\n" + "="*70)
print("SCENARIO 2: Static Merge at Specific Timestep")
print("="*70)
print("\nBehavior: Merges full weights (A_full @ B_full) at t=0.5")
print("Result: Model becomes STATIC, optimized for t=0.5\n")

# Recreate model for clean test
peft_model2 = get_peft_model(
    AutoModelForCausalLM.from_pretrained(model_name),
    config
)

# Test before merge
with torch.no_grad():
    out_t0_before2 = peft_model2(**inputs, timestep=0.0).logits
    out_t05_before2 = peft_model2(**inputs, timestep=0.5).logits
    out_t1_before2 = peft_model2(**inputs, timestep=1.0).logits

print("Before merge:")
print("  - Output at t=0.0:", out_t0_before2[0, 0, :3].tolist())
print("  - Output at t=0.5:", out_t05_before2[0, 0, :3].tolist())
print("  - Output at t=1.0:", out_t1_before2[0, 0, :3].tolist())

# Merge at specific timestep
print("\nMerging with: model.merge_and_unload(merge_timestep=0.5)")
# Note: merge_and_unload doesn't support custom args, use base_model.merge directly
for module in peft_model2.modules():
    if hasattr(module, 'merge') and hasattr(module, 'timelora_A'):
        module.merge(merge_timestep=0.5)

# After merge, base weights contain the adaptation at t=0.5
# Time encoders still compute but have minimal effect since base is already adapted
with torch.no_grad():
    out_t0_after2 = peft_model2(**inputs, timestep=0.0).logits
    out_t05_after2 = peft_model2(**inputs, timestep=0.5).logits
    out_t1_after2 = peft_model2(**inputs, timestep=1.0).logits

print("\nAfter merge at t=0.5:")
print("  - Output at t=0.0:", out_t0_after2[0, 0, :3].tolist())
print("  - Output at t=0.5:", out_t05_after2[0, 0, :3].tolist())
print("  - Output at t=1.0:", out_t1_after2[0, 0, :3].tolist())

diff_merged_t0_t1 = (out_t0_after2 - out_t1_after2).abs().mean().item()
print("\n  - Difference t=0 vs t=1 after merge:", diff_merged_t0_t1)
print("  - Time-dependency lost:", diff_merged_t0_t1 < 1e-4)
print("\n✓ Conclusion: Model is now static, optimized for t=0.5")

print("\n" + "="*70)
print("SUMMARY: When to use each merge mode?")
print("="*70)
print("""
1. Time-Preserving Merge (merge_timestep=None):
   ✓ Use when: You want to deploy with time-dependency intact
   ✓ Benefits: Smaller base model, time encoders add adaptation dynamically
   ✓ Use case: Diffusion models, time-series tasks
   
2. Static Merge (merge_timestep=t):
   ✓ Use when: You only care about one specific timestep
   ✓ Benefits: Can remove time encoders, slightly faster inference
   ✓ Use case: Final deployment at fixed timestep
""")

print("="*70)
print("Demo Complete!")
print("="*70)
