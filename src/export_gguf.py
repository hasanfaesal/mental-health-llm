import os
# Set this BEFORE importing any torch or unsloth modules
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''

from unsloth import FastLanguageModel
import torch

# --- Configuration ---
# 1. Point to your fine-tuned adapter folder
model_path = "mental_health_llama_model/final_model" 

# 2. Decide where to save the GGUF file
gguf_filename = "mental_health_llama3_8b.gguf"

# 3. Choose your quantization method (Important!)
# Options:
#  - "q4_k_m" : Recommended. Balanced size/speed (approx 5GB file).
#  - "q8_0"   : Higher quality, larger size (approx 8GB).
#  - "f16"    : Full precision, largest size (approx 16GB). No quality loss.
quant_method = "q4_k_m"

print("🦥 Loading your model for export... this may take a minute.")

# --- Load the Model ---
# We load the model normally first. Unsloth handles merging the adapter automatically.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print(f"🚀 Starting export to GGUF format ({quant_method})...")

# --- Export to GGUF ---
# This single command handles the merging, conversion, and saving.
model.save_pretrained_gguf(
    "gguf_model",            # Folder where the file will be saved
    tokenizer,
    quantization_method = quant_method
)

print("\n✅ Export Complete!")
print(f"Your model is located in: ./gguf_model/{gguf_filename} (or similar named file)")