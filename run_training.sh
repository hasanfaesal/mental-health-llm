#!/bin/bash
# Training script launcher for Mental Health Llama model

echo "====================================="
echo "Mental Health Llama Training Setup"
echo "====================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv package manager not found. Please install uv first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Sync dependencies using uv (dependencies already installed)
echo "Ensuring dependencies are synced..."
uv sync


# Set required environment variables for Unsloth
echo "Setting environment variables for Unsloth..."
export PYTORCH_CUDA_ALLOC_CONF=""
export CUDA_LAUNCH_BLOCKING=1

# Check CUDA availability and driver status
echo "Checking CUDA availability..."
python3 -c "
import torch
import os
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('Running on CPU only')
"
# Check NVIDIA driver status
echo "Checking NVIDIA driver status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not found. NVIDIA drivers may not be properly installed."
fi

# Create output directory if it doesn't exist
mkdir -p mental_health_llama_model

echo "====================================="
echo "Starting training..."
echo "====================================="

# Run training with default parameters
python3 train_mental_health_llama.py \
    --model_name "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_path "mental_health_counseling_conversations/combined_dataset.json" \
    --output_dir "./mental_health_llama_model" \
    --max_steps 100 \
    --validation_split 0.1 \
    --log_level INFO

echo "====================================="
echo "Training completed!"
echo "====================================="
