# Mental Health Llama Training

Fine-tune Llama-3-8B on Mental Health Counseling Dataset using QLoRA and Unsloth.

## 🚀 Quick Start with UV

This project is optimized for the `uv` package manager for faster dependency management.

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended: RTX A4000 or better)
- `uv` package manager

### Installation

#### Option 1: Using the setup script
```bash
./setup_uv.sh
```

#### Option 2: Manual setup with uv
```bash
# Install dependencies using pyproject.toml (recommended)
uv sync

# Or install from requirements.txt
uv pip install -r requirements.txt
```

#### Option 3: Legacy pip installation
```bash
./run_training.sh
```

### Training

#### Quick Training (100 steps for testing)
```bash
# Activate environment
source .venv/bin/activate

# Run training
python3 train_mental_health_llama.py
```

#### Full Training
```bash
python3 train_mental_health_llama.py \
    --max_steps -1 \
    --num_train_epochs 3 \
    --validation_split 0.1
```

#### Custom Configuration
```bash
python3 train_mental_health_llama.py \
    --model_name "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_path "mental_health_counseling_conversations/combined_dataset.json" \
    --output_dir "./my_custom_model" \
    --max_steps 500 \
    --validation_split 0.15 \
    --log_level DEBUG
```

## 📊 Performance Optimizations

- **Memory Efficient**: Optimized for 16GB GPUs
- **4-bit Quantization**: Using bitsandbytes for reduced memory usage
- **QLoRA**: Low-rank adaptation for parameter-efficient training
- **Gradient Checkpointing**: Reduces memory at cost of some compute
- **Optimized Batch Sizes**: Configured for RTX A4000

## 🔧 Configuration

### Model Arguments
- `--model_name`: Base model to fine-tune
- `--max_seq_length`: Maximum sequence length (default: 2048)

### Training Arguments
- `--max_steps`: Maximum training steps (-1 for epoch-based)
- `--validation_split`: Fraction for validation (default: 0.1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--output_dir`: Directory to save model

### Data Arguments
- `--dataset_path`: Path to JSON dataset
- `--max_samples`: Limit samples for testing

## 📁 Project Structure

```
mhealth/
├── train_mental_health_llama.py    # Main training script
├── qlora-train.py                  # Original training script
├── requirements.txt                # Pip dependencies
├── pyproject.toml                 # UV/modern Python config
├── setup_uv.sh                   # Fast UV setup
├── run_training.sh                # Training launcher
└── mental_health_counseling_conversations/
    ├── combined_dataset.json      # Training data
    └── README.md                  # Dataset documentation
```

## 🎯 Features

- **Robust Error Handling**: Comprehensive validation and recovery
- **Advanced Logging**: Detailed progress tracking with timestamps
- **Checkpoint Management**: Automatic saving and resuming
- **Validation Support**: Built-in train/validation splitting
- **Inference Testing**: Automatic model testing after training
- **Memory Optimization**: Configured for consumer GPUs

## 📈 Monitoring

The script supports multiple monitoring backends:
- **TensorBoard**: `tensorboard --logdir ./mental_health_llama_model/logs`
- **Weights & Biases**: Set `WANDB_PROJECT` environment variable

## 🔍 Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`

### Package Installation Issues
```bash
# Clean install with uv
uv cache clean
rm -rf .venv
uv venv
uv sync
```

### Dataset Issues
The script validates your dataset format. Ensure each line is valid JSON with `Context` and `Response` fields.

## 📝 License

This project follows the mental health dataset license requirements. See `mental_health_counseling_conversations/LICENSE-RAIL-D.txt` for details.
