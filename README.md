# Mental Health LLM

Fine-tuning LLaMA 3 8B for mental health counseling conversations using QLoRA and Unsloth.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model on HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/hasanfaesal/mental-health-llama3-8b-lora)
[![Base Model](https://img.shields.io/badge/Base-LLaMA%203%208B-green)](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)
[![Dataset](https://img.shields.io/badge/Dataset-Mental%20Health%20Counseling-orange)](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)

---

## Overview

This project fine-tunes Meta's LLaMA 3 8B model on real-world mental health counseling
conversations using parameter-efficient techniques. The goal is to produce a model that
generates empathetic, contextually aware responses to mental health concerns while
remaining small enough to run on consumer GPUs.

Key decisions:

- **QLoRA** (4-bit quantized LoRA) for memory-efficient fine-tuning on a single 16 GB GPU
- **Unsloth** for 2x faster training and reduced VRAM usage
- **LoRA rank 32** with alpha 64, targeting all attention and MLP projection layers for
  strong adaptation without full-parameter training

The trained LoRA adapter is available on [Hugging Face](https://huggingface.co/hasanfaesal/mental-health-llama3-8b-lora).

## Model Architecture

| Component | Details |
|---|---|
| Base Model | [unsloth/llama-3-8b-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit) |
| Method | QLoRA (4-bit NormalFloat quantization) |
| LoRA Rank (r) | 32 |
| LoRA Alpha | 64 |
| LoRA Dropout | 0.1 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Max Sequence Length | 2048 tokens |
| Optimizer | AdamW 8-bit |
| Adapter Size | ~281 MB (safetensors) |

## Training Results

Training was run for 100 steps (~25% of 1 epoch) on the full dataset of 3,512 Q&A pairs
with a 90/10 train/validation split.

| Step | Epoch | Training Loss | Gradient Norm | Learning Rate |
|------|-------|---------------|---------------|---------------|
| 10 | 0.025 | 2.9583 | 3.240 | 9.0e-06 |
| 20 | 0.051 | 2.8452 | 2.738 | 1.9e-05 |
| 30 | 0.076 | 2.3345 | 1.063 | 2.9e-05 |
| 50 | 0.127 | 2.1503 | 0.685 | 4.9e-05 |
| 70 | 0.177 | 2.1057 | 0.795 | 6.9e-05 |
| 100 | 0.253 | 2.0442 | 0.790 | 9.9e-05 |
| **Eval** | **0.253** | **2.0212** | -- | -- |

Loss decreased steadily from 2.96 to 2.04, with eval loss tracking closely at 2.02.
Gradient norms stabilized around 0.7-0.8, indicating stable training. The model could
benefit from additional training epochs.

## Quick Start

### Using the Trained Model

Load the LoRA adapter from Hugging Face and run inference:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="hasanfaesal/mental-health-llama3-8b-lora",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a compassionate and professional mental health assistant. Provide empathetic, supportive, and helpful responses while being mindful of ethical boundaries. Always encourage professional help when appropriate and never provide medical diagnoses.<|eot_id|><|start_header_id|>user<|end_header_id|>

I've been feeling really anxious lately and can't sleep well.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Without Unsloth

If you do not have Unsloth installed, you can load the adapter with PEFT directly:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=quantization_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "hasanfaesal/mental-health-llama3-8b-lora")
tokenizer = AutoTokenizer.from_pretrained("hasanfaesal/mental-health-llama3-8b-lora")
```

## Reproducing the Training

### Prerequisites

- Python 3.9+
- CUDA-capable GPU with at least 16 GB VRAM (tested on RTX A4000)
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Setup

```bash
git clone https://github.com/hasanfaesal/mental-health-llm.git
cd mental-health-llm

# Install dependencies
uv sync
# or: pip install -r requirements.txt

# Download the dataset
# See data/README.md for instructions, or download from:
# https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
# Place combined_dataset.json in the data/ directory.
```

### Train

```bash
# Quick test run (100 steps)
python src/train.py

# Full training (3 epochs)
python src/train.py --max_steps -1 --num_train_epochs 3

# Custom configuration
python src/train.py \
    --model_name "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_path "data/combined_dataset.json" \
    --output_dir "./mental_health_llama_model" \
    --max_steps 500 \
    --validation_split 0.15 \
    --log_level DEBUG
```

### Export to GGUF (for Ollama / llama.cpp)

```bash
python src/export_gguf.py
```

This merges the LoRA adapter with the base model and exports a quantized GGUF file
(default: `q4_k_m`, ~5 GB) suitable for local inference with Ollama or llama.cpp.

## Project Structure

```
mental-health-llm/
├── README.md                 # This file
├── LICENSE                   # MIT license (code)
├── pyproject.toml            # Project config and dependencies
├── requirements.txt          # Pip-compatible dependencies
├── uv.lock                   # Reproducible dependency resolution
│
├── src/                      # Primary source code
│   ├── train.py              # Main training script (QLoRA + Unsloth)
│   └── export_gguf.py        # GGUF export utility
│
├── scripts/                  # Alternative and utility scripts
│   ├── qlora_train.py        # Simpler training script variant
│   ├── qlora_unsloth.py      # Notebook-style Unsloth reference
│   ├── run_training.sh       # Shell training launcher
│   └── setup_uv.sh           # UV environment setup
│
└── data/                     # Dataset directory
    ├── README.md             # Download instructions
    └── LICENSE-RAIL-D.txt    # Dataset license
```

## Dataset

This project uses the [Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
dataset by Amod Sahasrabude:

- 3,512 real counseling Q&A pairs from licensed professionals
- 995 unique questions, each with multiple professional responses
- Topics: anxiety, depression, trauma, relationships, self-esteem, and more
- Licensed under RAIL-D (see `data/LICENSE-RAIL-D.txt`)

The dataset is not included in this repository. See `data/README.md` for download
instructions.

## Limitations and Ethical Considerations

This model is a research prototype and has significant limitations:

- **Not a substitute for professional care.** This model should never be used as a
  replacement for licensed mental health professionals. It cannot diagnose conditions,
  prescribe treatment, or handle crisis situations.
- **Limited training.** The model was trained for 100 steps on a relatively small dataset.
  Response quality will vary and may include inaccurate or inappropriate content.
- **No safety filters.** The model does not include built-in safety mechanisms for
  detecting or responding to crisis situations (e.g., suicidal ideation).
- **Bias.** The training data comes from specific counseling platforms and may not
  represent the full diversity of mental health experiences, cultural contexts, or
  therapeutic approaches.
- **Hallucination risk.** Like all language models, this model can generate plausible but
  factually incorrect or clinically inappropriate responses.

If you or someone you know is in crisis, please contact a professional service:
- **988 Suicide and Crisis Lifeline** (US): Call or text 988
- **Crisis Text Line**: Text HOME to 741741

## License

The project code is released under the [MIT License](LICENSE). The training dataset is
licensed separately under RAIL-D (see `data/LICENSE-RAIL-D.txt`).

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient fine-tuning
- [Amod Sahasrabude](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) for the counseling dataset
- [Meta AI](https://ai.meta.com/llama/) for the LLaMA 3 base model
