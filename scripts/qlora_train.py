# This source code is optimized for RTX A4000 (16GB) using Unsloth
# Reference: Adapted from Artidoro/qlora.py for specific Mental Health Dataset

import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration Classes (Mimicking qlora.py structure) ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="unsloth/llama-3-8b-bnb-4bit")
    max_seq_length: int = field(default=2048)
    load_in_4bit: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0)


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="data/combined_dataset.json",
        metadata={"help": "Path to the local JSON dataset"},
    )


@dataclass
class TrainingConfigs(TrainingArguments):
    output_dir: str = field(default="./mental_health_model")
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    max_steps: int = field(default=60)  # Set to 500+ for full training
    logging_steps: int = field(default=1)
    optim: str = field(default="adamw_8bit")
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="linear")
    seed: int = field(default=3407)
    fp16: bool = field(default=not torch.cuda.is_bf16_supported())
    bf16: bool = field(default=torch.cuda.is_bf16_supported())
    # Enable checkpointing
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=10)
    save_total_limit: int = field(default=3)
    resume_from_checkpoint: bool = field(default=True)


# --- Prompt Formatting ---
# Defining the "Chat" template for the model
PROMPT_TEMPLATE = """Below is a conversation between a user and a helpful mental health assistant.

### User:
{}

### Assistant:
{}"""


def format_mental_health_data(examples, tokenizer):
    """
    Maps the raw dataset columns (Context/Response) to the training format.
    Analogous to 'make_data_module' in qlora.py
    """
    inputs = examples["Context"]
    outputs = examples["Response"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        # Must add EOS token so the model learns where to stop
        text = PROMPT_TEMPLATE.format(input_text, output_text) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}


# --- Main Training Function ---
def train():
    # 1. Initialize Arguments
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingConfigs(
        output_dir=model_args.model_name_or_path.split("/")[-1] + "-finetuned"
    )

    logger.info(f"Loading Model: {model_args.model_name_or_path}")
    logger.info(f"Loading Data from: {data_args.dataset_path}")

    # 2. Load Model & Tokenizer with explicit dtype
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=model_args.max_seq_length,
        dtype=None,  # Let Unsloth auto-detect
        load_in_4bit=model_args.load_in_4bit,
        # trust_remote_code = True,  # Remove this line
    )

    # 3. Attach LoRA Adapters
    # Replaces 'find_all_linear_names' and 'get_peft_model' in qlora.py
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_args.seed,
    )

    # 4. Load and Process Dataset
    # Replaces 'make_data_module' in qlora.py
    if not os.path.exists(data_args.dataset_path):
        raise FileNotFoundError(f"Could not find dataset at {data_args.dataset_path}")

    dataset = load_dataset("json", data_files=data_args.dataset_path, split="train")
    dataset = dataset.map(
        lambda x: format_mental_health_data(x, tokenizer), batched=True
    )

    # 5. Initialize Trainer
    # SFTTrainer handles the 'DataCollator' logic internally
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # 6. Train
    logger.info("*** Starting Training ***")
    trainer_stats = trainer.train()
    logger.info("*** Training Completed ***")

    # 7. Save the Model
    # Replaces 'SavePeftModelCallback' in qlora.py
    logger.info(f"Saving model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # 8. Inference Test (Sanity Check)
    logger.info("*** Running Inference Test ***")
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
        [
            PROMPT_TEMPLATE.format(
                "I feel really anxious when I have to go to work.",
                "",  # Output generation start
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=128)


if __name__ == "__main__":
    train()
