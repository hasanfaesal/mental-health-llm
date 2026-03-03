#!/usr/bin/env python3
"""
Enhanced Mental Health Dataset Training Script for Llama-3-8B
=============================================================

This script fine-tunes the Llama-3-8B model on mental health counseling conversations
using QLoRA (Quantized Low-Rank Adaptation) with Unsloth for memory efficiency.

Features:
- Optimized for RTX A4000 (16GB) but adaptable to other GPUs
- Robust data validation and preprocessing
- Advanced training monitoring and checkpointing
- Memory-efficient training with gradient checkpointing
- Comprehensive error handling and logging

Author: Enhanced version for mental health counseling dataset
"""

import os

# Set this BEFORE importing any torch or unsloth modules
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime

# Import unsloth FIRST for optimizations
from unsloth import FastLanguageModel

# Core ML libraries (after unsloth)
import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer


# Setup comprehensive logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup detailed logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class ModelArguments:
    """Arguments for model configuration"""

    model_name_or_path: str = field(
        default="unsloth/llama-3-8b-bnb-4bit",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for model input"}
    )
    load_in_4bit: bool = field(
        default=True, metadata={"help": "Load model in 4-bit quantization"}
    )
    lora_r: int = field(
        default=32,  # Increased for better adaptation
        metadata={"help": "LoRA attention dimension"},
    )
    lora_alpha: int = field(
        default=64,  # Increased proportionally
        metadata={"help": "LoRA scaling parameter"},
    )
    lora_dropout: float = field(
        default=0.1,  # Added some dropout for regularization
        metadata={"help": "LoRA dropout rate"},
    )
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={"help": "Target modules for LoRA adaptation"},
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration"""

    dataset_path: str = field(
        default="data/combined_dataset.json",
        metadata={"help": "Path to the local JSON dataset"},
    )
    validation_split: float = field(
        default=0.1, metadata={"help": "Fraction of data to use for validation"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use (for testing)"},
    )


@dataclass
class TrainingArguments:
    """Enhanced training configuration"""

    output_dir: str = field(default="./mental_health_llama_model")
    per_device_train_batch_size: int = field(default=1)  # Reduced for memory efficiency
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(
        default=8
    )  # Increased to maintain effective batch size
    learning_rate: float = field(default=1e-4)  # Slightly reduced for stability
    num_train_epochs: int = field(default=3)
    max_steps: int = field(default=-1)  # Use epochs instead
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=200)
    evaluation_strategy: str = field(
        default="steps"
    )  # Fixed: was "steps" but needs to match validation data availability
    save_strategy: str = field(default="steps")
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)

    # Optimization settings
    optim: str = field(default="adamw_8bit")
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")

    # Precision settings
    fp16: bool = field(default=not torch.cuda.is_bf16_supported())
    bf16: bool = field(default=torch.cuda.is_bf16_supported())

    # Memory and performance
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=2)
    remove_unused_columns: bool = field(default=False)

    # Reproducibility
    seed: int = field(default=42)
    data_seed: int = field(default=42)

    # Monitoring
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    resume_from_checkpoint: Optional[str] = field(default=None)


def create_trainer(
    model,
    tokenizer,
    datasets: Dict[str, Dataset],
    training_args: TrainingArguments,
    model_args: ModelArguments,
) -> SFTTrainer:
    """Create and configure the SFT trainer"""

    logger.info("Initializing SFT Trainer...")

    # Adjust training arguments based on whether we have validation data
    has_eval_dataset = (
        "eval" in datasets
        and datasets["eval"] is not None
        and len(datasets["eval"]) > 0
    )

    if not has_eval_dataset:
        # If no validation data, disable evaluation and load_best_model_at_end
        eval_strategy = "no"
        save_strategy = "steps"  # Keep save strategy but disable load_best_model_at_end
        load_best_model_at_end = False
        eval_steps = None
        metric_for_best_model = None
        logger.info("No validation dataset found. Disabling evaluation.")
    else:
        # If we have validation data, use the configured evaluation strategy
        eval_strategy = training_args.evaluation_strategy
        save_strategy = training_args.save_strategy
        load_best_model_at_end = training_args.load_best_model_at_end
        eval_steps = training_args.eval_steps
        metric_for_best_model = training_args.metric_for_best_model
        logger.info(f"Using validation dataset with {len(datasets['eval'])} samples.")

    # Import the correct TrainingArguments from transformers
    from transformers import TrainingArguments as TransformersTrainingArguments

    # Build kwargs dict to handle optional parameters
    training_kwargs = {
        "output_dir": training_args.output_dir,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "num_train_epochs": training_args.num_train_epochs,
        "max_steps": training_args.max_steps,
        "warmup_steps": training_args.warmup_steps,
        "logging_steps": training_args.logging_steps,
        "save_steps": training_args.save_steps,
        "eval_strategy": eval_strategy,  # Changed from evaluation_strategy
        "save_strategy": save_strategy,
        "save_total_limit": training_args.save_total_limit,
        "load_best_model_at_end": load_best_model_at_end,
        "optim": training_args.optim,
        "weight_decay": training_args.weight_decay,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "fp16": training_args.fp16,
        "bf16": training_args.bf16,
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "dataloader_num_workers": training_args.dataloader_num_workers,
        "remove_unused_columns": training_args.remove_unused_columns,
        "seed": training_args.seed,
        "data_seed": training_args.data_seed,
        "report_to": training_args.report_to,
    }

    # Only add eval-related args if we have eval data
    if has_eval_dataset:
        training_kwargs["eval_steps"] = eval_steps
        training_kwargs["metric_for_best_model"] = metric_for_best_model
        training_kwargs["greater_is_better"] = training_args.greater_is_better

    # Only add resume_from_checkpoint if it's set
    if training_args.resume_from_checkpoint:
        training_kwargs["resume_from_checkpoint"] = training_args.resume_from_checkpoint

    # Convert our custom TrainingArguments to Transformers TrainingArguments
    transformers_args = TransformersTrainingArguments(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("eval") if has_eval_dataset else None,
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        dataset_num_proc=4,
        packing=False,  # Important for conversation format
        args=transformers_args,
    )

    return trainer


# =============================================================================
# Data Processing Functions
# =============================================================================


def validate_dataset(dataset_path: str) -> bool:
    """Validate the dataset file exists and has correct format"""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return False

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            # Try to read first few lines to validate JSON format
            for i, line in enumerate(f):
                if i >= 5:  # Check first 5 lines
                    break
                data = json.loads(line.strip())
                if "Context" not in data or "Response" not in data:
                    logger.error(f"Missing required fields in dataset line {i + 1}")
                    return False
        logger.info(f"Dataset validation successful: {dataset_path}")
        return True
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False


def create_prompt_template(system_message: str = None) -> str:
    """Create a comprehensive prompt template for mental health conversations"""
    if system_message is None:
        system_message = (
            "You are a compassionate and professional mental health assistant. "
            "Provide empathetic, supportive, and helpful responses while being "
            "mindful of ethical boundaries. Always encourage professional help "
            "when appropriate and never provide medical diagnoses."
        )

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{user_message}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{assistant_response}}<|eot_id|>"""


def format_mental_health_data(
    examples: Dict, tokenizer, prompt_template: str
) -> Dict[str, List[str]]:
    """
    Format mental health dataset for training

    Args:
        examples: Batch of examples from the dataset
        tokenizer: The tokenizer to use
        prompt_template: Template for formatting conversations

    Returns:
        Dict with 'text' key containing formatted conversations
    """
    contexts = examples["Context"]
    responses = examples["Response"]
    texts = []

    for context, response in zip(contexts, responses):
        # Clean and validate text
        context = context.strip() if isinstance(context, str) else str(context)
        response = response.strip() if isinstance(response, str) else str(response)

        if not context or not response:
            logger.warning("Empty context or response found, skipping...")
            continue

        # Format using the template
        formatted_text = prompt_template.format(
            user_message=context, assistant_response=response
        )

        texts.append(formatted_text)

    return {"text": texts}


def load_and_prepare_dataset(
    data_args: DataArguments, tokenizer, prompt_template: str
) -> Dict[str, Dataset]:
    """Load and prepare the mental health dataset"""

    # Validate dataset
    if not validate_dataset(data_args.dataset_path):
        raise FileNotFoundError(f"Dataset validation failed: {data_args.dataset_path}")

    logger.info(f"Loading dataset from: {data_args.dataset_path}")

    # Load dataset
    try:
        dataset = load_dataset("json", data_files=data_args.dataset_path, split="train")
        logger.info(f"Loaded {len(dataset)} samples from dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Apply max_samples limit if specified
    if data_args.max_samples and data_args.max_samples < len(dataset):
        dataset = dataset.select(range(data_args.max_samples))
        logger.info(f"Limited dataset to {data_args.max_samples} samples")

    # Split dataset if validation split is specified
    datasets = {}
    if data_args.validation_split > 0:
        split_dataset = dataset.train_test_split(
            test_size=data_args.validation_split, seed=42
        )
        datasets["train"] = split_dataset["train"]
        datasets["eval"] = split_dataset["test"]
        logger.info(
            f"Split dataset: {len(datasets['train'])} train, {len(datasets['eval'])} validation"
        )
    else:
        datasets["train"] = dataset
        logger.info("Using full dataset for training (no validation split)")

    # Format datasets
    for split_name, split_dataset in datasets.items():
        logger.info(f"Formatting {split_name} dataset...")
        datasets[split_name] = split_dataset.map(
            lambda x: format_mental_health_data(x, tokenizer, prompt_template),
            batched=True,
            remove_columns=split_dataset.column_names,
            desc=f"Formatting {split_name} dataset",
        )

    return datasets


# =============================================================================
# Model Training Functions
# =============================================================================


def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup the model and tokenizer with LoRA configuration"""

    logger.info(f"Loading model: {model_args.model_name_or_path}")

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=model_args.max_seq_length,
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=model_args.load_in_4bit,
        trust_remote_code=False,
    )

    # Configure LoRA
    logger.info("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=model_args.target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Print model info
    model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# Main Training Function
# =============================================================================


def main():
    """Main training function"""

    parser = argparse.ArgumentParser(
        description="Train Llama-3-8B on Mental Health Dataset"
    )
    parser.add_argument(
        "--model_name", default="unsloth/llama-3-8b-bnb-4bit", help="Model name or path"
    )
    parser.add_argument(
        "--dataset_path",
        default="data/combined_dataset.json",
        help="Dataset path",
    )
    parser.add_argument(
        "--output_dir", default="./mental_health_llama_model", help="Output directory"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum training steps (for quick testing)",
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum samples for testing"
    )
    parser.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Setup logging with specified level
    global logger
    logger = setup_logging(args.log_level)

    # Initialize configurations
    model_args = ModelArguments(model_name_or_path=args.model_name)
    data_args = DataArguments(
        dataset_path=args.dataset_path,
        validation_split=args.validation_split,
        max_samples=args.max_samples,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=3 if args.max_steps <= 0 else 1,
    )

    logger.info("=" * 80)
    logger.info("Mental Health Dataset Training - Starting")
    logger.info("=" * 80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_path}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"Max Steps: {training_args.max_steps}")
    logger.info(f"Validation Split: {data_args.validation_split}")

    try:
        # 1. Setup model and tokenizer
        logger.info("Step 1: Setting up model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(model_args)

        # 2. Create prompt template
        prompt_template = create_prompt_template()

        # 3. Load and prepare dataset
        logger.info("Step 2: Loading and preparing dataset...")
        datasets = load_and_prepare_dataset(data_args, tokenizer, prompt_template)

        # 4. Create trainer
        logger.info("Step 3: Creating trainer...")
        trainer = create_trainer(model, tokenizer, datasets, training_args, model_args)

        # 5. Start training
        logger.info("Step 4: Starting training...")
        logger.info("=" * 80)

        # Check for existing checkpoints
        if os.path.exists(training_args.output_dir):
            checkpoint_dirs = [
                d
                for d in os.listdir(training_args.output_dir)
                if d.startswith("checkpoint-")
            ]
            if checkpoint_dirs:
                latest_checkpoint = os.path.join(
                    training_args.output_dir, max(checkpoint_dirs)
                )
                logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                trainer.train(resume_from_checkpoint=latest_checkpoint)
            else:
                trainer.train()
        else:
            trainer.train()

        logger.info("Training completed successfully!")

        # 6. Save the final model
        logger.info("Step 5: Saving model...")
        final_output_dir = os.path.join(training_args.output_dir, "final_model")
        model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        logger.info(f"Model saved to: {final_output_dir}")

        # 7. Run inference test
        logger.info("Step 6: Running inference test...")
        test_prompt = (
            prompt_template.format(
                user_message="I've been feeling really anxious lately and can't sleep well.",
                assistant_response="",
            ).split("assistant<|end_header_id|>")[0]
            + "assistant<|end_header_id|>"
        )

        FastLanguageModel.for_inference(model)
        inputs = tokenizer([test_prompt], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test inference response: {response}")

        logger.info("=" * 80)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
