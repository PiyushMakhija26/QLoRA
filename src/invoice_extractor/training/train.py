import os
from dataclasses import asdict
from typing import Any

import torch
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from invoice_extractor.config import Config
from invoice_extractor.data.loader import prepare_dataset
from invoice_extractor.utils.logging import setup_logger
from invoice_extractor.utils.seed import set_seed

logger = setup_logger(__name__)


def setup_wandb(cfg: Config) -> None:
    """Configures Weights & Biases environment variables and initialization."""
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.wandb.entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    os.environ["WANDB_MODE"] = cfg.wandb.mode

    config_dict = asdict(cfg)
    wandb.init(
        project=cfg.wandb.project,
        config=config_dict,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else [],
    )
    # Log git commit if available
    try:
        import subprocess

        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        wandb.config.update({"git_commit": git_hash}, allow_val_change=True)  # type: ignore[no-untyped-call]
    except Exception:
        logger.warning("Could not log git commit hash.")


def load_model_and_tokenizer(cfg: Config) -> tuple[Any, Any]:
    """Loads base model and tokenizer with GPU check and quantization fallbacks."""
    model_id = cfg.model.name

    # Setup data type
    if cfg.model.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif cfg.model.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cuda_available = torch.cuda.is_available()

    # Quantization Setup
    bnb_config = None
    if cfg.model.quantization.load_in_4bit and cuda_available:
        try:
            if cfg.model.quantization.bnb_4bit_compute_dtype == "bfloat16":
                compute_dtype = torch.bfloat16
            elif cfg.model.quantization.bnb_4bit_compute_dtype == "float16":
                compute_dtype = torch.float16
            else:
                compute_dtype = torch.float32

            bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            logger.info("Initializing 4-bit Quantization (NF4)...")
        except Exception as e:
            logger.warning(
                f"Failed to configure bitsandbytes: {e}. Falling back to full model resolution load."
            )
            bnb_config = None

    device_map = "auto" if cuda_available else None
    logger.info(f"Loading model: {model_id} (Dtype: {torch_dtype}, Device Map: {device_map})")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except Exception as e:
        if bnb_config is not None:
            logger.warning(f"Error loading quantized model: {e}. Retrying without bitsandbytes...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
            )
        else:
            raise e

    return model, tokenizer


def configure_peft(model: Any, cfg: Config) -> Any:
    """Wraps base model in a LoRA/PEFT adapter container."""
    if not cfg.model.use_peft:
        logger.info("LoRA training disabled in configuration. Tuning all parameters.")
        return model

    logger.info("Configuring PEFT LoRA adapter...")

    # Prepare model for 4-bit/8-bit training
    is_quantized = getattr(model, "is_quantized", False)
    if is_quantized or cfg.model.quantization.load_in_4bit:
        try:
            model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
                model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
            )
            logger.info("Prepared model for kbit training.")
        except Exception as e:
            logger.warning(f"Could not run prepare_model_for_kbit_training: {e}")

    lora_cfg = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        target_modules=list(cfg.model.lora.target_modules),
        lora_dropout=cfg.model.lora.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def train_lora(cfg: Config) -> None:
    """Executes the complete QLoRA training pipeline."""
    # 1. Reproducibility
    set_seed(cfg.seed)

    # 2. Setup Experiment Tracker
    setup_wandb(cfg)

    # 3. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 4. Configure PEFT/LoRA adapter
    model = configure_peft(model, cfg)

    # 5. Prepare datasets
    logger.info("Loading training and validation datasets...")
    train_dataset = prepare_dataset(cfg, tokenizer, split="train")
    val_dataset = prepare_dataset(cfg, tokenizer, split="val")

    # Use HF Seq2Seq data collator to handle dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, pad_to_multiple_of=8, return_tensors="pt"
    )

    # 6. Set training arguments
    logger.info("Setting up HF training arguments...")
    training_args = TrainingArguments(  # type: ignore[call-arg]
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=cfg.training.logging_steps,
        eval_steps=cfg.training.eval_steps,
        evaluation_strategy=cfg.training.evaluation_strategy,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        output_dir=cfg.training.output_dir,
        bf16=cfg.training.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=cfg.training.fp16
        or (
            cfg.training.bf16
            and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported())
        ),
        report_to="wandb" if cfg.wandb.mode != "disabled" else "none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # 7. Instantiate Trainer
    trainer = Trainer(  # type: ignore[call-arg]
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 8. Train
    logger.info("Starting training loop...")
    trainer.train()

    # 9. Save adapters
    logger.info("Saving adapters and tokenizer configurations...")
    output_adapter_dir = os.path.join(cfg.training.output_dir, "final_adapter")
    os.makedirs(output_adapter_dir, exist_ok=True)

    # Save PEFT adapter weights
    if cfg.model.use_peft:
        model.save_pretrained(output_adapter_dir)
    else:
        trainer.save_model(output_adapter_dir)

    tokenizer.save_pretrained(output_adapter_dir)
    logger.info(f"Model trained and saved successfully to {output_adapter_dir}")

    wandb.finish()
