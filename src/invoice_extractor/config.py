from dataclasses import dataclass


@dataclass
class LoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]


@dataclass
class QuantizationConfig:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str


@dataclass
class ModelConfig:
    name: str
    torch_dtype: str
    use_peft: bool
    lora: LoraConfig
    quantization: QuantizationConfig


@dataclass
class TrainingConfig:
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    weight_decay: float
    lr_scheduler_type: str
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    evaluation_strategy: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    output_dir: str
    bf16: bool
    fp16: bool
    report_to: str


@dataclass
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    max_seq_length: int
    num_samples: int
    train_size: int
    val_size: int
    test_size: int
    checksum_file: str


@dataclass
class EvaluationConfig:
    batch_size: int
    temperature: float
    max_new_tokens: int
    few_shot_examples: int
    results_dir: str


@dataclass
class WandbConfig:
    project: str
    entity: str | None
    mode: str
    tags: list[str]


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvaluationConfig
    seed: int
    wandb: WandbConfig
    is_baseline: bool = False
    adapter_path: str | None = None
