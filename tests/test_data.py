import os
import tempfile

from transformers import AutoTokenizer

from invoice_extractor.config import (
    Config,
    DataConfig,
    EvaluationConfig,
    LoraConfig,
    ModelConfig,
    QuantizationConfig,
    TrainingConfig,
    WandbConfig,
)
from invoice_extractor.data.generator import (
    generate_dataset,
    generate_invoice,
    save_dataset_and_calculate_checksum,
)
from invoice_extractor.data.loader import load_raw_dataset, prepare_dataset
from invoice_extractor.data.schema import InvoiceSchema


def test_generate_invoice() -> None:
    res = generate_invoice(seed=42)
    assert "text" in res
    assert "target" in res
    assert isinstance(res["text"], str)
    assert isinstance(res["target"], dict)

    # Verify target matches Pydantic schema validation
    InvoiceSchema.model_validate(res["target"])


def test_generate_dataset() -> None:
    res = generate_dataset(num_samples=5, seed=42)
    assert len(res) == 5
    # Check that seeds create diverse invoices
    assert res[0]["target"]["invoice_number"] != res[1]["target"]["invoice_number"]


def test_save_and_checksum() -> None:
    dataset = generate_dataset(num_samples=5, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.jsonl")
        sha = save_dataset_and_calculate_checksum(dataset, filepath)

        # Verify file exists
        assert os.path.exists(filepath)
        assert len(sha) == 64  # SHA256 length in hex

        # Verify load back
        loaded = load_raw_dataset(filepath)
        assert len(loaded) == 5
        assert loaded[0]["target"]["vendor"] == dataset[0]["target"]["vendor"]


def test_dataset_loader() -> None:
    dataset = generate_dataset(num_samples=5, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.jsonl")
        save_dataset_and_calculate_checksum(dataset, train_path)

        # Create minimal Config object
        cfg = Config(
            model=ModelConfig(
                name="Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype="bfloat16",
                use_peft=True,
                lora=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj"]),
                quantization=QuantizationConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype="bfloat16",
                ),
            ),
            training=TrainingConfig(
                learning_rate=2e-4,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                logging_steps=1,
                eval_steps=1,
                evaluation_strategy="steps",
                save_strategy="steps",
                save_steps=1,
                save_total_limit=1,
                output_dir=tmpdir,
                bf16=True,
                fp16=False,
                report_to="none",
            ),
            data=DataConfig(
                train_path=train_path,
                val_path=train_path,
                test_path=train_path,
                max_seq_length=256,
                num_samples=5,
                train_size=5,
                val_size=5,
                test_size=5,
                checksum_file=os.path.join(tmpdir, "checksums.json"),
            ),
            evaluation=EvaluationConfig(
                batch_size=2,
                temperature=0.0,
                max_new_tokens=64,
                few_shot_examples=1,
                results_dir=tmpdir,
            ),
            seed=42,
            wandb=WandbConfig(project="test", entity=None, mode="disabled", tags=[]),
        )

        # Mock load tokenizer (load lightweight fast tokenizer or mock it)
        # For tests, we use the actual Qwen fast tokenizer which loads in milliseconds
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized = prepare_dataset(cfg, tokenizer, split="train")
        assert len(tokenized) == 5
        assert "input_ids" in tokenized.column_names
        assert "attention_mask" in tokenized.column_names
        assert "labels" in tokenized.column_names

        # Verify label masking (-100 values exist in labels)
        first_sample = tokenized[0]
        assert -100 in first_sample["labels"]
        # Verify some actual label IDs exist too
        assert any(x != -100 for x in first_sample["labels"])
