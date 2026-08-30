import json
import os
from typing import Any

from datasets import Dataset

from invoice_extractor.config import Config
from invoice_extractor.utils.logging import setup_logger

logger = setup_logger(__name__)

SYSTEM_PROMPT = (
    "You are an expert document extraction agent. Your job is to extract invoice/receipt information "
    "from the messy input text and output it as a valid JSON object matching the target schema exactly. "
    "Do not include any explanation or extra text outside the JSON block."
)

USER_PROMPT_TEMPLATE = "Extract invoice details from the following messy text:\n\n{text}"


def format_assistant_response(target: dict[str, Any]) -> str:
    """Formats the target JSON object inside a markdown code block."""
    return f"```json\n{json.dumps(target, indent=2)}\n```"


def load_raw_dataset(path: str) -> list[dict[str, Any]]:
    """Loads raw dataset from a JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_dataset(cfg: Config, tokenizer: Any, split: str = "train") -> Dataset:
    """Loads raw JSONL, formats it with chat templates, and tokenizes it for training/eval."""
    if split == "train":
        path = cfg.data.train_path
        limit = cfg.data.train_size
    elif split == "val":
        path = cfg.data.val_path
        limit = cfg.data.val_size
    else:
        path = cfg.data.test_path
        limit = cfg.data.test_size

    raw_data = load_raw_dataset(path)
    raw_data = raw_data[:limit]
    logger.info(f"Loaded {len(raw_data)} samples for split: {split} (limit: {limit})")

    # Convert list of dicts to HF Dataset
    dataset = Dataset.from_list(raw_data)

    def tokenize_fn(example: dict[str, Any]) -> dict[str, Any]:
        text = example["text"]
        target = example["target"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            {"role": "assistant", "content": format_assistant_response(target)},
        ]

        # Format the full sequence text (input + labels)
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Format only the prompt part to measure its token length
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        full_enc = tokenizer(full_text, truncation=True, max_length=cfg.data.max_seq_length)
        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=cfg.data.max_seq_length)

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        prompt_len = len(prompt_enc["input_ids"])
        if prompt_len > len(input_ids):
            prompt_len = len(input_ids)

        # Mask the prompt tokens to -100 so the cross-entropy loss ignores them
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return dataset.map(
        tokenize_fn, remove_columns=dataset.column_names, desc=f"Tokenizing {split} split"
    )
