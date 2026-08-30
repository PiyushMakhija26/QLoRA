import json
import os
from typing import Any, cast

import hydra
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from invoice_extractor.config import Config
from invoice_extractor.data.loader import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    format_assistant_response,
    load_raw_dataset,
)
from invoice_extractor.evaluation.metrics import evaluate_prediction
from invoice_extractor.utils.logging import setup_logger
from invoice_extractor.utils.seed import set_seed

logger = setup_logger(__name__)


def run_evaluation(
    cfg: Config, adapter_path: str | None = None, is_baseline: bool = False
) -> dict[str, Any]:
    """Runs evaluation on the held-out test set for either fine-tuned or baseline models."""
    set_seed(cfg.seed)

    model_id = cfg.model.name
    logger.info(f"Loading tokenizer and base model for evaluation: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    torch_dtype = (
        torch.bfloat16 if cfg.model.torch_dtype == "bfloat16" and cuda_available else torch.float32
    )

    logger.info(f"Loading model on {device} with dtype {torch_dtype}...")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if cuda_available else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(
            f"Failed to load model with device_map auto: {e}. Retrying without device_map..."
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, trust_remote_code=True
        )
        if cuda_available:
            base_model = base_model.to("cuda")  # type: ignore[arg-type]

    model: Any
    if adapter_path and not is_baseline:
        logger.info(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()

    # 2. Load dataset
    try:
        orig_cwd = hydra.utils.get_original_cwd()
    except ValueError:
        orig_cwd = os.getcwd()

    test_path = os.path.join(orig_cwd, cfg.data.test_path)
    logger.info(f"Loading test set from {test_path}...")
    test_data = load_raw_dataset(test_path)

    few_shot_examples = []
    if is_baseline:
        train_path = os.path.join(orig_cwd, cfg.data.train_path)
        logger.info(f"Baseline run: Loading few-shot examples from {train_path}...")
        try:
            train_data = load_raw_dataset(train_path)
            few_shot_examples = train_data[: cfg.evaluation.few_shot_examples]
        except Exception as e:
            logger.warning(
                f"Could not load train set for few-shot: {e}. Falling back to zero-shot."
            )

    results: list[dict[str, Any]] = []

    # 3. Generate predictions
    logger.info(f"Running evaluation on {len(test_data)} examples...")
    for item in tqdm(test_data, desc="Evaluating"):
        text = item["text"]
        target = item["target"]

        # Build prompt messages
        if is_baseline and few_shot_examples:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for ex in few_shot_examples:
                messages.append(
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=ex["text"])}
                )
                messages.append(
                    {"role": "assistant", "content": format_assistant_response(ex["target"])}
                )
            messages.append({"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)})
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt")
        # Ensure inputs are moved to same device as model
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.evaluation.max_new_tokens,
                temperature=cfg.evaluation.temperature,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = output_ids[0][input_len:]
        pred_text = cast(str, tokenizer.decode(gen_tokens, skip_special_tokens=True))

        # Run metric evaluation
        sample_metrics = evaluate_prediction(pred_text, target)

        results.append(
            {"text": text, "target": target, "prediction_raw": pred_text, "metrics": sample_metrics}
        )

    # 4. Compute aggregate metrics
    aggregated = compute_aggregate_metrics(results)

    # 5. Save results to output
    results_dir = os.path.join(orig_cwd, cfg.evaluation.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    run_type = "baseline" if is_baseline else "finetuned"
    output_json_path = os.path.join(results_dir, f"results_{run_type}.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model_name": cfg.model.name,
                    "is_baseline": is_baseline,
                    "adapter_path": adapter_path,
                },
                "summary": aggregated,
                "details": results,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {output_json_path}")
    return aggregated


def compute_aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    json_valid = 0.0
    schema_compliant = 0.0
    exact_match = 0.0
    vendor_acc = 0.0
    vendor_sim = 0.0
    date_acc = 0.0
    invoice_number_acc = 0.0
    currency_acc = 0.0
    tax_amount_err = 0.0
    total_amount_err = 0.0
    line_items_precision = 0.0
    line_items_recall = 0.0
    line_items_f1 = 0.0

    error_counts: dict[str, int] = {}

    for r in results:
        m = r["metrics"]
        json_valid += m["json_valid"]
        schema_compliant += m["schema_compliant"]
        exact_match += m["exact_match"]
        vendor_acc += m["vendor_acc"]
        vendor_sim += m["vendor_sim"]
        date_acc += m["date_acc"]
        invoice_number_acc += m["invoice_number_acc"]
        currency_acc += m["currency_acc"]
        tax_amount_err += m["tax_amount_err"]
        total_amount_err += m["total_amount_err"]
        line_items_precision += m["line_items_precision"]
        line_items_recall += m["line_items_recall"]
        line_items_f1 += m["line_items_f1"]

        for err in m["errors"]:
            error_counts[err] = error_counts.get(err, 0) + 1

    error_freq = {k: v / total for k, v in error_counts.items()}

    return {
        "json_valid_rate": json_valid / total,
        "schema_compliant_rate": schema_compliant / total,
        "exact_match_rate": exact_match / total,
        "vendor_accuracy": vendor_acc / total,
        "vendor_similarity": vendor_sim / total,
        "date_accuracy": date_acc / total,
        "invoice_number_accuracy": invoice_number_acc / total,
        "currency_accuracy": currency_acc / total,
        "tax_amount_mean_absolute_error": tax_amount_err / total,
        "total_amount_mean_absolute_error": total_amount_err / total,
        "line_items_precision": line_items_precision / total,
        "line_items_recall": line_items_recall / total,
        "line_items_f1": line_items_f1 / total,
        "error_frequencies": error_freq,
        "error_counts": error_counts,
        "sample_count": total,
    }
