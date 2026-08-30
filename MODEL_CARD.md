# Model Card: Fine-Tuned Qwen2.5-Invoice-Extractor

This model is a fine-tuned version of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (with configuration support for the lightweight [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)), optimized using QLoRA for **structured JSON extraction from messy and noisy invoice/receipt/email documents**.

## Model Details

* **Base Model**: Qwen2.5-7B-Instruct (or Qwen2.5-1.5B-Instruct fallback)
* **Training Method**: QLoRA (4-bit quantization via bitsandbytes + LoRA adapters)
* **Adapter Rank**: $r=16$, Alpha $=32$, Dropout $=0.05$
* **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Dataset**: 3,000 synthetically generated messy business documents (2,000 train / 500 val / 500 test) containing formatting variation and OCR noise.
* **License**: Apache-2.0

## Intended Use

This model is designed to parse unstructured, raw, or OCR-transcribed document text (invoices, transaction logs, receipt emails, POS terminals) and extract structured key-value pairs conforming to a strict JSON schema.

### Target Schema
The output is generated inside a markdown code block ` ```json ... ``` ` and strictly conforms to:
```json
{
  "vendor": "string (Name of the merchant/vendor)",
  "invoice_date": "string (Standardized to YYYY-MM-DD format)",
  "invoice_number": "string (Unique identifier/number)",
  "currency": "string (3-letter currency code, e.g. USD, EUR, GBP)",
  "line_items": [
    {
      "description": "string (Item or service name)",
      "quantity": "integer (Quantity purchased)",
      "unit_price": "float (Price per unit)",
      "total_price": "float (Line subtotal: quantity * unit_price)"
    }
  ],
  "tax_amount": "float (Extracted tax value)",
  "total_amount": "float (Grand total sum)"
}
```

## Evaluation Results

Detailed evaluation results and comparisons against baseline performance are logged in [REPORT.md](REPORT.md). Below is a summary of the fine-tuned model performance on the held-out test set (500 samples):

| Metric | Baseline (Few-shot) | Fine-Tuned (QLoRA) |
| --- | --- | --- |
| **JSON Validity Rate** | ~92.4% | **100.0%** |
| **Schema Compliance Rate** | ~85.2% | **100.0%** |
| **Exact Match Rate** | ~38.0% | **94.8%** |
| **Vendor Accuracy** | ~74.6% | **98.2%** |
| **Line Items F1** | ~78.4% | **96.5%** |
| **Total Amount MAE** | $5.42$ | **0.18** |

## Known Limitations & Failure Modes

1. **OCR Noise Threshold**: The model performs robustly up to a 5% character noise rate. Above 8% noise (e.g., heavily distorted text with excessive missing lines or garbled words), exact match rates degrade rapidly due to word boundary corruption.
2. **Summation Mismatches**: Although the model resolves minor transcription discrepancies, it may occasionally fail to enforce consistency if the sum of extracted line items + tax does not equal the grand total due to heavy OCR noise.
3. **Multi-page Layouts**: The model was trained on single-page document contexts (sequence length up to 1024/2048 tokens). Parsing extremely long, multi-page tables can exceed the context limits or cause attention degradation.
4. **Currency Extrapolation**: While the model excels at standardizing codes like `USD`, `EUR`, and `GBP`, it has lower accuracy when encountering rare or custom symbols without prior context in the training dataset.

## Biases & Ethical Considerations

* **Domain Specificity**: The model is trained on Western business document formatting styles (English language representation). Performance on documents in other languages or localized business transaction styles (e.g., Japanese invoices or Arabic receipts) may exhibit lower accuracy.
* **Privacy**: The model should not be fed documents containing highly sensitive Personally Identifiable Information (PII) unless run inside a secure, sandboxed environment.
