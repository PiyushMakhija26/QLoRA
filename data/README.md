# Invoice Extraction Dataset (Synthetic Document Dataset)

This directory contains the synthetic dataset of unstructured invoice/receipt documents and their corresponding structured JSON ground truth targets.

## Sourcing & Generation Methodology

Real-world financial documents are highly unstructured, vary in template design, and often contain noise from OCR (Optical Character Recognition) scanners or typos from manual entry. 

To simulate these challenges, we programmatically generate invoice data using diverse layout templates, format variations, and noise injection:

### 1. Document Layout Templates
We use 4 distinct layout templates to generate the documents:
* **Template A (Formal Invoice)**: Grid/aligned text with explicit section headers (`Vendor`, `Invoice No`, `Description`, `Qty`, `Unit Price`, `Total Price`, `Subtotal`, `Tax`, `Total Due`).
* **Template B (Email Receipt)**: Natural conversational prose ("Hi team, please find attached invoice...") listing items with quantity, description, unit price, and subtotal.
* **Template C (Thermal Receipt)**: Compact POS format with dash dividers, missing item headers, and thermal printing terminal layouts.
* **Template D (OCR Scan Output)**: Messy, single-line scanning with stray characters, merged fields, and line breaks.

### 2. Formatting Variation
* **Dates**: Randomized formats, e.g. YYYY-MM-DD (`2026-08-29`), DD/MM/YYYY (`29/08/2026`), MM/DD/YYYY (`08/29/2026`), or descriptive month formats (`August 29, 2026`).
* **Prices/Currencies**: Formatting with symbols (`$`, `€`, `£`) or currency codes (`USD`, `EUR`, `GBP`) placed before or after values (e.g. `$50.00`, `50.00 USD`, `USD 50.00`).
* **Spelling Typos**: Typo injections on vendor names and item descriptions based on keyboard distance.

### 3. OCR Noise Injection
With low probability, standard character-substitution errors are introduced to replicate scanner issues:
* `0` $\rightarrow$ `O`
* `1` $\rightarrow$ `l` or `I`
* `5` $\rightarrow$ `S`
* `8` $\rightarrow$ `B`
* `2` $\rightarrow$ `Z`
* Messy spacing (duplicate spaces and line-alignment shifting).

The model's goal is to parse this messy representation, ignore OCR noise, standardize all fields (e.g., standardizing dates to `YYYY-MM-DD` and quantities/amounts to floats/ints), and output valid JSON conforming to the Pydantic schema.

---

## Data Split & Versioning

* **Seed**: `42` (Fixed seed for all randomized data generation to ensure determinism).
* **Splits**:
  * **Train**: 2,000 examples
  * **Val**: 500 examples
  * **Test**: 500 examples (Held-out set, never used for training or model selection).
* **Versioning**: Every generation run computes SHA256 checksums of the generated files and records them in `data/checksums.json` for integrity validation.
