# Production-Grade QLoRA Fine-Tuning Pipeline for Structured Extraction

This repository contains a complete, production-grade ML pipeline for fine-tuning an open-weight LLM (e.g. Qwen2.5-7B-Instruct or Qwen2.5-1.5B-Instruct) using QLoRA to extract structured JSON metadata from messy, OCR-distorted document text.

The project features a rigorous evaluation harness (field-level F1 calculations with Hungarian bipartite matching and automated error categorization), complete configuration management, and unit testing/CI checks.

---

## Project Layout

```
LORA/
├── src/
│   └── invoice_extractor/
│       ├── __init__.py
│       ├── config.py           # Hydra / YAML configs mapped to Pydantic/dataclasses
│       ├── data/
│       │   ├── __init__.py
│       │   ├── generator.py    # Synthetic invoice dataset generator
│       │   ├── loader.py       # Prompt building, tokenization, formatting
│       │   └── schema.py       # Pydantic schemas for verification
│       ├── training/
│       │   ├── __init__.py
│       │   └── train.py        # QLoRA fine-tuning training loop via HF Trainer
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py      # Field-level F1, Hungarian matching for line items
│       │   └── eval.py         # Evaluation run + baseline comparative run
│       └── utils/
│           ├── __init__.py
│           ├── logging.py      # Structured python logging setup
│           └── seed.py         # Seed setting helper
├── configs/
│   ├── config.yaml             # Main Hydra config
│   ├── model/
│   │   ├── qwen2.5_7b.yaml     # 7B default config
│   │   └── qwen2.5_1.5b.yaml   # 1.5B low-VRAM fallback config
│   ├── training/
│   │   └── default.yaml        # Quantization, LoRA hyperparams, training steps
│   └── data/
│       └── default.yaml        # Tokenizer config, max length, paths
├── scripts/
│   ├── generate_data.py        # CLI entrypoint for synthetic data generation
│   ├── train.py                # CLI entrypoint for training
│   ├── eval.py                 # CLI entrypoint for evaluation
│   └── run_ablations.py        # CLI script to execute the ablation configs
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # Unit tests for dataset generation & loading
│   ├── test_metrics.py         # Unit tests for evaluation metric correctness
│   └── test_config.py          # Unit tests for configuration parsing
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI for linting, typing, and pytest
├── .pre-commit-config.yaml     # Local pre-commit hook file
├── Dockerfile                  # Reproducible training/eval docker environment
├── pyproject.toml              # UV-based python dependencies and build system
├── README.md                   # Setup, installation, reproduction steps
├── REPORT.md                   # Fine-tuning, ablation, and error analysis results
├── MODEL_CARD.md               # Fine-tuned model description and limitations
└── LICENSE                     # Apache 2.0 License
```

---

## Installation & Setup

We recommend using [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver) to manage your virtual environment and dependencies.

### 1. Local Setup
Clone the repository and run:
```bash
# Verify Python version >= 3.10 is installed
python --version

# Install dependencies and sync virtual environment automatically
uv run pytest
```
`uv` will automatically create a virtual environment under `.venv`, download and install the required packages (including PyTorch, Hugging Face `transformers`, `peft`, and developer packages), and run the unit tests.

### 2. Google Colab / Kaggle Setup
To train the model on a free-tier Colab T4 or Kaggle GPU environment:
```python
# Install required packages
!pip install -q transformers peft bitsandbytes datasets accelerate hydra-core pydantic scipy wandb

# Clone repo and add src/ to python path
import sys
sys.path.append("/content/LORA/src")
```

### 3. Docker Environment
To build and run inside a reproducible Docker container:
```bash
docker build -t lora-extractor .
docker run -it lora-extractor pytest
```

---

## Reproduction Guide

The complete ablation studies and baseline evaluation runs can be reproduced using simple CLI scripts.

### 1. Run Unit Tests & Static Verification
Make sure all checks pass before running your training:
```bash
# Run Unit Tests
uv run pytest

# Run Ruff Linter & Format Checks
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/

# Run Mypy Static Type Checks
uv run mypy src/
```

### 2. Step-by-Step Data, Training, & Eval Commands
```bash
# Generate synthetic dataset splits (Train: 2k, Val: 500, Test: 500)
uv run python scripts/generate_data.py

# Run baseline evaluation (few-shot prompting) on held-out test set
uv run python scripts/eval.py is_baseline=True

# Run QLoRA fine-tuning training loop (Defaults to Qwen2.5-1.5B-Instruct)
uv run python scripts/train.py

# Override base model to 7B in CLI (if CUDA memory allows)
uv run python scripts/train.py model=qwen2.5_7b

# Run evaluation on the fine-tuned adapter weights
uv run python scripts/eval.py
```

### 3. Automated End-to-End Ablation Matrix Runner
We provide a unified script that generates data, evaluates the few-shot baseline, trains and evaluates all ablation adapters sequentially (comparing rank sizes, learning rates, and dataset sizes), and compiles the results into a markdown table inside `results/results_summary.md`.

* **To verify functionality (CPU/Fast Run)**:
  ```bash
  uv run python scripts/run_ablations.py --dry-run
  ```
  *(Runs in seconds with a tiny mock dataset and 1-epoch adapter trainings)*.

* **To execute the full ablation matrix (GPU/Full Run)**:
  ```bash
  uv run python scripts/run_ablations.py
  ```

### 4. Interactive UI/UX Dashboard
We provide a local Streamlit-based web interface to interactively test document extractions and visualize comparative metrics/ablation charts.

* **To run the dashboard locally (in sandbox offline mode by default)**:
  ```bash
  uv run python scripts/app.py
  ```
  This opens a browser tab (typically at `http://localhost:8501`) displaying:
  1. **Real-time Extraction Arena**: Paste any invoice text, choose the extraction model (baseline vs fine-tuned), run extraction, and see structural/schema compliance checks and total sum checks.
  2. **Evaluation Metrics**: View performance comparisons and pie charts of error distributions.
  3. **Hyperparameter Ablations**: View Plotly scatter and bar plots comparing LoRA ranks and dataset sizes.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
