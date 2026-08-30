import json
import os
import subprocess
import sys
from typing import Any


def run_command(cmd: list[str]) -> None:
    print(f"\n[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[FAILED] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    # 1. Generate data
    print("=== Step 1: Generating Dataset ===")
    gen_cmd = [sys.executable, "scripts/generate_data.py"]
    if dry_run:
        gen_cmd.extend(["data.train_size=10", "data.val_size=5", "data.test_size=5"])
    run_command(gen_cmd)

    # Define parameters
    default_r = 16
    default_lr = "2e-4"
    default_size = 10 if dry_run else 2000

    epochs = 1 if dry_run else 3

    runs = {
        "Baseline (Few-shot)": {"is_baseline": True, "args": []},
        "Default (r=16, lr=2e-4, 100% data)": {
            "is_baseline": False,
            "args": [
                f"model.lora.r={default_r}",
                f"model.lora.lora_alpha={2 * default_r}",
                f"training.learning_rate={default_lr}",
                f"data.train_size={default_size}",
                f"training.num_train_epochs={epochs}",
            ],
        },
        "LoRA Rank r=8": {
            "is_baseline": False,
            "args": [
                "model.lora.r=8",
                "model.lora.lora_alpha=16",
                f"training.learning_rate={default_lr}",
                f"data.train_size={default_size}",
                f"training.num_train_epochs={epochs}",
            ],
        },
        "LoRA Rank r=64": {
            "is_baseline": False,
            "args": [
                "model.lora.r=64",
                "model.lora.lora_alpha=128",
                f"training.learning_rate={default_lr}",
                f"data.train_size={default_size}",
                f"training.num_train_epochs={epochs}",
            ],
        },
        "Learning Rate lr=5e-5": {
            "is_baseline": False,
            "args": [
                f"model.lora.r={default_r}",
                f"model.lora.lora_alpha={2 * default_r}",
                "training.learning_rate=5e-5",
                f"data.train_size={default_size}",
                f"training.num_train_epochs={epochs}",
            ],
        },
        "Learning Rate lr=1e-4": {
            "is_baseline": False,
            "args": [
                f"model.lora.r={default_r}",
                f"model.lora.lora_alpha={2 * default_r}",
                "training.learning_rate=1e-4",
                f"data.train_size={default_size}",
                f"training.num_train_epochs={epochs}",
            ],
        },
        "Dataset Size 25%": {
            "is_baseline": False,
            "args": [
                f"model.lora.r={default_r}",
                f"model.lora.lora_alpha={2 * default_r}",
                f"training.learning_rate={default_lr}",
                f"data.train_size={2 if dry_run else 500}",
                f"training.num_train_epochs={epochs}",
            ],
        },
        "Dataset Size 50%": {
            "is_baseline": False,
            "args": [
                f"model.lora.r={default_r}",
                f"model.lora.lora_alpha={2 * default_r}",
                f"training.learning_rate={default_lr}",
                f"data.train_size={5 if dry_run else 1000}",
                f"training.num_train_epochs={epochs}",
            ],
        },
    }

    summary_results = {}

    for name, spec in runs.items():
        print(f"\n=== Executing Experiment: {name} ===")

        # Generate safe name for folders
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("=", "")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "")
            .replace(",", "")
        )
        out_dir = f"outputs/ablation_{safe_name}"

        if spec["is_baseline"]:
            eval_cmd = [
                sys.executable,
                "scripts/eval.py",
                "is_baseline=True",
                f"evaluation.results_dir=results/{safe_name}",
            ]
            if dry_run:
                eval_cmd.append("data.test_size=5")
            run_command(eval_cmd)
        else:
            train_cmd = [
                sys.executable,
                "scripts/train.py",
                f"training.output_dir={out_dir}",
            ] + spec["args"]
            if dry_run:
                train_cmd.extend(
                    ["training.logging_steps=1", "training.eval_steps=1", "training.save_steps=1"]
                )
            run_command(train_cmd)

            eval_cmd = [
                sys.executable,
                "scripts/eval.py",
                f"adapter_path={out_dir}/final_adapter",
                f"evaluation.results_dir=results/{safe_name}",
            ]
            if dry_run:
                eval_cmd.append("data.test_size=5")
            run_command(eval_cmd)

        json_path = (
            f"results/{safe_name}/results_baseline.json"
            if spec["is_baseline"]
            else f"results/{safe_name}/results_finetuned.json"
        )
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
                summary_results[name] = data["summary"]
        else:
            print(f"[WARNING] Expected results file not found at {json_path}")

    write_summary_report(summary_results)


def write_summary_report(results: dict[str, Any]) -> None:
    os.makedirs("results", exist_ok=True)

    md = "# Ablation and Baseline Evaluation Summary\n\n"
    md += "| Experiment | JSON Validity | Schema Compliant | Exact Match | Vendor Sim | Line Items F1 | Tax MAE | Total MAE |\n"
    md += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"

    for name, metrics in results.items():
        json_val = f"{metrics.get('json_valid_rate', 0.0) * 100:.1f}%"
        schema_comp = f"{metrics.get('schema_compliant_rate', 0.0) * 100:.1f}%"
        em = f"{metrics.get('exact_match_rate', 0.0) * 100:.1f}%"
        v_sim = f"{metrics.get('vendor_similarity', 0.0) * 100:.1f}%"
        li_f1 = f"{metrics.get('line_items_f1', 0.0) * 100:.1f}%"
        tax_mae = f"{metrics.get('tax_amount_mean_absolute_error', 0.0):.2f}"
        tot_mae = f"{metrics.get('total_amount_mean_absolute_error', 0.0):.2f}"

        md += f"| {name} | {json_val} | {schema_comp} | {em} | {v_sim} | {li_f1} | {tax_mae} | {tot_mae} |\n"

    summary_md_path = "results/results_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nAblations complete! Summary written to {summary_md_path}")

    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
