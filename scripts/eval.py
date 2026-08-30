import os

import hydra
from omegaconf import DictConfig

from invoice_extractor.config import Config
from invoice_extractor.evaluation.eval import run_evaluation
from invoice_extractor.utils.logging import setup_logger

logger = setup_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    typed_cfg = Config(
        model=cfg.model,
        training=cfg.training,
        data=cfg.data,
        evaluation=cfg.evaluation,
        seed=cfg.seed,
        wandb=cfg.wandb,
        is_baseline=cfg.get("is_baseline", False),
        adapter_path=cfg.get("adapter_path", None),
    )

    # Read adapter path from Hydra or default output directory
    try:
        orig_cwd = hydra.utils.get_original_cwd()
    except ValueError:
        orig_cwd = os.getcwd()

    adapter_path = cfg.get("adapter_path", None)
    if not adapter_path:
        default_out = os.path.join(orig_cwd, cfg.training.output_dir, "final_adapter")
        if os.path.exists(default_out):
            adapter_path = default_out

    is_baseline = cfg.get("is_baseline", False)

    logger.info(f"Starting evaluation. Baseline: {is_baseline}, Adapter: {adapter_path}")
    metrics = run_evaluation(typed_cfg, adapter_path=adapter_path, is_baseline=is_baseline)
    logger.info(f"Aggregated metrics: {metrics}")


if __name__ == "__main__":
    main()
