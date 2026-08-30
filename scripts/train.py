import hydra
from omegaconf import DictConfig

from invoice_extractor.config import Config
from invoice_extractor.training.train import train_lora


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Cast DictConfig to typed Config dataclass
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
    train_lora(typed_cfg)


if __name__ == "__main__":
    main()
