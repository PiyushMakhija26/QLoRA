import json
import os

import hydra
from omegaconf import DictConfig

from invoice_extractor.data.generator import generate_dataset, save_dataset_and_calculate_checksum
from invoice_extractor.utils.logging import setup_logger

logger = setup_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("Generating dataset with configuration...")

    seed = cfg.seed
    train_size = cfg.data.train_size
    val_size = cfg.data.val_size
    test_size = cfg.data.test_size

    total_size = train_size + val_size + test_size
    logger.info(
        f"Total dataset size to generate: {total_size} (Train: {train_size}, Val: {val_size}, Test: {test_size})"
    )

    dataset = generate_dataset(total_size, seed)

    train_data = dataset[:train_size]
    val_data = dataset[train_size : train_size + val_size]
    test_data = dataset[train_size + val_size :]

    # Locate output paths relative to original execution directory
    try:
        orig_cwd = hydra.utils.get_original_cwd()
    except ValueError:
        orig_cwd = os.getcwd()

    train_path = os.path.join(orig_cwd, cfg.data.train_path)
    val_path = os.path.join(orig_cwd, cfg.data.val_path)
    test_path = os.path.join(orig_cwd, cfg.data.test_path)
    checksum_path = os.path.join(orig_cwd, cfg.data.checksum_file)

    logger.info(
        f"Saving splits to: \n- Train: {train_path}\n- Val: {val_path}\n- Test: {test_path}"
    )
    train_sha = save_dataset_and_calculate_checksum(train_data, train_path)
    val_sha = save_dataset_and_calculate_checksum(val_data, val_path)
    test_sha = save_dataset_and_calculate_checksum(test_data, test_path)

    checksums = {
        "train": {"path": cfg.data.train_path, "sha256": train_sha, "size": len(train_data)},
        "val": {"path": cfg.data.val_path, "sha256": val_sha, "size": len(val_data)},
        "test": {"path": cfg.data.test_path, "sha256": test_sha, "size": len(test_data)},
        "seed": seed,
    }

    os.makedirs(os.path.dirname(checksum_path), exist_ok=True)
    with open(checksum_path, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    logger.info("Dataset generated successfully!")
    logger.info(f"Checksums file written to {checksum_path}")


if __name__ == "__main__":
    main()
