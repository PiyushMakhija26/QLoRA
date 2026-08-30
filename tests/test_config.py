from hydra import compose, initialize


def test_config_parsing() -> None:
    """Tests that Hydra configs load and compose correctly."""
    # Initialize hydra relative to this test file's location
    # configs is at the project root
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[])

        # Test model fields
        assert "name" in cfg.model
        assert cfg.model.name == "Qwen/Qwen2.5-1.5B-Instruct"

        # Test training fields
        assert "learning_rate" in cfg.training
        assert cfg.training.learning_rate == 2e-4

        # Test data fields
        assert "train_size" in cfg.data
        assert cfg.data.train_size == 2000

        # Test evaluation fields
        assert "batch_size" in cfg.evaluation
        assert cfg.evaluation.batch_size == 8
