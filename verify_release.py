"""Automated checks for the declared CARL architecture and dataset configs."""

from pathlib import Path

from carl.components import ModalityEncoder
from carl.runtime import build_model, load_config


EXPECTED = {
    "WESAD": {"modalities": [1, 1, 1, 1, 1], "classes": 3, "bias": 2.0},
    "EmoWork": {"modalities": [4, 1, 1, 1, 1], "classes": 2, "bias": 0.5},
    "CASE": {"modalities": [1, 1, 1, 1, 1, 3], "classes": 2, "bias": 0.5},
}


def main() -> None:
    assert ModalityEncoder.CHANNELS == (64, 128, 256, 512, 512)
    assert ModalityEncoder.KERNELS == (17, 11, 7, 5, 3)
    for path in sorted(Path(__file__).parent.joinpath("configs").glob("*.yaml")):
        config = load_config(path)
        name = config["dataset"]["name"]
        expected = EXPECTED[name]
        channel_counts = [len(item["channels"]) for item in config["dataset"]["modalities"]]
        assert channel_counts == expected["modalities"]
        assert config["dataset"]["num_classes"] == expected["classes"]
        assert config["model"]["fusion_blocks"] == [3, 4, 5]
        assert config["model"]["relationship_blocks"] == [3, 4, 5]
        assert config["model"]["attention_bias_scale"] == expected["bias"]
        assert config["model"]["graph_residual_weight"] == 0.2
        assert config["model"]["graph_temperature_initial"] == 6.0
        assert config["loss"] == {
            "reconstruction": 0.1,
            "orthogonality": 0.1,
            "semantic": 1.0,
        }
        model = build_model(config)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name}: configuration OK; trainable parameters={parameters:,}")
    print("Release architecture/configuration checks passed.")


if __name__ == "__main__":
    main()
