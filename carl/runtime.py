"""Configuration, reproducibility, metrics, and model-construction helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from .losses import CARLLossWeights
from .model import CARL


def load_config(path: str | Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("dataset", "model", "loss", "training"):
        if section not in config:
            raise KeyError(f"Configuration is missing the '{section}' section.")
    return config


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_model(config: Mapping[str, object]) -> CARL:
    dataset = config["dataset"]
    model = config["model"]
    modality_channels = [len(item["channels"]) for item in dataset["modalities"]]
    return CARL(
        modality_channels=modality_channels,
        input_length=int(dataset["input_length"]),
        num_classes=int(dataset["num_classes"]),
        fusion_blocks=model["fusion_blocks"],
        relationship_blocks=model["relationship_blocks"],
        num_heads=int(model["num_heads"]),
        graph_alpha=float(model["graph_alpha"]),
        relation_beta=float(model["relation_beta"]),
        attention_bias_scale=float(model["attention_bias_scale"]),
        graph_residual_weight=float(model["graph_residual_weight"]),
        graph_temperature_initial=float(model["graph_temperature_initial"]),
        relationship_warmup_epochs=int(model["relationship_warmup_epochs"]),
        relationship_momentum=float(model["relationship_momentum"]),
    )


def loss_weights(config: Mapping[str, object]) -> CARLLossWeights:
    values = config["loss"]
    return CARLLossWeights(
        reconstruction=float(values["reconstruction"]),
        orthogonality=float(values["orthogonality"]),
        semantic=float(values["semantic"]),
    )


def classification_metrics(targets: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted")),
    }


def save_json(value: object, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
