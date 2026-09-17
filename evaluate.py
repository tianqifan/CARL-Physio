"""Evaluate one released CARL LOSO checkpoint without retraining."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from carl.data import MultimodalWindowDataset, NormalizationStats, apply_channel_standardization, load_preprocessed_npz, modality_spec
from carl.runtime import build_model
from carl.training import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    X, y, subject_ids = load_preprocessed_npz(args.data)
    mask = subject_ids == checkpoint["test_subject"]
    stats = NormalizationStats(
        mean=np.asarray(checkpoint["normalization"]["mean"]),
        std=np.asarray(checkpoint["normalization"]["std"]),
    )
    _, channel_indices = modality_spec(config)
    dataset = MultimodalWindowDataset(
        apply_channel_standardization(X[mask], stats), y[mask], channel_indices
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(evaluate(model, loader, device))


if __name__ == "__main__":
    main()
