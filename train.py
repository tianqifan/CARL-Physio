"""Train and evaluate CARL with subject-independent LOSO validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from carl.data import (
    MultimodalWindowDataset,
    apply_channel_standardization,
    fit_channel_standardization,
    load_preprocessed_npz,
    loso_split,
    modality_spec,
)
from carl.runtime import build_model, load_config, loss_weights, save_json, set_seed
from carl.training import evaluate, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="NPZ with X, y, and subject_ids arrays")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects", nargs="+", default=["all"], help="Test subject IDs or 'all'")
    parser.add_argument("--device", default=None, help="Override the configured device")
    return parser.parse_args()


def _loader(dataset, batch_size: int, shuffle: bool, drop_last: bool, seed: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator if shuffle else None,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    X, y, subject_ids = load_preprocessed_npz(args.data)
    subjects = np.unique(subject_ids)
    requested = subjects if args.subjects == ["all"] else np.asarray(args.subjects, dtype=subjects.dtype)
    unknown = set(requested.tolist()).difference(subjects.tolist())
    if unknown:
        raise ValueError(f"Unknown subject IDs: {sorted(unknown)}")

    training = config["training"]
    base_seed = int(training["seed"])
    device = torch.device(args.device or training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu if intended.")
    _, channel_indices = modality_spec(config)
    expected_channels = max(index for group in channel_indices for index in group) + 1
    if X.shape[1] != int(config["dataset"]["input_length"]):
        raise ValueError("The archive time dimension does not match dataset.input_length.")
    if X.shape[2] < expected_channels:
        raise ValueError("The archive does not contain every configured modality channel.")

    output_dir = Path(args.output_dir)
    fold_results: List[Dict[str, object]] = []
    for test_subject in requested:
        fold_index = int(np.flatnonzero(subjects == test_subject)[0])
        fold_seed = base_seed + fold_index * 1000
        train_mask, val_mask, test_mask, test_subject, val_subject = loso_split(subject_ids, fold_index)
        stats = fit_channel_standardization(X[train_mask])
        train_X = apply_channel_standardization(X[train_mask], stats)
        val_X = apply_channel_standardization(X[val_mask], stats)
        test_X = apply_channel_standardization(X[test_mask], stats)
        train_set = MultimodalWindowDataset(train_X, y[train_mask], channel_indices)
        val_set = MultimodalWindowDataset(val_X, y[val_mask], channel_indices)
        test_set = MultimodalWindowDataset(test_X, y[test_mask], channel_indices)
        batch_size = int(training["batch_size"])
        train_loader = _loader(train_set, batch_size, True, True, fold_seed, device)
        val_loader = _loader(val_set, batch_size, False, False, fold_seed, device)
        test_loader = _loader(test_set, batch_size, False, False, fold_seed, device)

        set_seed(fold_seed)
        model = build_model(config).to(device)
        # The original protocol fixes stochastic training operations to the declared
        # experiment seed after fold-specific model/data-loader initialization.
        set_seed(base_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training["epochs"]),
            eta_min=float(training["learning_rate"]) * 0.1,
        )
        weights = loss_weights(config)
        best_val_accuracy = 0.0
        best_state = None
        history = []
        for epoch in range(int(training["epochs"])):
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                weights,
                epoch,
                float(training["gradient_clip_norm"]),
            )
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step()
            history.append({"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics})
            print(
                f"subject={test_subject} epoch={epoch + 1:02d} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
            )
            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected.")
        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test_loader, device)
        fold_dir = output_dir / f"subject_{test_subject}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": best_state,
            "config": config,
            "test_subject": test_subject.item() if hasattr(test_subject, "item") else test_subject,
            "validation_subject": val_subject.item() if hasattr(val_subject, "item") else val_subject,
            "normalization": stats.as_dict(),
            "best_validation_accuracy": best_val_accuracy,
            "test_metrics": test_metrics,
        }
        torch.save(checkpoint, fold_dir / "best_model.pt")
        save_json(history, fold_dir / "history.json")
        result = {
            "test_subject": checkpoint["test_subject"],
            "validation_subject": checkpoint["validation_subject"],
            "best_validation_accuracy": best_val_accuracy,
            **test_metrics,
        }
        fold_results.append(result)
        save_json(result, fold_dir / "result.json")

    summary = {
        "dataset": config["dataset"]["name"],
        "seed": base_seed,
        "folds": fold_results,
        "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_results])),
        "accuracy_std": float(np.std([row["accuracy"] for row in fold_results])),
        "weighted_f1_mean": float(np.mean([row["weighted_f1"] for row in fold_results])),
        "weighted_f1_std": float(np.std([row["weighted_f1"] for row in fold_results])),
    }
    save_json(summary, output_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
