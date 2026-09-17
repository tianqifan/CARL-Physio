"""Dataset-independent loading and LOSO utilities for preprocessed CARL data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def as_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class MultimodalWindowDataset(Dataset):
    """Split an ``[N,T,C]`` array into modality tensors on access."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        channel_indices: Sequence[Sequence[int]],
    ) -> None:
        if windows.ndim != 3:
            raise ValueError(f"X must have shape [N,T,C], received {windows.shape}.")
        if len(windows) != len(labels):
            raise ValueError("X and y contain different numbers of windows.")
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.channel_indices = tuple(tuple(int(i) for i in group) for group in channel_indices)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor]:
        window = self.windows[index]  # [T,C]
        modalities = [window[:, indices].transpose(0, 1) for indices in self.channel_indices]
        return modalities, self.labels[index]


def load_preprocessed_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the same ``X/y/subject_ids`` archive interface used in the experiments."""

    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Preprocessed archive not found: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = {"X", "y", "subject_ids"}.difference(archive.files)
        if missing:
            raise KeyError(f"{archive_path} is missing arrays: {sorted(missing)}")
        X = np.asarray(archive["X"], dtype=np.float32)
        y = np.asarray(archive["y"], dtype=np.int64)
        subject_ids = np.asarray(archive["subject_ids"])
    if X.ndim != 3 or len(X) != len(y) or len(y) != len(subject_ids):
        raise ValueError("Expected X=[N,T,C], y=[N], and subject_ids=[N].")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values; finish preprocessing before training.")
    return X, y, subject_ids


def loso_split(
    subject_ids: np.ndarray, fold_index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object, object]:
    """Reproduce the held-out test subject and rotating validation-subject split."""

    subjects = np.unique(subject_ids)
    if len(subjects) < 3:
        raise ValueError("LOSO with a subject-level validation set requires at least 3 subjects.")
    if not 0 <= fold_index < len(subjects):
        raise IndexError(f"fold_index must be in [0,{len(subjects) - 1}].")
    test_subject = subjects[fold_index]
    remaining = subjects[subjects != test_subject]
    validation_subject = remaining[fold_index % len(remaining)]
    test_mask = subject_ids == test_subject
    validation_mask = subject_ids == validation_subject
    train_mask = ~(test_mask | validation_mask)
    return train_mask, validation_mask, test_mask, test_subject, validation_subject


def fit_channel_standardization(train_windows: np.ndarray) -> NormalizationStats:
    """Fit per-channel z-score parameters using training subjects only."""

    mean = train_windows.mean(axis=(0, 1), dtype=np.float64)
    std = train_windows.std(axis=(0, 1), dtype=np.float64)
    std = np.where(std > 1e-8, std, 1.0)
    return NormalizationStats(mean=mean, std=std)


def apply_channel_standardization(
    windows: np.ndarray, stats: NormalizationStats
) -> np.ndarray:
    return ((windows - stats.mean[None, None, :]) / stats.std[None, None, :]).astype(
        np.float32, copy=False
    )


def modality_spec(config: Mapping[str, object]) -> Tuple[List[str], List[List[int]]]:
    modalities = config["dataset"]["modalities"]
    names = [entry["name"] for entry in modalities]
    indices = [entry["channels"] for entry in modalities]
    return names, indices
