"""Package the preprocessed EmoWork arrays into the common CARL NPZ interface.

This script does not process or redistribute raw data. It performs only the
array concatenation and valence-label selection used by the experiment loader.
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Preprocessed EmoWork_all.npz")
    parser.add_argument("--output", required=True, help="Destination common-interface NPZ")
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as archive:
        required = {"eeg", "ecg", "eda", "bvp", "temp", "labels", "pid"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"Input archive is missing arrays: {sorted(missing)}")
        arrays = []
        for key in ("eeg", "ecg", "eda", "bvp", "temp"):
            value = np.asarray(archive[key])
            if value.ndim == 2:
                value = value[:, None, :]
            arrays.append(value)
        X = np.concatenate(arrays, axis=1).transpose(0, 2, 1).astype(np.float32)
        # labels columns: stress, arousal, valence, suppression
        y = np.asarray(archive["labels"][:, 2], dtype=np.int64)
        subject_ids = np.asarray(archive["pid"], dtype=np.int64)

    np.savez_compressed(args.output, X=X, y=y, subject_ids=subject_ids)
    values, counts = np.unique(y, return_counts=True)
    print(
        f"saved={args.output} X={X.shape} subjects={len(np.unique(subject_ids))} "
        f"classes={dict(zip(values.tolist(), counts.tolist()))}"
    )


if __name__ == "__main__":
    main()
