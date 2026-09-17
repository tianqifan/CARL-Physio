# CARL: core reproducibility release

This repository contains the minimal training and evaluation implementation of
**Conflict-Aware Relational Learning (CARL)** used in the manuscript. It is not
a simplified illustration: the released encoder dimensions, graph-fusion
blocks, shared/private decomposition, conditional predictor, relationship
update, attention mechanism, classifier, losses, optimizer, scheduler, LOSO
split, checkpoint rule, seed handling, and evaluation metrics correspond to the
experimental implementation. Architecture and training settings are exposed in
the source code and the three dataset configurations below.

## Included

```text
CARL_core_release/
|-- carl/
|   |-- components.py       # encoder, FD/decoder, PSL, CG, graph attention
|   |-- model.py            # complete CARL forward pass
|   |-- losses.py           # classification + three auxiliary losses
|   |-- data.py             # NPZ interface, LOSO, train-only normalization
|   |-- runtime.py          # configuration, seeds, metrics, construction
|   `-- training.py         # train/evaluation loops
|-- configs/
|   |-- wesad.yaml
|   |-- emowork.yaml
|   `-- case.yaml
|-- train.py                # one fold, selected folds, or complete LOSO
|-- evaluate.py             # checkpoint-only evaluation
|-- smoke_test.py           # dataset-free forward/backward verification
`-- verify_release.py       # architecture/configuration consistency checks
```

Internal ablation, plotting, and reviewer-analysis scripts are not required to
run CARL and are therefore outside this core release.

## Dataset access

Dataset files are not redistributed. Obtain them from their official sources
and follow their access conditions:

- **WESAD:** [dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) and [original paper](https://doi.org/10.1145/3242969.3242985).
- **EmoWork:** [dataset paper](https://doi.org/10.1038/s41597-025-06531-2), [controlled-access dataset](https://doi.org/10.5281/zenodo.15181219), and [official technical-validation code](https://github.com/Kaist-ICLab/EmoWork).
- **CASE:** [dataset paper](https://doi.org/10.1038/s41597-019-0209-0) and [official dataset collection](https://doi.org/10.6084/m9.figshare.c.4260668).

Dataset-specific raw preprocessing code and derived dataset files are excluded
because the acquired data are governed by their respective provider terms and,
for controlled-access data, the signed data-use agreement. The exact signal
selection, filtering, label construction, invalid-window handling, and window
counts are reported in the manuscript. The common settings are 256-Hz
resampling and non-overlapping 10-s windows.

## Preprocessed data interface

Each dataset is supplied to the release as one NumPy archive:

```python
np.savez(
    "WESAD_preprocessed.npz",
    X=X,                    # float array [number_of_windows, 2560, channels]
    y=y,                    # integer class labels [number_of_windows]
    subject_ids=subjects,  # subject identifier for every window
)
```

Channel order must match the chosen file in `configs/`. The released data layer
groups all channels belonging to one physiological modality into one graph
node; for example, CASE's three EMG channels form a single EMG node and
EmoWork's four EEG channels form one EEG node. It does not average those
channels. After each LOSO split, per-channel mean and standard deviation are
estimated **only from the training subjects** and applied unchanged to the
validation and held-out test subjects. These statistics are stored in every
checkpoint.

The inspected WESAD and CASE preprocessing outputs already use this schema.
The inspected EmoWork output stores separate modality arrays, so package it
without reprocessing the signals:

```bash
python prepare_emowork_archive.py \
  --input /path/to/EmoWork_all.npz \
  --output /path/to/EmoWork_valence_preprocessed.npz
```

The script selects the valence column and concatenates EEG, ECG, EDA, BVP, and
TEMP in the channel order declared by `configs/emowork.yaml`; ACC is not used.

## Installation and verification

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python verify_release.py
```

The smoke test uses synthetic tensors only. It verifies construction, the
forward pass, all four losses, back-propagation, and relationship updating; it
does not report an experimental result.

## Reproduce LOSO training

Run one held-out subject first:

```bash
python train.py \
  --config configs/wesad.yaml \
  --data /path/to/WESAD_preprocessed.npz \
  --subjects 2 \
  --output-dir results/wesad
```

Run the complete LOSO experiment by replacing `--subjects 2` with
`--subjects all`. Equivalent commands use `configs/emowork.yaml` and
`configs/case.yaml`. Each fold writes the best validation-accuracy checkpoint,
epoch history, and held-out-subject result. `summary.json` reports the
subject-level mean and standard deviation of Accuracy and Weighted-F1.

Evaluate a saved fold without retraining:

```bash
python evaluate.py \
  --checkpoint results/wesad/subject_2/best_model.pt \
  --data /path/to/WESAD_preprocessed.npz \
  --device cuda
```

The held-out test subject is never used for normalization, hyperparameter
selection, or checkpoint selection.

## Method-to-code map

- **Graph backbone:** `ConflictAwareGraphFusion._semantic_graph` and
  `GraphBiasedAttention`.
- **FD:** `FeatureDecomposer` and `FeatureDecoder`.
- **PSL:** `ConditionalSemanticPredictor`, including directional prediction
  errors.
- **CG:** `RelationshipTracker` and signed modulation inside
  `ConflictAwareGraphFusion.forward`.
- **Objective:** `compute_carl_loss`, with
  `L_cls + 0.1 L_recon + 0.1 L_ortho + 1.0 L_sem`.

## Citation and license

Add the final article citation and repository DOI after acceptance. A public
release should also include the author-selected software license; no license is
asserted here because that is an author/legal choice rather than a model or
reproducibility setting.
