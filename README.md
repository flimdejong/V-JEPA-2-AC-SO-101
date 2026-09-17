# V-JEPA-2-AC-SO-101

This repository seeks to run V-JEPA-2-AC on an SO-101 arm, with additional support for fine-tuning the predictor.

## Setup

Environment is managed with `uv`. See: https://docs.astral.sh/uv/getting-started/installation/

Clone with submodules:

    git clone --recurse-submodules https://github.com/flimdejong/V-JEPA-2-AC-SO-101.git
    cd V-JEPA-2-AC-SO-101

All commands below are run from the repo root.

### Submodules

| Path | Purpose |
|---|---|
| `external/jepa-wms` | JEPA-WM / V-JEPA-2-AC model, training and planning code |
| `external/lerobot` | LeRobot fork (kinematics, processors, SO-101 follower support) |
| `external/lerobot-dataset-to-HDF5` | Download raw LeRobot datasets and convert to HDF5 |
| `external/SO-ARM100` | SO-101 URDF and hardware description |

### Install dependencies

    uv venv
    source .venv/bin/activate
    uv pip install -e external/lerobot
    uv pip install -e external/jepa-wms
    uv pip install numpy pandas scipy torch draccus

## Data pipeline

**1. Download the raw LeRobot dataset**

    python external/lerobot-dataset-to-HDF5/scripts/download_lerobot_dataset.py \
        --repo_ids "[Flimdejong/block_or_pen]" \
        --datasets_dir data

Downloads to `data/raw/Flimdejong/block_or_pen/`.

**2. Convert joint-space actions to 7D EE-space deltas**

    python convert_dataset.py --repo-id Flimdejong/block_or_pen \
        --datasets-dir data

Overwrites `action` in place and patches `meta/info.json` / `meta/stats.json` to match.
This step refuses to re-run on a dataset that's already been converted.

**3. Convert to HDF5**

    python external/lerobot-dataset-to-HDF5/scripts/convert_lerobot_to_hdf5.py \
        --repo_id Flimdejong/block_or_pen \
        --datasets_dir data \
        --raw_subdir raw \
        --hdf5_subdir hdf5 \
        --splits '["train"]'

Writes to `data/hdf5/Flimdejong/block_or_pen/`.