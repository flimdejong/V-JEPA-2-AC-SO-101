# V-JEPA-2-AC-SO-101

This repository seeks to run V-JEPA-2-AC on an SO-101 arm, with additional support for fine-tuning the predictor (not finished).

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
    uv pip install numpy pandas scipy torch draccus lerobot[dataset,placo-dep,feetech] h5py nevergrad tensordict

## Zero-shot goal reaching with JEPA-WM (`deply.py`)

Moves the SO-101 gripper toward a goal image using a pretrained world model, without any training on this robot.

The model is `jepa_wm_droid` from [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms): a frozen DINOv3 ViT-L encoder with a predictor trained on DROID (Franka arms). At every step the script

1. encodes the current camera image and compares it with the encoded goal image,
2. uses CEM planning with the predictor to find the action that brings the predicted image closest to the goal,
3. translates that action from the model's axes to the robot's axes (see Calibration),
4. executes a small step and repeats.

Control is limited to the vertical plane seen by the side camera: **x (forward/back) and z (up/down)**. Sideways position, wrist orientation and gripper opening are held at their start values for simplicity.

### Requirements

- SO-101 follower and leader arm, set up with LeRobot.
- One fixed camera looking at the arm from the side where the 2D forward/back and up/down is clearly visible.
- The dependencies listed above

### Calibration (do this first)

The model was trained on a different robot, so its idea of x, y and z does not match the SO-101. `calibration.py` measures the mismatch:

```bash
python calibration.py
```

Place the arm at a start pose and press `n`. The arm moves 5 cm forward, back, up and down on its own. The script prints a `RESPONSE` table; paste it into `plan_dinov3.py`.

**Redo the calibration whenever the camera is moved.**

### Usage

Scripted goal (recommended for measurements). You place the arm at the start; the arm moves to the goal by itself, records it, and returns:

```bash
python scripts/deploy.py                           # goal 5 cm forward
python scripts/deploy.py --direction up            # goal 5 cm up
python scripts/deploy.py --direction back --cm 3   # goal 3 cm back
python scripts/deploy.py --cm 0 --steps 3          # no movement: measures the noise floor of the latent distance
```

Hand-set goal. You place the arm at the goal, then at the start:

```bash
python scripts/deploy.py --teleop
```

| Flag | Meaning | Default |
|---|---|---|
| `--direction` | `forward`, `back`, `up` or `down` | `forward` |
| `--cm` | distance from start to goal, in cm | `5` |
| `--steps` | number of planning steps before the run stops | `10` |
| `--horizon` | planning horizon. The model's config uses 3; 1 is what worked on this arm | `1` |
| `--teleop` | set the goal by hand (ignores `--direction` and `--cm`) | off |
| `--name` | name of the run folder | `reach_<direction>_<cm>cm` or `reach_teleop` |

Press `n` to confirm a pose. The run stops after `--steps` steps, or earlier with `Esc`. Keep hands out of the camera view.

### Saved results and plots

Every run gets its own folder. Repeating a command never overwrites an earlier run; a number is added to the name instead.

```
runs/
  reach_forward_5cm/
    data.csv       one row per step: latent distance, gripper position, goal position,
                   true distance, planned move, action score
    config.json    flags, model name, calibration table and goal position used
    start.png      camera image at the first step
    goal.png       camera image at the goal
    final.png      camera image after the last step
  reach_forward_5cm_2/
    ...
```

Plot one or more runs:

```bash
python scripts/plot_runs.py runs/reach_forward_5cm/data.csv runs/reach_up_5cm/data.csv
python scripts/plot_runs.py runs/*/data.csv
```

This writes `runs_plot.png` with, per run, the true distance and latent distance per step, and the gripper path in the x-z plane. It also prints a summary table.


## Data pipeline

**1. Download the raw LeRobot dataset**

    python external/lerobot-dataset-to-HDF5/scripts/download_lerobot_datasets.py \
        --repo_ids "[Flimdejong/block_or_pen]" \
        --datasets_dir data

Downloads to `data/raw/Flimdejong/block_or_pen/`.

**2. Convert joint-space actions to 7D EE-space deltas**

    python scripts/convert_dataset.py --repo-id Flimdejong/block_or_pen \
        --datasets-dir data

Overwrites `action` in place and patches `meta/info.json` / `meta/stats.json` to match.
This step refuses to re-run on a dataset that's already been converted.

**3. Convert to HDF5**

    python external/lerobot-dataset-to-HDF5/scripts/convert_lerobot_to_hdf5.py \
        --repo_id Flimdejong/block_or_pen \
        --datasets_dir data \
        --raw_subdir raw \
        --hdf5_subdir hdf5

Writes to `data/hdf5/Flimdejong/block_or_pen/`.