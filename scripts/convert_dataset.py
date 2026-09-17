"""
convert_dataset.py

Finds a LeRobot v3.0 dataset already downloaded under
datasets/raw/<repo_id>/ (via download_lerobot_datasets.py), computes
7D EE-space deltas for every row, and overwrites 'action' with those
deltas IN PLACE, according to SO-101 kinematics. The new 7D deltas are:

    action := [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
    , where dx,dy,dz are in meters, d_roll/pitch/yaw are in radians/ euler angles, and d_gripper is normalized to [0,1].

meta/info.json and meta/stats.json are patched in place to match the new 7D action. 
Video files are left
untouched. Refuses to run if the dataset's action is already 7D
(i.e. already converted).

python convert_dataset.py --repo-id Flimdejong/block_or_pen
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from lerobot.lerobot_types import RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    transition_to_observation,
)
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Per LeRobot docs, prefer the URDF from the SO-ARM100 repo:
# https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
URDF_PATH = "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
DELTA_LABELS = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "d_gripper"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 7D EE-space deltas and overwrite 'action' in place."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help='Dataset repo id, e.g. "Flimdejong/block_or_pen".',
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="datasets",
        help="Base datasets folder, matching download_lerobot_datasets.py's --datasets_dir.",
    )
    parser.add_argument(
        "--raw-subdir",
        type=str,
        default="raw",
        help="Subfolder under --datasets-dir holding raw downloads.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=URDF_PATH,
        help="Path to the robot's URDF file for forward kinematics.",
    )
    parser.add_argument(
        "--target-frame",
        type=str,
        default=TARGET_FRAME,
        help="URDF frame name to compute the end-effector pose at.",
    )
    return parser.parse_args()


def find_raw_dataset_dir(datasets_dir: str, raw_subdir: str, repo_id: str) -> Path:
    """
    Locate the datasets/raw/<repo_id>/ folder that download_lerobot_datasets.py
    already created, and confirm it actually has data in it.
    """
    raw_root = Path(datasets_dir) / raw_subdir / repo_id
    if not raw_root.exists():
        raise FileNotFoundError(
            f"No raw dataset found at {raw_root}. "
            "Run download_lerobot_datasets.py --repo_ids '[\"" + repo_id + "\"]' first."
        )

    parquet_files = sorted(raw_root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {raw_root}/data/. "
            "Is this a fully-downloaded LeRobot v3.0 dataset?"
        )

    print(f"Found dataset at {raw_root} ({len(parquet_files)} parquet file(s))")
    return raw_root

def build_fk_processor(
    motor_names: list[str],
    urdf_path: str = "./SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    target_frame: str = "gripper_frame_link",
):
    """
    Build the joint-space -> EE-space kinematics pipeline.
    """
    # Define kinematics
    follower_kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame,
        joint_names=motor_names,
    )

    # Convert follower joints to EE observation.
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_kinematics_solver, motor_names=motor_names
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return follower_joints_to_ee

def compute_ee_delta_row(pose_state: dict, pose_action: dict) -> np.ndarray:
    """
    Compute the 7-dim motion [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
    that takes the end-effector from pose_state to pose_action.
    """
    # Position delta
    pos_state = np.array([pose_state["ee.x"], pose_state["ee.y"], pose_state["ee.z"]])
    pos_action = np.array([pose_action["ee.x"], pose_action["ee.y"], pose_action["ee.z"]])
    delta_position = pos_action - pos_state

    # Rotation delta
    rotation_state = R.from_rotvec(
        [pose_state["ee.wx"], pose_state["ee.wy"], pose_state["ee.wz"]]
    )
    rotation_action = R.from_rotvec(
        [pose_action["ee.wx"], pose_action["ee.wy"], pose_action["ee.wz"]]
    )
    rotation_delta = rotation_state.inv() * rotation_action
    delta_euler = rotation_delta.as_euler("xyz", degrees=False)

    # Gripper delta. Lerobot goes from 0-100, JEPA expects 0-1
    delta_gripper = (pose_action["ee.gripper_pos"] - pose_state["ee.gripper_pos"])/100

    return np.concatenate([delta_position, delta_euler, [delta_gripper]]).astype(np.float32)

def compute_deltas_for_file(
    parquet_path: Path, motor_names: list[str], fk_processor
) -> np.ndarray:
    """
    Load one parquet file, replace its 'action' column with the 7D
    EE-space delta, overwrite the SAME file in place, and return the
    delta array so main() can aggregate stats across all files.
    """
    df = pd.read_parquet(parquet_path)
    deltas = []

    for row_idx in range(len(df)):
        current_row = df.iloc[row_idx]

        state_joints = np.asarray(current_row["observation.state"], dtype=np.float64)
        action_joints = np.asarray(current_row["action"], dtype=np.float64)

        state_dict = build_motor_dict(motor_names, state_joints)
        action_dict = build_motor_dict(motor_names, action_joints)

        ee_pose_state = fk_processor(state_dict)
        ee_pose_action = fk_processor(action_dict)

        deltas.append(compute_ee_delta_row(ee_pose_state, ee_pose_action))

    delta_array = np.stack(deltas)  # (num_rows, 7)
    df["action"] = list(delta_array)
    df.to_parquet(parquet_path)  # overwrite in place, same path
    print(f"  patched {parquet_path} ({len(df)} rows)")
    return delta_array

def build_motor_dict(motor_names: list[str], joint_values: np.ndarray) -> dict:
    """
    Zip joint values with motor names into the {motor}.pos dict
    the LeRobot FK pipeline expects.
    """

    motor_dict = {}
    for i, name in enumerate(motor_names):
        value = joint_values[i]
        motor_dict[f"{name}.pos"] = float(value)

    return motor_dict

def patch_info_json(raw_root: Path) -> None:
    """Update meta/info.json's 'action' feature to reflect the new 7D shape/names."""
    info_path = raw_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"  WARNING: {info_path} not found, skipping info.json patch.")
        return

    info = json.loads(info_path.read_text())
    if "action" not in info.get("features", {}):
        print("  WARNING: 'action' not found in info.json features, skipping.")
        return

    info["features"]["action"]["shape"] = [7]
    info["features"]["action"]["names"] = DELTA_LABELS
    info_path.write_text(json.dumps(info, indent=2))
    print(f"  patched {info_path}")


def patch_stats_json(raw_root: Path, all_deltas: np.ndarray) -> None:
    """
    Update meta/stats.json's 'action' entry with mean/std/min/max computed
    from the actual 7D deltas just written, across all parquet files.
    """
    stats_path = raw_root / "meta" / "stats.json"
    if not stats_path.exists():
        print(f"  No {stats_path} found, skipping (nothing to update).")
        return

    stats = json.loads(stats_path.read_text())
    stats["action"] = {
        "mean": all_deltas.mean(axis=0).tolist(),
        "std": all_deltas.std(axis=0).tolist(),
        "min": all_deltas.min(axis=0).tolist(),
        "max": all_deltas.max(axis=0).tolist(),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  patched {stats_path}")
    print(
        "  NOTE: verify stats.json's key names (mean/std/min/max) match this "
        "LeRobot version's convention — some versions use different keys "
        "(e.g. quantiles) that this script doesn't know to update."
    )

def already_converted(raw_root: Path) -> bool:
    info_path = raw_root / "meta" / "info.json"
    if not info_path.exists():
        return False

    info = json.loads(info_path.read_text())
    action_feat = info.get("features", {}).get("action", {})
    shape = action_feat.get("shape")

    if shape == [7]:
        names = action_feat.get("names")
        if names is not None and names != DELTA_LABELS:
            print(
                f"  WARNING: action shape is 7D but names={names} != {DELTA_LABELS}. "
                "Treating as already-converted, but double-check this dataset's history."
            )
        return True

    return False

def main():
    args = parse_args()

    raw_root = find_raw_dataset_dir(args.datasets_dir, args.raw_subdir, args.repo_id)

    if already_converted(raw_root):
        raise RuntimeError(
            f"{raw_root} already has 7D EE-delta actions (per meta/info.json). "
            "Re-running would convert deltas of deltas. Re-download if you need a fresh raw copy."
        )

    parquet_files = sorted(raw_root.glob("data/**/*.parquet"))

    fk_processor = build_fk_processor(MOTOR_NAMES, args.urdf_path, args.target_frame)

    print(f"Patching {len(parquet_files)} parquet file(s) in place...")
    all_deltas = []
    for parquet_path in parquet_files:
        deltas = compute_deltas_for_file(parquet_path, MOTOR_NAMES, fk_processor)
        all_deltas.append(deltas)
    all_deltas = np.concatenate(all_deltas, axis=0)

    print("Patching metadata to match...")
    patch_info_json(raw_root)
    patch_stats_json(raw_root, all_deltas)

    print(f"Done. {len(all_deltas)} rows now have 7D EE-space 'action', metadata updated.")
    print("Video files were not touched.")


if __name__ == "__main__":
    main()