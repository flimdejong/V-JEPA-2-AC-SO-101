"""
convert_to_delta_ee.py

Converts a LeRobot-format dataset's `action` feature from joint-space targets
into end-effector-space deltas (position delta + rotation delta as an
axis-angle vector + gripper delta)
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from scipy.spatial.transform import Rotation as R

from lerobot.lerobot_types import RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    transition_to_observation,
)
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE

def compute_ee_delta(pose_t: dict, pose_t_plus_1: dict) -> np.ndarray:
    """
    Compute the 7-dim motion [dx, dy, dz, dwx, dwy, dwz, dgripper] that takes
    the end-effector from pose_t to pose_t_plus_1.
    """
    pos_t = np.array([pose_t["ee.x"], pose_t["ee.y"], pose_t["ee.z"]])
    pos_t_plus_1 = np.array([pose_t_plus_1["ee.x"], pose_t_plus_1["ee.y"], pose_t_plus_1["ee.z"]])
    delta_position = pos_t_plus_1 - pos_t

    rotation_t = R.from_rotvec([pose_t["ee.wx"], pose_t["ee.wy"], pose_t["ee.wz"]])
    rotation_t_plus_1 = R.from_rotvec(
        [pose_t_plus_1["ee.wx"], pose_t_plus_1["ee.wy"], pose_t_plus_1["ee.wz"]]
    )
    rotation_delta = rotation_t.inv() * rotation_t_plus_1
    delta_rotvec = rotation_delta.as_rotvec()

    delta_gripper = pose_t_plus_1["ee.gripper_pos"] - pose_t["ee.gripper_pos"]

    return np.concatenate([delta_position, delta_rotvec, [delta_gripper]]).astype(np.float32)

def build_fk_processor(urdf_path: str, motor_names: list[str]):
    """Build the forward-kinematics pipeline once, reused for every frame."""
    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=motor_names)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

def joints_to_ee_pose(joint_positions, motor_names: list[str], fk_processor) -> dict:
    """Run forward kinematics on one frame's joint positions, return the EE pose dict."""
    raw_joint_observation = {
        f"{name}.pos": float(val) for name, val in zip(motor_names, joint_positions)
    }
    return fk_processor(raw_joint_observation)

def process_and_write_all(dst_root: Path, motor_names, fk_processor) -> None:
    parquet_paths = sorted(dst_root.glob("data/chunk-*/*.parquet"))

    for pq_path in parquet_paths:
        df = pd.read_parquet(pq_path)
        actions = pd.Series(index=df.index, dtype=object)

        for ep_id, ep_group in df.groupby("episode_index"): # per episode
            group = ep_group.sort_values("frame_index")
            index_labels = group.index.tolist()
            poses = [joints_to_ee_pose(s, motor_names, fk_processor) for s in group["observation.state"]]

            for i in range(len(index_labels) - 1):
                actions.loc[index_labels[i]] = compute_ee_delta(poses[i], poses[i + 1])
            actions.loc[index_labels[-1]] = np.zeros(7, dtype=np.float32)

        df["action"] = actions
        df.to_parquet(pq_path)

def main():
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    fk_processor = build_fk_processor(
        urdf_path="./SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        motor_names=motor_names,
    )

    repo_id = "Flimdejong/block_or_pen"
    src_snapshot = Path(snapshot_download(repo_id, repo_type="dataset"))

    # Copy to our managed root
    dst_root = Path("~/lerobot_datasets").expanduser()
    shutil.copytree(src_snapshot, dst_root, dirs_exist_ok=True)

    process_and_write_all(dst_root, motor_names, fk_processor)
    print(f"Done: {dst_root}")

if __name__ == "__main__":
    main()