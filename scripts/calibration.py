"""
Axis calibration (2D): which way does the model think the robot's x and z point?
X is forward/backwards, z is up/down, from the robots perspective.
"""

import time

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.keyboard_input import init_keyboard_listener

from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

MODEL_NAME = "jepa_wm_droid"
OFFSET = 0.05        # metres the arm moves along each axis
PROBE = 0.05         # metres, size of the move the predictor is asked about
N_AVG = 8
FPS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AXES = ["x", "y", "z"]     # names of the three axes (index 0, 1, 2)
ROBOT_AXES = [0, 2]        # the robot axes we test: we only do 2D

def main():
    camera_config = {"top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
    follower = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM1", id="follower_2", cameras=camera_config, use_degrees=True))
    leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", id="leader_2", use_degrees=True))
    motors = list(follower.bus.motors.keys())

    kinematics = RobotKinematics(
        urdf_path="external/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=motors,
    )
    joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motors)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    ee_to_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.3, -0.3, -0.1], "max": [0.3, 0.3, 0.4]},
                max_ee_step_m=0.05,
            ),
            InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=motors, initial_guess_current_joints=True),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    print("Loading model...")
    model, _ = torch.hub.load("facebookresearch/jepa-wms", MODEL_NAME)
    model.to(DEVICE).eval()

    follower.connect()
    leader.connect()
    listener, events = init_keyboard_listener()

    try:
        print(f"\nMove the LEADER to a start pose with {OFFSET * 100:.0f}+ cm free forward, back, up and down, then press 'n'.")
        while not events["exit_early"]:
            follower.send_action(leader.get_action())
            time.sleep(1.0 / FPS)
        time.sleep(1.0)

        ee = joints_to_ee(follower.get_observation())
        start = np.array([ee["ee.x"], ee["ee.y"], ee["ee.z"]])
        rot = (ee["ee.wx"], ee["ee.wy"], ee["ee.wz"])       # orientation and gripper are held fixed
        grip = ee["ee.gripper_pos"]

        def position():
            e = joints_to_ee(follower.get_observation())
            return np.array([e["ee.x"], e["ee.y"], e["ee.z"]])

        def move(from_xyz, to_xyz):
            """Move the gripper in a straight line in 12 small steps, then wait for it to settle."""
            for p in np.linspace(from_xyz, to_xyz, 13)[1:]:
                action = {"ee.x": float(p[0]), "ee.y": float(p[1]), "ee.z": float(p[2]),
                          "ee.wx": float(rot[0]), "ee.wy": float(rot[1]), "ee.wz": float(rot[2]),
                          "ee.gripper_pos": float(grip)}
                follower.send_action(ee_to_joints((action, follower.get_observation())))
                time.sleep(0.05)
            time.sleep(1.0)

        def look():
            """Encode N_AVG frames and average the latents (same as the planning script)."""
            zs = []
            for _ in range(N_AVG):
                frame = follower.get_observation()["top"]
                with torch.no_grad():
                    zs.append(model.encode(torch.from_numpy(frame).permute(2, 0, 1)[None, None]))
                time.sleep(0.05)
            return torch.stack(zs).mean(0)

        print("\nThe arm now moves on its own. Hands out of view.")
        z_start = look()
        goals = []                                          # (axis, sign, latent)
        for axis in ROBOT_AXES:
            for sign in (+1, -1):
                target = start.copy()
                target[axis] += sign * OFFSET
                move(start, target)
                reached = (position() - start) * 100        # cm
                label = f"{'+' if sign > 0 else '-'}{AXES[axis]}"
                print(f"  robot {label}: reached dx={reached[0]:+.1f} dy={reached[1]:+.1f} dz={reached[2]:+.1f} cm")
                goals.append((axis, sign, look()))
                move(target, start)

        P = PROBE
        probes = torch.tensor([
            # dx  dy  dz  rotation  gripper
            [ 0,  0,  0,  0, 0, 0,  0],    # stay
            [+P,  0,  0,  0, 0, 0,  0],    # +x
            [-P,  0,  0,  0, 0, 0,  0],    # -x
            [ 0, +P,  0,  0, 0, 0,  0],    # +y
            [ 0, -P,  0,  0, 0, 0,  0],    # -y
            [ 0,  0, +P,  0, 0, 0,  0],    # +z
            [ 0,  0, -P,  0, 0, 0,  0],    # -z
        ], dtype=torch.float32, device=DEVICE)[None]

        with torch.no_grad():
            predicted = model.unroll(z_start, act_suffix=probes)

        response = np.zeros((3, 3))
        for axis, sign, z_goal in goals:
            # L2 objective from jepa-wms
            objective = ReprTargetDistMPCObjective(cfg={}, target_enc=z_goal, sum_all_diffs=False, alpha=0)
            c = objective(predicted, probes).cpu().numpy()
            stay, plus_x, minus_x, plus_y, minus_y, plus_z, minus_z = c

            # For each model axis: positive if the imagined "+" move is the closer one.
            prefer = np.array([minus_x - plus_x,
                               minus_y - plus_y,
                               minus_z - plus_z])

            # Add it to the row for this robot axis (flipped for the "-" goal), averaging the two.
            response[axis] += sign * prefer / 2

        response *= 1000

        print("\n===== Summary: when the robot really moves along one axis, what does the model prefer? =====")
        print("(positive = same direction, negative = opposite direction, near 0 = no opinion)")
        print(f"{'':<12}{'model x':>10}{'model y':>10}{'model z':>10}")
        for a in ROBOT_AXES:
            print(f"robot {AXES[a]:<6}" + "".join(f"{v:>10.2f}" for v in response[a]))

        print("RESPONSE = np.array([")
        for a in range(3):
            note = "" if a in ROBOT_AXES else " (not used)"
            print(f"    [{response[a, 0]:6.2f}, {response[a, 1]:6.2f}, {response[a, 2]:6.2f}],    # robot {AXES[a]}{note}")
        print("])")

    finally:
        follower.disconnect()
        leader.disconnect()
        listener.stop()


if __name__ == "__main__":
    main()