import argparse
import torch
from scipy.spatial.transform import Rotation
import time
import csv
import os
import numpy as np
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
from lerobot.utils.utils import log_say

import torchvision
from evals.simu_env_planning.planning.planning.planner import CEMPlanner
from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

MODEL_NAME = "jepa_wm_droid"
FPS = 30
MOCK = False
DRY_RUN = False
MOVE_AXES = [0, 2]   # robot x and z

# Calibrate your robot before use with calibration.py!!!
RESPONSE = np.array([
    [-2.62, -4.54, 2.69],    # robot x
    [ 0.0,  0.0, 0.0],       # robot y
    [-0.76, -2.10, 3.23],    # robot z
])
DIRECTIONS = {"forward": [1, 0, 0], "back": [-1, 0, 0], "up": [0, 0, 1], "down": [0, 0, -1]}
N_AVG = 8 # Number of average frames
LEADER_PORT = "/dev/ttyACM0"
LEADER_ID = "leader_2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=list(DIRECTIONS), default="forward")
    parser.add_argument("--cm", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--name", default=None)
    parser.add_argument("--teleop", action="store_true")     # set the goal by hand instead of a scripted move
    args = parser.parse_args()
    goal_offset = np.array(DIRECTIONS[args.direction]) * args.cm / 100
    run_name = args.name or ("reach_teleop" if args.teleop else f"reach_{args.direction}_{args.cm:g}cm")

    # Create the robot and follower configurations
    camera_config = {"top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
    follower_config = SO100FollowerConfig(
        port="/dev/ttyACM1",
        id="follower_2",
        cameras=camera_config,
        use_degrees=True,
    )

    # Initialize the follower and leader robot
    follower = SO100Follower(follower_config)
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID, use_degrees=True))

    # Define kinematics
    follower_kinematics_solver = RobotKinematics(
        urdf_path="external/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    # Convert follower joints to EE observation.
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_kinematics_solver, motor_names=list(follower.bus.motors.keys())
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Convert EE action to follower joints (with safety bounds).
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.3, -0.3, -0.1], "max": [0.3, 0.3, 0.4]},
                max_ee_step_m=0.05,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect the robots
    follower.connect()
    leader.connect()

    # Initialize keyboard listener
    listener, events = init_keyboard_listener()

    log_rows = []

    try:
        if not follower.is_connected:
            raise ValueError("Follower is not connected!")

        def teleop_until_n(message):
            print(message)
            while not events["exit_early"]:
                follower.send_action(leader.get_action())
                time.sleep(1.0 / FPS)
            events["exit_early"] = False
            time.sleep(1.0)

        def capture_goal():
            joint_obs = follower.get_observation()
            frames = [joint_obs["top"].copy()]
            for _ in range(N_AVG - 1):
                time.sleep(0.05)
                frames.append(follower.get_observation()["top"].copy())
            ee = follower_joints_to_ee(joint_obs)
            return frames, np.array([ee["ee.x"], ee["ee.y"], ee["ee.z"]], dtype=float)

        if args.teleop:
            # Goal set by hand
            teleop_until_n("Move the LEADER arm to the GOAL pose, keep your hands out of view, then press 'n'.")
            goal_frames, goal_xyz = capture_goal()
            teleop_until_n("Now move the LEADER arm to the START pose, then press 'n'.")
        else:
            # Scripted goal: move by goal_offset, capture, move back. Wrist and gripper stay as at the start.
            teleop_until_n(f"Goal: {args.cm:g} cm {args.direction}. Move the LEADER arm to the START pose, then press 'n'.")
            start_ee = follower_joints_to_ee(follower.get_observation())
            start_pose = {k: float(v) for k, v in start_ee.items() if k.startswith("ee.")}
            start_xyz = np.array([start_pose["ee.x"], start_pose["ee.y"], start_pose["ee.z"]])

            def move(from_xyz, to_xyz):
                for p in np.linspace(from_xyz, to_xyz, 13)[1:]:
                    action = {**start_pose, "ee.x": float(p[0]), "ee.y": float(p[1]), "ee.z": float(p[2])}
                    follower.send_action(ee_to_follower_joints((action, follower.get_observation())))
                    time.sleep(0.05)
                time.sleep(1.0)

            print("The arm now moves to the goal on its own.")
            move(start_xyz, start_xyz + goal_offset)
            goal_frames, goal_xyz = capture_goal()
            reached = (goal_xyz - start_xyz) * 100
            print(f"Goal captured. Real offset from start: dx={reached[0]:+.1f} dy={reached[1]:+.1f} dz={reached[2]:+.1f} cm")
            move(start_xyz + goal_offset, start_xyz)

        goal_frame = goal_frames[0]

        ###########################################
        #### Initializing model and CEM planner ###
        ###########################################
        if not MOCK:
            model, preprocessor = torch.hub.load("facebookresearch/jepa-wms", MODEL_NAME)
            model.to(DEVICE).eval()

            def encode_avg(frames):
                with torch.no_grad():
                    encs = [model.encode(torch.from_numpy(f).permute(2, 0, 1)[None, None])   # raw uint8, (1,1,3,H,W)
                            for f in frames]
                return torch.stack(encs).mean(0)

            def save_model_view(frame, path):
                view = preprocessor.transform_obs_visual(frame[None, None])[0]
                torchvision.utils.save_image(preprocessor.inverse_transform(view), path)

            goal_enc = encode_avg(goal_frames).detach()
            os.makedirs("runs", exist_ok=True)
            save_model_view(goal_frame, f"runs/{run_name}_goal.png")

            objective = ReprTargetDistMPCObjective(
                cfg={},
                target_enc=goal_enc,
                sum_all_diffs=False,
                alpha=0, # alpha = 0, meaning no proprio
            )

            planner = CEMPlanner(
                unroll=model.unroll,
                action_dim=model.action_dim,
                horizon=1,
                num_samples=300,
                num_elites=10,
                iterations=15,
                var_scale=0.1,
                max_norms=[0.1, 0.75],
                max_norm_dims=[[0, 1, 2, 3, 4, 5], [6]],
                num_act_stepped=1,
            )
            planner.set_objective(objective)

        target_pose = None
        step = 0

        for _ in range(30):              # ~1 s of frames, discarded
            follower.get_observation()
            time.sleep(1.0 / FPS)

        while True:

            if events["stop_recording"] or step >= args.steps:
                print("Stopping.")
                break

            joint_obs = follower.get_observation()
            ee_obs = follower_joints_to_ee(joint_obs)   # FK --> 7D

            # rotation vector -> Euler angles
            rotvec = [ee_obs["ee.wx"], ee_obs["ee.wy"], ee_obs["ee.wz"]]
            euler = Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=False)      # -> [roll, pitch, yaw] in radians

            current_pose = torch.tensor([
                ee_obs["ee.x"], ee_obs["ee.y"], ee_obs["ee.z"],
                *euler,
                ee_obs["ee.gripper_pos"] / 100.0,
            ], device=DEVICE)

            if MOCK == True:
                if target_pose is None:
                    target_pose = current_pose.clone()            # fixed start target
                if step < 60:                                     # move for 2 s, then hold
                    target_pose[1] += 0.001                       # +y, 1 mm per step
                step += 1
                delta_action = target_pose - current_pose         # current + delta == target
                delta_action[6] = target_pose[6]                  # gripper line below uses this as absolute
            else:
                frames = [joint_obs["top"]]
                for _ in range(N_AVG - 1):
                    time.sleep(0.05)
                    frames.append(follower.get_observation()["top"])
                z = encode_avg(frames)

                dist = (z - goal_enc).pow(2).mean().item()        # L2, same measure the planner minimises
                print(f"\nstep {step}: latent L2 to goal: {dist:.5f}")

                if step == 0:
                    save_model_view(frames[0], f"runs/{run_name}_start.png")
                step += 1

                result = planner.plan(z, steps_left=None)
                delta_action = preprocessor.denormalize_actions(result.actions[0].unsqueeze(0).cpu()).squeeze(0).to(DEVICE)

                raw = delta_action.detach().cpu().numpy()
                print(f"planner wants: dx={raw[0]:+.4f} dy={raw[1]:+.4f} dz={raw[2]:+.4f} m | "
                      f"rot=({raw[3]:+.3f}, {raw[4]:+.3f}, {raw[5]:+.3f}) rad | grip={raw[6]:+.3f}")

                # Calibration
                model_xyz = delta_action[0:3].detach().cpu().numpy()
                robot_xyz = np.zeros(3)
                robot_xyz[MOVE_AXES] = np.linalg.pinv(RESPONSE[MOVE_AXES].T) @ model_xyz
                if np.linalg.norm(robot_xyz) > 0:
                    robot_xyz *= np.linalg.norm(model_xyz) / np.linalg.norm(robot_xyz)
                print(f"translated to robot: dx={robot_xyz[0]:+.4f} dy={robot_xyz[1]:+.4f} dz={robot_xyz[2]:+.4f} m")
                delta_action[0:3] = torch.tensor(robot_xyz, device=DEVICE, dtype=delta_action.dtype)

                # Measure the real distance to the goal
                pos = current_pose[0:3].cpu().numpy()
                to_goal = goal_xyz - pos
                dist_xz_cm = np.hypot(to_goal[0], to_goal[2]) * 100

                # Action score: does the planned move point at the goal? +1 yes, 0 sideways, -1 away.
                ideal = to_goal[[0, 2]]
                planned = robot_xyz[[0, 2]]
                action_score = ideal @ planned / (np.linalg.norm(ideal) * np.linalg.norm(planned))

                print(f"true distance to goal: {dist_xz_cm:.1f} cm "
                      f"[still to go: dx={to_goal[0]*100:+.1f} dy={to_goal[1]*100:+.1f} dz={to_goal[2]*100:+.1f} cm] "
                      f"| action score {action_score:+.2f}")
                log_rows.append([step - 1, dist, *pos, *goal_xyz, dist_xz_cm, robot_xyz[0], robot_xyz[2], action_score])

                # --- Heavy clamp for initial testing ---
                MAX_POS_DELTA = 0.02   # meters per axis
                MAX_ROT_DELTA = 0.1    # radians per axis (euler)
                MAX_GRIPPER_DELTA = 0.25  # 0-1 scale, since gripper_pos is later *100

                delta_action = torch.cat([
                    torch.clamp(delta_action[0:3], -MAX_POS_DELTA, MAX_POS_DELTA),   # x, y, z
                    torch.clamp(delta_action[3:6], -MAX_ROT_DELTA, MAX_ROT_DELTA),   # euler
                    torch.clamp(delta_action[6:7], -MAX_GRIPPER_DELTA, MAX_GRIPPER_DELTA),  # gripper
                ])

                if target_pose is None:
                    target_pose = current_pose.clone()
                target_pose[MOVE_AXES] += delta_action[MOVE_AXES]
                # keep the target at most 3 cm ahead of the real arm
                target_pose[MOVE_AXES] = current_pose[MOVE_AXES] + (target_pose[MOVE_AXES] - current_pose[MOVE_AXES]).clamp(-0.03, 0.03)
                delta_action = target_pose - current_pose            # the lines below then produce target_pose
                delta_action[6] = target_pose[6]                     # gripper: absolute, as the code below expects

            target_pos_np = (current_pose[0:3] + delta_action[0:3]).detach().cpu().numpy()
            target_euler_np = (current_pose[3:6] + delta_action[3:6]).detach().cpu().numpy()
            target_gripper = delta_action[6].detach().cpu().item()

            # Convert back to rotation vector
            new_rotvec = Rotation.from_euler("xyz", target_euler_np).as_rotvec()

            ee_action = {
                "ee.x": float(target_pos_np[0]),
                "ee.y": float(target_pos_np[1]),
                "ee.z": float(target_pos_np[2]),
                "ee.wx": float(new_rotvec[0]),
                "ee.wy": float(new_rotvec[1]),
                "ee.wz": float(new_rotvec[2]),
                "ee.gripper_pos": float(target_gripper * 100.0),
            }

            # Solve and execute
            joint_action = ee_to_follower_joints((ee_action, joint_obs))
            if not DRY_RUN:
                follower.send_action(joint_action)
            time.sleep(0.5)

            time.sleep(1.0 / FPS)

        if not MOCK:
            time.sleep(1.0)
            save_model_view(follower.get_observation()["top"], f"runs/{run_name}_final.png")

    finally:
        if log_rows:
            os.makedirs("runs", exist_ok=True)
            path = f"runs/{run_name}.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "latent_l2", "x", "y", "z", "goal_x", "goal_y", "goal_z",
                            "dist_xz_cm", "plan_dx", "plan_dz", "action_score"])
                w.writerows(log_rows)
            print(f"Saved {len(log_rows)} steps to {path}")

        log_say("Stop the robot")
        follower.disconnect()
        leader.disconnect()
        listener.stop()

if __name__ == "__main__":
    main()