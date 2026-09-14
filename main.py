import torch
from scipy.spatial.transform import Rotation
import time

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
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import log_say

from evals.simu_env_planning.planning.planning.planner import CEMPlanner
from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistL1MPCObjective

# What to do:
# Feed current and goal image into V-JEPA-2 AC encoder. This gets us 2 latents. Then do CEM search using the predictor
# and pick first action, like MPC.
# Then JEPA gives a 7D pose in the form of (dx, dy, dz, d_euler_x, d_euler_y, d_euler_z, d_gripper).
# Convert JEPAs pose, which is a 7D delta, to joint angles.

FPS = 30
MOCK = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    # Create the robot and follower configurations
    camera_config = {"top": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=FPS)}
    follower_config = SO100FollowerConfig(
        port="/dev/ttyACM0",
        id="follower_1",
        cameras=camera_config,
        use_degrees=True,
    )

    # Initialize the follower robot
    follower = SO100Follower(follower_config)

    # Define kinematics
    follower_kinematics_solver = RobotKinematics(
        urdf_path="./SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
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
                end_effector_bounds={"min": [-0.3, -0.3, 0.0], "max": [0.3, 0.3, 0.4]},
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

    # Connect the robot
    follower.connect()

    # Initialize keyboard listener
    listener, events = init_keyboard_listener()

    try:
        if not follower.is_connected:
            raise ValueError("Follower is not connected!")

        print("Connected, starting loop...")

        print("Define the goal state, then press 'n' to confirm.")
        follower.bus.disable_torque()
        goal_visual = None
        while goal_visual is None:
            joint_obs = follower.get_observation()
            if events["exit_early"]:
                if "top" in joint_obs:
                    goal_visual = (
                        torch.from_numpy(joint_obs["top"])
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )
                    print("Goal state captured.")
                else:
                    print("Camera not ready yet, try again in a moment.")
                events["exit_early"] = False  # always reset, whether or not we captured
            time.sleep(1.0 / FPS)

        print("Now move the robot back to the starting position, then press 'n' when ready.")
        while not events["exit_early"]:
            time.sleep(1.0 / FPS)
        events["exit_early"] = False  # reset so it doesn't leak into the main loop

        follower.bus.enable_torque()

        ###########################################
        #### Initializing JEPA and CEM planner ####
        ###########################################
        if not MOCK:
            model, preprocessor = torch.hub.load("facebookresearch/jepa-wms", "vjepa2_ac_droid")
            model.to(DEVICE).eval()

            goal_obs = {"visual": goal_visual, "proprio": torch.zeros(1, 1, 7)}
            goal_enc = model.encode(goal_visual).detach()

            objective = ReprTargetDistL1MPCObjective(
                cfg={},
                target_enc=goal_enc,
                sum_all_diffs=False,
                alpha=0,
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

        while True:

            if events["stop_recording"]:
                print("Stop requested, exiting control loop.")
                break

            joint_obs = follower.get_observation()

            # current_image is shape [H, W, 3]
            current_image = torch.from_numpy(joint_obs["top"]).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(DEVICE) # (B, T, C, H, W)
            ee_obs = follower_joints_to_ee(joint_obs)   # FK --> 7D

            # FK gives: ee.x, ee.y, ee.z (position), ee.wx, ee.wy, ee.wz (rotvec), ee.gripper_pos (0-100)
            # We convert this to euler angles as JEPA takes that as input to the WM
        
            # rotation vector -> Euler angles
            rotvec = [ee_obs["ee.wx"], ee_obs["ee.wy"], ee_obs["ee.wz"]]
            euler = Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=False)      # -> [roll, pitch, yaw] in radians

            current_pose = torch.tensor([
                ee_obs["ee.x"], ee_obs["ee.y"], ee_obs["ee.z"],
                *euler,
                ee_obs["ee.gripper_pos"] / 100.0, # 100 because JEPA takes in 0-1
            ], device=DEVICE)

            if MOCK == True:
                # Just to verify. Zero delta: keeps position/orientation identical, fixes gripper at 50%
                delta_action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=DEVICE)
            else:
                # Send into MPC planner
                obs = {"visual": current_image, "proprio": current_pose.reshape(1, 1, 7)}
                z = model.encode(current_image)

                print("We are planning!")
                result = planner.plan(z, steps_left=None)
                delta_action = preprocessor.denormalize_actions(result.actions[0].unsqueeze(0).cpu()).squeeze(0).to(DEVICE)

                # --- Heavy clamp for initial testing ---
                MAX_POS_DELTA = 0.005   # meters per axis
                MAX_ROT_DELTA = 0.05    # radians per axis (rotation vector component)
                MAX_GRIPPER_DELTA = 0.25  # 0-1 scale, since gripper_pos is later *100

                delta_action = torch.cat([
                    torch.clamp(delta_action[0:3], -MAX_POS_DELTA, MAX_POS_DELTA),   # x, y, z
                    torch.clamp(delta_action[3:6], -MAX_ROT_DELTA, MAX_ROT_DELTA),   # wx, wy, wz
                    torch.clamp(delta_action[6:7], -MAX_GRIPPER_DELTA, MAX_GRIPPER_DELTA),  # gripper
                ])

                print(f"raw-ish clamped delta: {delta_action.detach().cpu().numpy()}")
                print(f"unclamped: {result.actions[0].detach().cpu().numpy()} -> clamped: {delta_action.detach().cpu().numpy()}")

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

            print("Proposed_action: ",ee_action)

            # Solve and execute
            joint_action = ee_to_follower_joints((ee_action, joint_obs))
            follower.send_action(joint_action)

            time.sleep(1.0 / FPS)

    finally:
        log_say("Stop the robot")
        follower.disconnect()
        listener.stop()

if __name__ == "__main__":
    main()