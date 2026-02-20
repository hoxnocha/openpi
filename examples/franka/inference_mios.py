from math import dist
from openpi_client import image_tools
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.policies import droid_policy
from openpi.models import model as _model
from openpi import transforms as _transforms
from openpi.training import weight_loaders

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time

import pandas as pd
from PIL import Image
import tqdm
import tyro
import collections
from openpi.training import config as _config
from openpi.policies import policy_config
import sys
import contextlib
import dataclasses
import datetime
import faulthandler
import pyrealsense2 as rs
import numpy as np
import cv2
from typing import Tuple, Dict, Any
import sys
mios_path = "/home/ty/easymios"
sys.path.append(mios_path)
from mios import Mios  

faulthandler.enable()

CONTROL_FREQUENCY = 50
DT = 1.0 / CONTROL_FREQUENCY
MOVEJ_STRIDE = 25  # 

REPLAN_STEPS = 50  # deque 空了就 replan


class RealSenseStreamer:
    def __init__(self, serial_number: str, width: int, height: int, fps: int):
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    def start(self):
        self.pipeline.start(self.config)

    def flush(self):
        while True:
            frames = self.pipeline.poll_for_frames()
            if not frames:
                break

    def get_frame(self) -> np.ndarray | None:
        try:
            frames = self.pipeline.wait_for_frames(30)
            color = frames.get_color_frame()
            if color:
                return np.asanyarray(color.get_data())
            return None
        except Exception:
            return None

    def stop(self):
        with contextlib.suppress(Exception):
            self.pipeline.stop()


class RobotEnv:
    def __init__(
        self,
        left_camera_id: str,
        right_camera_id: str,
        wrist_camera_id: str,
        ip: str,
        port: int,
        database: str,
        
    ):
        self.camera_serials = {
            "left": left_camera_id,
            "right": right_camera_id,
            "wrist": wrist_camera_id,
        }
        self.streamers: Dict[str, RealSenseStreamer] = {}

        ctx = rs.context()
        connected = {dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()}

        for name, serial in self.camera_serials.items():
            if serial not in connected:
                print(f"[WARN] Camera serial not found: name={name} serial={serial}")
                continue
            self.streamers[name] = RealSenseStreamer(serial, width=640, height=480, fps=30)
            print(f"[INFO] Camera configured: {name} {serial}")

        self.robot = Mios(ip, port, database)
        self.robot.unlock()
        self.robot.home_gripper()
        self.robot.moveJ([0.0278,-0.147,-0.0007,-2.277,-0.0129,2.089,0.789]) # default pose for data collection
        
        
        self.last_gripper_target = 0 # 0 for open, 1 for grap
        time.sleep(2.0)  # wait for motion to complete
        current_q = self.robot.get_state()["result"]["q"]
        dist = np.linalg.norm(np.array(current_q) - np.array([0.027,-0.147,-0.0007,-2.276,-0.0128,2.089,0.789]))
        if dist > 0.1:
            print(f"WARNING: Robot did not reach target! Dist: {dist}")
    # raise RuntimeError("Robot initialization failed")
        else:
            print("Robot reached initial position.")
            print(f"[INFO] Mios initialized at {ip}:{port}, db={database}")

    def start_streams(self):
        for s in self.streamers.values():
            s.start()
        for s in self.streamers.values():
            s.flush()

    def stop_streams(self):
        for s in self.streamers.values():
            s.stop()

    def get_frames(self) -> Dict[str, np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        for name, s in self.streamers.items():
            f = s.get_frame()
            if f is None:
                continue
            frames[f"{name}_{s.serial_number}"] = f
        return frames

    def get_observation(self) -> Dict[str, Any]:
        raw_frames = self.get_frames()
        try:
            st = self.robot.get_state()
            st_data = st["result"]
            joint = np.asarray(st_data.get("q"), dtype=np.float32).reshape(-1)[:7]
            grip = float(np.asarray(st_data.get("gripper_width")).reshape(-1)[0])
            cart = np.asarray(st_data.get("O_T_EE"), dtype=np.float32).reshape(-1)[:16]
        except Exception as e:
            print("[ERROR] get_state failed:", e)
            joint = np.zeros(7, dtype=np.float32)
            grip = float(0.080511)
            cart = np.zeros(16, dtype=np.float32)

        robot_state = {
            "joint_positions": joint,
            "gripper_position": grip,
            "cartesian_position": cart,
        }
        return {"image": raw_frames, "robot_state": robot_state}

    def step_movej(self, action: np.ndarray):
        # action: (8,) absolute joint targets (0-6) + gripper (7)
        q_target = np.asarray(action[:7], dtype=np.float32).reshape(-1).tolist()
        raw_gripper = float(np.asarray(action[7]).reshape(-1)[0])

        # --- 1. Gripper Binarization Logic ---
        # Threshold: Assuming policy output > 0.5 means OPEN, < 0.5 means CLOSE
        # Adjust '0.5' if your policy uses a different range (e.g. 0.04 if output is meters)
        GRIPPER_OPEN_WIDTH = 0.08
        GRIPPER_CLOSED_WIDTH = 0.0
        THRESHOLD = 0.065

        # Determine target width based on threshold
        if raw_gripper > THRESHOLD:
            g_target = 0
        else:
            g_target = 1

        # --- 2. Robot Movement ---
        t0 = time.time()
        self.robot.moveJ(q_target)
        # print("moveJ time:", time.time() - t0) # Optional: comment out to reduce clutter

        # --- 3. Gripper Filtering (Deduping) ---
        # Grasp: last_gripper_target=0, g_target=1, emtpy2grasp
        if self.last_gripper_target == 0 and g_target == 1:
            print("[INFO] Gripper grasping")
            self.robot.grasp(0.0, 0.05, 50, 1, 1)
        # Release: last_gripper_target=1, g_target=0, grasp2emty
        elif self.last_gripper_target == 1 and g_target == 0:
            print("[INFO] Gripper releasing")
            self.robot.move_gripper(GRIPPER_OPEN_WIDTH)

        print(f"move time : {time.time() - t0:.3f}s ")

        self.last_gripper_target = g_target


@dataclasses.dataclass
class Args:
    left_camera_id: str = "233622072733"
    right_camera_id: str = "YOUR_RIGHTCAM_SERIAL"
    wrist_camera_id: str = "213322073390"

    ip: str = "10.157.174.42"
    port: int = 12000
    database: str = "diffL"

    external_camera: str = "left"  # "left" or "right"
    max_timesteps: int = 1200

    prompt: str = "Press the pump dispenser on the bottle all the way down."


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original)
        if interrupted:
            raise KeyboardInterrupt


def extract_observation(args: Args, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    image_obs = obs_dict["image"]

    left_img, right_img, wrist_img = None, None, None
    for key, img in image_obs.items():
        if args.left_camera_id in key and "left" in key:
            left_img = img
        elif args.right_camera_id in key and "right" in key:
            right_img = img
        elif args.wrist_camera_id in key and "wrist" in key:
            wrist_img = img

    if left_img is None:
        left_img = np.zeros((480, 640, 3), dtype=np.uint8)
    if right_img is None:
        right_img = np.zeros((480, 640, 3), dtype=np.uint8)
    if wrist_img is None:
        wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)

    rs_state = obs_dict["robot_state"]
    joint = np.asarray(rs_state["joint_positions"], dtype=np.float32).reshape(-1)[:7]
    grip = float(np.asarray(rs_state["gripper_position"]).reshape(-1)[0])
    state = np.concatenate([joint, np.asarray([grip], dtype=np.float32)], axis=0).astype(np.float32)
    assert state.shape == (8,), f"Bad state shape: {state.shape}"

    ext = left_img if args.external_camera == "left" else right_img

    return {"external_image": ext, "wrist_image": wrist_img, "state": state}


def main(args: Args):
    assert args.external_camera in ("left", "right")

    env = RobotEnv(
        left_camera_id=args.left_camera_id,
        right_camera_id=args.right_camera_id,
        wrist_camera_id=args.wrist_camera_id,
        ip=args.ip,
        port=args.port,
        database=args.database,
    )
    env.start_streams()

    print("[INFO] Loading policy locally...")
    pretrained_cfg = _config.get_config("pi0_pump_bottle_lora")
    checkpoint_dir = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/12000")
    policy = policy_config.create_trained_policy(pretrained_cfg, checkpoint_dir)

    history = []

    try:
        while True:
            prompt = input("Enter prompt: ")
            #prompt = args.prompt

            # warm-up exposure
            for _ in range(60):
                env.get_frames()

            action_plan = collections.deque() 
            
            print("[INFO] Running rollout (Ctrl-C to stop)")

            
            while True:
                
                    if not action_plan:

                        curr_obs = extract_observation(args, env.get_observation())
                        ext_img = curr_obs["external_image"]
                        wrist_img = curr_obs["wrist_image"]
                        state = curr_obs["state"]

                        t0_infer = time.time()
                        request_data = {
                            "observation/image": image_tools.resize_with_pad(ext_img, 224, 224),
                            "observation/wrist_image": image_tools.resize_with_pad(wrist_img, 224, 224),
                            "observation/state": np.asarray(state, dtype=np.float32),
                            "prompt": prompt,
                        }

                        # action chunk (50, 8) /(B， A)
                        action_chunk = policy.infer(request_data)["actions"]

                        action_chunk = np.asarray(action_chunk, dtype=np.float32)
                        
                        #action_chunk[-1] = 
                        downsampled_actions = action_chunk[MOVEJ_STRIDE - 1 :: MOVEJ_STRIDE]
                        action_plan.extend(downsampled_actions)
                        
                        print(f"[INFO] Infer done ({time.time()-t0_infer:.3f}s). "
                              f"Original: {action_chunk.shape}, Downsampled: {downsampled_actions.shape}")

                    if action_plan:
                        #step_start = time.time()
                        action = action_plan.popleft()
                        env.step_movej(action)
                    
                        #step_duration = DT * MOVEJ_STRIDE
                        
                        #elapsed = time.time() - step_start
                        #if elapsed < step_duration:
                        #    time.sleep(step_duration - elapsed)
                    

    except KeyboardInterrupt:
        pass
        
    finally:
        env.stop_streams()
        print("[INFO] Streams stopped.")


if __name__ == "__main__":
    main(tyro.cli(Args))
