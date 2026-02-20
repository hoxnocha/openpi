"""
generate_lerobot_dataset_jax.py
适配 JAX/OpenPi 模型：全程使用 Numpy 处理，移除 torch.no_grad 和 .cpu()
（已整合 contact mask 同步对齐）
"""
import os
import numpy as np
from tqdm import tqdm
import roboticstoolbox as rtb
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from datasets import Dataset, Features, Sequence, Value
from scipy.spatial.transform import Rotation as R

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

# ================= 配置 =================
SOURCE_REPO_ID = "ty/pi0_insert_plug_force"
TARGET_REPO_ID = "ty/residual_policy_insert_plug"
CHECKPOINT_DIR = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi0_insert_plug_lora")
CONFIG_NAME = "pi0_insert_plug_lora"

MODEL_INPUT_DIM = 8
FORCE_INDEX = 8         # 修正后的索引
JOINT_DIM = 7
# =======================================

def compute_tcp_residual(T_curr, T_pred, T_gt):
    # 纯 Numpy 计算，保持不变
    R_curr, t_curr = T_curr[:3, :3], T_curr[:3, 3]
    R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]
    R_gt,   t_gt   = T_gt[:3, :3],   T_gt[:3, 3]
    
    diff_world_pos = t_gt - t_pred
    diff_tcp_pos = R_curr.T @ diff_world_pos
    
    R_diff_tcp_mat = R_curr.T @ (R_gt @ R_pred.T) @ R_curr
    residual_rot_vec = R.from_matrix(R_diff_tcp_mat).as_rotvec()
    return np.concatenate([diff_tcp_pos, residual_rot_vec])

def main():
    print("[INFO] Loading Contact Mask...")
    # 加载你的 32092 长度的 mask 数组
    contact_mask_array = np.load("contact_mask_final.npy")

    print("[INFO] Loading JAX Policy...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    print(f"[INFO] Loading Source Dataset: {SOURCE_REPO_ID}")
    source_dataset = LeRobotDataset(SOURCE_REPO_ID)
    
    print("[INFO] Initializing Panda Model...")
    robot = rtb.models.Panda()

    # 数据容器 (新增 contact_mask)
    data_buffer = {
        "observation.state": [],
        "action": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "next.done": [],
        "contact_mask": [] 
    }

    print(f"[INFO] Processing {source_dataset.num_episodes} episodes...")
    
    for ep_idx in tqdm(range(source_dataset.num_episodes)):
        episode_range = source_dataset.episode_data_index
        from_idx = episode_range["from"][ep_idx].item()
        to_idx = episode_range["to"][ep_idx].item()
        
        for i in range(to_idx - from_idx - 1):
            curr_idx = from_idx + i
            next_idx = curr_idx + 1
            
            item = source_dataset[curr_idx]
            next_item = source_dataset[next_idx]
            
            # --- 1. 处理 State (NumPy) ---
            full_state = item["observation.state"].numpy() 
            
            # 长度检查 (如果不合规，直接 continue，完美规避错位)
            if len(full_state) < 14:
                continue

            joint_state = full_state[:MODEL_INPUT_DIM] 
            force_data = full_state[FORCE_INDEX : FORCE_INDEX+6]
            new_input_state = np.concatenate([joint_state, force_data])
            
            if new_input_state.shape[0] != 14:
                continue

            # --- 2. Pi0 推理 (JAX) ---
            def prepare_img(img):
                np_img = img.permute(1, 2, 0).numpy()
                if np_img.max() <= 1.0: np_img = (np_img * 255).astype(np.uint8)
                return image_tools.resize_with_pad(np_img, 224, 224)

            req = {
                "observation/image": prepare_img(item["observation.image"]),
                "observation/wrist_image": prepare_img(item["observation.wrist_image"]),
                "observation/state": joint_state, 
                "prompt": "Insert the plug into the socket.",
            }
            
            res = policy.infer(req)
                
            # --- 3. 计算 Residual ---
            curr_joints = joint_state[:JOINT_DIM]
            pred_joints = res["actions"][0][:JOINT_DIM]
            gt_joints = next_item["observation.state"][:JOINT_DIM].numpy()
            
            T_curr = robot.fkine(curr_joints).A
            T_pred = robot.fkine(pred_joints).A
            T_gt   = robot.fkine(gt_joints).A
            
            residual = compute_tcp_residual(T_curr, T_pred, T_gt)
            
            # --- 4. 存入 Buffer ---
            data_buffer["observation.state"].append(new_input_state.astype(np.float32))
            data_buffer["action"].append(residual.astype(np.float32))
            data_buffer["episode_index"].append(ep_idx)
            data_buffer["frame_index"].append(i)
            data_buffer["timestamp"].append(item["timestamp"].item())
            data_buffer["next.done"].append(False if i < (to_idx - from_idx - 2) else True)
            
            # 新增：直接使用绝对索引 curr_idx 取出 mask 值存入
            data_buffer["contact_mask"].append(float(contact_mask_array[curr_idx]))

    print("[INFO] Constructing HF Dataset...")
    features = Features({
        "observation.state": Sequence(length=14, feature=Value(dtype="float32")),
        "action": Sequence(length=6, feature=Value(dtype="float32")),
        "episode_index": Value(dtype="int64"),
        "frame_index": Value(dtype="int64"),
        "timestamp": Value(dtype="float32"),
        "next.done": Value(dtype="bool"),
        "contact_mask": Value(dtype="float32") # 新增 Feature
    })
    
    hf_dataset = Dataset.from_dict(data_buffer, features=features)
    hf_dataset.save_to_disk(f"data/{TARGET_REPO_ID}")
    print(f"[SUCCESS] Saved to data/{TARGET_REPO_ID}")

if __name__ == "__main__":
    main()