import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 引入 Robotics Toolbox
import roboticstoolbox as rtb
from spatialmath import SE3

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

# ================= 配置 =================
MODEL_INPUT_DIM = 8     
JOINT_DIM_FOR_FK = 7    
FORCE_INDEX = 10        
FS = 50                 
K_SIGMA = 10         
WIN_S = 2.0             
HIGHLIGHT_COLOR = 'green'
HIGHLIGHT_ALPHA = 0.15
# =======================================

def wrap_angle(angle_rad):
    """
    将角度误差归一化到 [-pi, pi] 之间。
    防止出现预测 179 度，实际 -179 度，直接相减变成 358 度的情况。
    """
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def find_quietest_window_1d(x, win_len=100, step=10):
    if len(x) < win_len: return np.mean(x), np.std(x)
    best_std = np.inf
    bias = 0
    for i in range(0, len(x)-win_len, step):
        seg = x[i:i+win_len]
        std = np.std(seg)
        if std < best_std:
            best_std = std
            bias = np.mean(seg)
    return bias, best_std

def evaluate_6d_cartesian(policy, dataset, episode_index, robot_model):
    episode_data_index = dataset.episode_data_index
    from_idx = episode_data_index["from"][episode_index].item()
    to_idx = episode_data_index["to"][episode_index].item()
    total_len = to_idx - from_idx
    eval_len = total_len - 1
    
    # 存储 6 维误差 (X, Y, Z, R, P, Y)
    cartesian_errors = [] 
    force_trace = []
    
    print(f"[INFO] Evaluating Episode {episode_index} in 6D Cartesian Space...")
    
    for i in range(eval_len):
        global_idx = from_idx + i
        item = dataset[global_idx]
        
        # 1. 记录力
        full_state = item["observation.state"].numpy()
        force_trace.append(full_state[FORCE_INDEX])
        
        # 2. 推理
        model_input = full_state[:MODEL_INPUT_DIM] 
        
        def prepare_img(img):
            np_img = img.permute(1, 2, 0).numpy()
            if np_img.max() <= 1.0: np_img = (np_img * 255).astype(np.uint8)
            return np_img
            
        req = {
            "observation/image": image_tools.resize_with_pad(prepare_img(item["observation.image"]), 224, 224),
            "observation/wrist_image": image_tools.resize_with_pad(prepare_img(item["observation.wrist_image"]), 224, 224),
            "observation/state": model_input,
            "prompt": "Press the pump dispenser on the bottle all the way down.",
        }
        
        with torch.no_grad():
            res = policy.infer(req)
        
        # 3. 获取关节
        pred_joints = res["actions"][0][:JOINT_DIM_FOR_FK]
        next_item = dataset[global_idx + 1]
        gt_joints = next_item["observation.state"].numpy()[:JOINT_DIM_FOR_FK]
        
        # 4. FK 计算
        T_pred = robot_model.fkine(pred_joints)
        T_gt = robot_model.fkine(gt_joints)
        
        # 5. 分解误差
        # --- 位置误差 (X, Y, Z) ---
        # 单位: 米 -> 毫米
        pos_err = (T_pred.t - T_gt.t) * 1000 
        
        # --- 姿态误差 (R, P, Y) ---
        # 单位: 弧度 -> 度
        # order='xyz' 表示静态轴旋转 (Roll-Pitch-Yaw)
        rpy_pred = T_pred.rpy(order='xyz') 
        rpy_gt = T_gt.rpy(order='xyz')
        
        # 计算差值并处理周期性 (Wrap to -pi...pi)
        diff_rad = wrap_angle(rpy_pred - rpy_gt)
        rpy_err = np.degrees(diff_rad)
        
        # 合并 6 维: [dx, dy, dz, dr, dp, dy]
        step_err = np.concatenate([pos_err, rpy_err])
        cartesian_errors.append(step_err)
        
    return np.array(cartesian_errors), np.array(force_trace)

def plot_6d_results(errors, force, ep_idx):
    """
    绘制 7 行子图：Force + XYZ + RPY
    """
    # 分段逻辑
    win_len = int(WIN_S * FS)
    bias, noise_std = find_quietest_window_1d(force, win_len=win_len)
    threshold = K_SIGMA * noise_std
    mask = np.abs(force - bias) > threshold
    indices = np.where(mask)[0]
    start, end = (indices[0], indices[-1]) if len(indices) > 0 else (None, None)
    
    # 绘图布局 (7 行 1 列)
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    
    labels = [
        "Pos X (mm)", "Pos Y (mm)", "Pos Z (mm)",
        "Rot R (deg)", "Rot P (deg)", "Rot Y (deg)"
    ]
    colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:orange']
    
    # --- 1. Force ---
    ax = axes[0]
    ax.plot(force, color='black', label='Force Z')
    ax.axhline(bias + threshold, color='red', linestyle=':')
    ax.axhline(bias - threshold, color='red', linestyle=':')
    if start:
        ax.axvspan(start, end, color=HIGHLIGHT_COLOR, alpha=HIGHLIGHT_ALPHA, label='Contact')
    ax.set_title(f"Episode {ep_idx}: Force Z (Contact Segment)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- 2. 6D Errors ---
    plot_len = len(errors)
    
    for i in range(6):
        ax = axes[i+1] # 从第2行开始
        data = errors[:, i]
        
        ax.plot(data, color=colors[i], label=labels[i])
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # 接触段高亮 & 统计
        if start:
            s, e = max(0, start), min(plot_len, end)
            if e > s:
                ax.axvspan(s, e, color=HIGHLIGHT_COLOR, alpha=HIGHLIGHT_ALPHA)
                
                # 计算接触段内的 Mean Abs Error
                mae = np.mean(np.abs(data[s:e]))
                ax.text(0.02, 0.85, f"Contact MAE: {mae:.2f}", transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8), fontweight='bold')
        
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        
        # 只在最后一张图显示 x 轴标签
        if i == 5:
            ax.set_xlabel("Frame Index")
            
    plt.suptitle(f"6D Cartesian Error Analysis (Pred Action vs GT Next State)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = f"eval_6d_cartesian_ep{ep_idx}.png"
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved plot to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_index", type=int, default=0)
    args = parser.parse_args()
    
    # 路径配置
    CHECKPOINT_DIR = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/12000")
    CONFIG_NAME = "pi0_pump_bottle_lora" 
    DATASET_REPO_ID = "ty/pi0_pump_bottle_force"
    DATASET_ROOT = os.path.expanduser(f"~/.cache/huggingface/lerobot/{DATASET_REPO_ID}")

    print("[INFO] Loading Policy...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    print("[INFO] Loading Dataset...")
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT)
    
    print("[INFO] Initializing Panda model...")
    robot = rtb.models.Panda()
    
    # 运行评估
    errors_6d, force = evaluate_6d_cartesian(policy, dataset, args.episode_index, robot)
    
    # 绘图
    plot_6d_results(errors_6d, force, args.episode_index)

if __name__ == "__main__":
    main()