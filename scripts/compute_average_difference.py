import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

def compute_episode_deviations(policy, dataset, episode_index):
    """
    计算单个 Episode 的 (Pred_Action - Next_State) 偏差。
    返回: np.array shape (frames, 8)
    """
    episode_data_index = dataset.episode_data_index
    from_idx = episode_data_index["from"][episode_index].item()
    to_idx = episode_data_index["to"][episode_index].item()
    
    # 只能评估到倒数第二帧
    episode_length = to_idx - from_idx
    eval_length = episode_length - 1
    
    deviations = []
    
    # print(f"  Episode {episode_index}: {eval_length} transitions...")
    
    # 预加载数据以加快速度 (可选，这里保持逐帧读取以节省内存)
    for frame_idx in range(eval_length):
        global_frame_index = from_idx + frame_idx
        
        # --- 1. Load Current Frame (t) ---
        item_t = dataset[global_frame_index]
        
        def prepare_image(img_tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            return img_np

        left_image = prepare_image(item_t["observation.image"])
        wrist_image = prepare_image(item_t["observation.wrist_image"])
        state_t = item_t["observation.state"].numpy()
        
        prompt = item_t.get("task", "Press the pump dispenser on the bottle all the way down.")
        if isinstance(prompt, torch.Tensor):
            prompt = "Press the pump dispenser on the bottle all the way down."
        
        request_data = {
            "observation/image": image_tools.resize_with_pad(left_image, 224, 224),
            "observation/wrist_image": image_tools.resize_with_pad(wrist_image, 224, 224),
            "observation/state": state_t,
            "prompt": prompt,
        }
        
        # --- 2. Inference ---
        # 禁用梯度计算加快速度
        with torch.no_grad():
            result = policy.infer(request_data)
        
        pred_action_chunk = result["actions"]
        pred_action_t = pred_action_chunk[0]  # 取当前帧对应的动作
        
        # --- 3. Load Next Frame (t+1) GT ---
        item_next = dataset[global_frame_index + 1]
        state_next = item_next["observation.state"].numpy()
        
        # --- 4. Compute Diff ---
        diff = pred_action_t - state_next
        deviations.append(diff)
    
    return np.array(deviations)

def plot_aggregated_deviations(mean_dev, std_dev, num_episodes, counts_per_frame):
    """
    绘制所有 Episode 的平均偏差曲线和标准差阴影。
    mean_dev: (max_frames, 8)
    std_dev: (max_frames, 8)
    """
    num_frames, num_dims = mean_dev.shape
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    joint_names = [f"Joint {i+1}" for i in range(7)] + ["Gripper"]
    
    for i in range(min(num_dims, 8)):
        ax = axes[i]
        
        # 获取有效数据的长度（去掉末尾全 NaN 的部分，让图紧凑点）
        valid_indices = np.where(~np.isnan(mean_dev[:, i]))[0]
        if len(valid_indices) == 0:
            continue
        last_idx = valid_indices[-1]
        
        x = np.arange(last_idx + 1)
        y_mean = mean_dev[:last_idx+1, i]
        y_std = std_dev[:last_idx+1, i]
        
        # 绘制平均线
        ax.plot(x, y_mean, label='Mean Deviation', color='tab:blue', linewidth=2)
        
        # 绘制标准差阴影 (Mean ± Std)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                        color='tab:blue', alpha=0.2, label='±1 Std Dev')
        
        # 零线
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_title(f'{joint_names[i]} (Avg over {num_episodes} eps)')
        ax.set_ylabel('Diff (rad / m)')
        ax.grid(True, alpha=0.3)
        
        # 计算该 Joint 在所有帧上的平均绝对误差 (Scalar MAE)
        total_mae = np.nanmean(np.abs(y_mean))
        ax.legend([f"Global MAE: {total_mae:.4f}"], loc='upper right')

        if i >= 6:
            ax.set_xlabel('Frame Index')

    plt.suptitle(f'Action Prediction Error Aggregated over {num_episodes} Episodes\n(Pred Action - Next State)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = f'aggregated_deviation_{num_episodes}eps.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[INFO] Plot saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to average")
    args = parser.parse_args()
    
    # --- Config ---
    CHECKPOINT_DIR = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/12000")
    CONFIG_NAME = "pi0_pump_bottle_lora" 
    DATASET_REPO_ID = "ty/pi0_pump_bottle" 
    DATASET_ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/ty/pi0_pump_bottle")

    # 1. Load Policy
    print(f"[INFO] Loading policy: {CONFIG_NAME}...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    # 2. Load Dataset
    print(f"[INFO] Loading dataset: {DATASET_REPO_ID}...")
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT)
    
    total_available_eps = len(dataset.episode_data_index["from"])
    num_episodes_to_process = min(args.num_episodes, total_available_eps)
    
    print(f"[INFO] Starting evaluation on {num_episodes_to_process} episodes...")
    
    all_deviations = []
    
    # 3. Loop Episodes
    for i in range(num_episodes_to_process):
        print(f"Processing Episode {i+1}/{num_episodes_to_process}...", end="\r")
        devs = compute_episode_deviations(policy, dataset, i) # (T, 8)
        all_deviations.append(devs)
    print("\n[INFO] Inference complete.")
    
    # 4. Data Alignment (Padding with NaN)
    max_len = max(len(d) for d in all_deviations)
    num_dims = 8
    
    # 创建容器 (Episodes, MaxFrames, Joints)
    aligned_data = np.full((num_episodes_to_process, max_len, num_dims), np.nan)
    
    for i, d in enumerate(all_deviations):
        length = len(d)
        aligned_data[i, :length, :] = d
        
    # 5. Compute Statistics (Ignore NaNs)
    # 计算每一帧有多少个有效的 episode (用于了解数据覆盖率，可选)
    counts_per_frame = np.sum(~np.isnan(aligned_data[:, :, 0]), axis=0)
    
    # 计算平均值和标准差
    mean_deviations = np.nanmean(aligned_data, axis=0) # (MaxFrames, 8)
    std_deviations = np.nanstd(aligned_data, axis=0)   # (MaxFrames, 8)
    
    # 6. Plot
    plot_aggregated_deviations(mean_deviations, std_deviations, num_episodes_to_process, counts_per_frame)

if __name__ == "__main__":
    main()