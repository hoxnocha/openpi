import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

def compute_episode_frame_mses(policy, dataset, episode_index):
    """Compute MSE for each frame in an episode."""
    episode_data_index = dataset.episode_data_index
    from_idx = episode_data_index["from"][episode_index].item()
    to_idx = episode_data_index["to"][episode_index].item()
    
    episode_length = to_idx - from_idx
    mses = []
    
    print(f"Processing episode {episode_index} with {episode_length} frames...")
    
    for frame_idx in range(episode_length):
        global_frame_index = from_idx + frame_idx
        
        # Load frame data
        item = dataset[global_frame_index]
        
        # Get ground truth action
        gt_action = item["action"] if "action" in item else item["actions"]
        gt_action = gt_action.numpy()
        
        # Prepare observation data
        def prepare_image(img_tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            return img_np

        left_image = prepare_image(item["observation.image"])
        wrist_image = prepare_image(item["observation.wrist_image"])
        state = item["observation.state"].numpy()
        
        # Get prompt
        prompt = item.get("task", "Press the pump dispenser on the bottle all the way down.")
        if isinstance(prompt, torch.Tensor):
            prompt = "Press the pump dispenser on the bottle all the way down."
        
        # Prepare request data
        request_data = {
            "observation/image": image_tools.resize_with_pad(left_image, 224, 224),
            "observation/wrist_image": image_tools.resize_with_pad(wrist_image, 224, 224),
            "observation/state": state,
            "prompt": prompt,
        }
        
        # Run inference
        result = policy.infer(request_data)
        pred_action_chunk = result["actions"]
        pred_action = pred_action_chunk[0]  # Take first action
        
        # Compute MSE
        diff = pred_action - gt_action
        mse = np.mean(diff**2)
        mses.append(mse)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{episode_length} frames...")
    
    return np.array(mses)

def plot_episode_mse_curve(mses, episode_index):
    """Plot MSE curve for frames in an episode."""
    plt.figure(figsize=(12, 6))
    
    plt.plot(mses, linewidth=2, marker='o', markersize=3, alpha=0.7)
    
    plt.xlabel('Frame Index')
    plt.ylabel('MSE')
    plt.title(f'Action Prediction MSE Across Frames in Episode {episode_index}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add statistics
    mean_mse = np.mean(mses)
    std_mse = np.std(mses)
    plt.axhline(y=mean_mse, color='r', linestyle='--', alpha=0.7, label=f'Mean MSE: {mean_mse:.6f}')
    plt.legend()
    
    plt.savefig(f'episode_{episode_index}_mse_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(".6f")
    print(".6f")
    print(".6f")
    print(".6f")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index to analyze")
    args = parser.parse_args()
    
    CHECKPOINT_DIR = "/home/ty/.cache/openpi/openpi-assets/checkpoints/12000"
    CONFIG_NAME = "pi0_pump_bottle_lora" 
    
    DATASET_REPO_ID = "ty/pi0_pump_bottle" 
    
    print(f"Loading policy: {CONFIG_NAME} from {CHECKPOINT_DIR}...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("Policy loaded successfully!")

    print(f"Loading dataset: {DATASET_REPO_ID}...")
    dataset = LeRobotDataset(DATASET_REPO_ID, root="/home/ty/.cache/huggingface/lerobot/ty/pi0_pump_bottle")
    
    episode_index = args.episode_index
    num_episodes = len(dataset.episode_data_index["from"])
    
    if episode_index >= num_episodes:
        print(f"Episode {episode_index} not found. Total episodes: {num_episodes}")
        return
    
    # Compute MSE for each frame in the episode
    mses = compute_episode_frame_mses(policy, dataset, episode_index)
    
    # Plot the MSE curve
    plot_episode_mse_curve(mses, episode_index)


if __name__ == "__main__":
    main()
    
    CHECKPOINT_DIR = "/home/ty/.cache/openpi/openpi-assets/checkpoints/12000"
    CONFIG_NAME = "pi0_pump_bottle_lora" 
    
    # 3. 数据集 ID (必须已存在于 ~/.cache/huggingface/lerobot/...)
    DATASET_REPO_ID = "ty/pi0_pump_bottle" 
    
    # 4. 要测试的集数和帧数
    EPISODE_INDEX = 0  # 第几个视频/Episode
    FRAME_INDEX = 50   # 第几帧
    # ===========================================

    print(f"Loading policy: {CONFIG_NAME} from {CHECKPOINT_DIR}...")
    # 加载配置和权重
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("Policy loaded successfully!")

    print(f"Loading dataset: {DATASET_REPO_ID}...")
    # 加载 LeRobot 数据集 (确保你已经用 convert 脚本生成好了)
    dataset = LeRobotDataset(DATASET_REPO_ID, root="/home/ty/.cache/huggingface/lerobot/ty/pi0_pump_bottle")
    
    # 获取该 Episode 的起始帧和结束帧索引
    episode_data_index = dataset.episode_data_index
    from_idx = episode_data_index["from"][EPISODE_INDEX].item()
    to_idx = episode_data_index["to"][EPISODE_INDEX].item()
    
    # 计算全局帧索引
    global_frame_index = from_idx + FRAME_INDEX
    if global_frame_index >= to_idx:
        raise ValueError(f"Frame {FRAME_INDEX} exceeds length of episode {EPISODE_INDEX}")

    # 读取一帧数据
    item = dataset[global_frame_index]
    
    # --- 1. 提取 Ground Truth (GT) ---
    # 根据你的 info.json，动作存储在 "action" 或 "actions" 字段
    # LeRobotDataset 返回的是 Torch Tensor
    gt_action = item["action"] if "action" in item else item["actions"]
    gt_action = gt_action.numpy()
    
    # --- 2. 提取并预处理观察数据 (Observation) ---
    # LeRobot 返回的图像是 (C, H, W) 且归一化到了 [0, 1] 或 [0, 255]
    # OpenPi 通常期望 uint8 的 numpy array (H, W, C) 用于 resize
    
    def prepare_image(img_tensor):
        # (C, H, W) -> (H, W, C)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # 如果是浮点数 [0,1]，转回 [0,255]
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        return img_np

    left_image = prepare_image(item["observation.image"])       # 对应你 config 里的 observation/image
    wrist_image = prepare_image(item["observation.wrist_image"]) # 对应 observation/wrist_image
    state = item["observation.state"].numpy()
    
    # 获取指令 (Prompt)
    # 如果数据集里有 "task" 或 "instruction" 字段
    prompt = item["task"] if "task" in item else "Press the pump dispenser on the bottle all the way down." # 这里可以使用默认值或从数据集读取
    if isinstance(prompt, torch.Tensor):
        # 解码字符串 (如果被编码了)
        prompt = "Press the pump dispenser on the bottle all the way down." # 暂时硬编码，或者你需要从 info.json 映射

    print(f"\n--- Frame Info ---")
    print(f"Episode: {EPISODE_INDEX}, Frame: {FRAME_INDEX}")
    print(f"Prompt: {prompt}")
    print(f"State (Joints): {state[:7]}")

    
    request_data = {
        "observation/image": image_tools.resize_with_pad(left_image, 224, 224),
        "observation/wrist_image": image_tools.resize_with_pad(wrist_image, 224, 224),
        "observation/state": state,
        "prompt": prompt,
    }

    # --- 4. 运行推理 ---
    print("\nRunning inference...")
    result = policy.infer(request_data)
    pred_action_chunk = result["actions"] # 这是一个 Chunk，例如 (50, 8)
    
    # 取出预测的第一个动作 (对应当前帧)
    pred_action = pred_action_chunk[0]

    # --- 5. 计算差值并展示 ---
    print("\n" + "="*40)
    print("       ACTION COMPARISON (First Step)")
    print("="*40)
    
    # 打印前 7 维 (关节角度) 和 第 8 维 (夹爪)
    diff = pred_action - gt_action
    mse = np.mean(diff**2)
    
    print(f"{'Dim':<5} | {'GT Action':<12} | {'Pred Action':<12} | {'Diff':<12}")
    print("-" * 50)
    for i in range(len(gt_action)):
        print(f"{i:<5} | {gt_action[i]:<12.4f} | {pred_action[i]:<12.4f} | {diff[i]:<12.4f}")
    
    print("-" * 50)
    print(f"Total MSE: {mse:.6f}")
    print("="*40)

    #

def plot_comparison(gt, pred):
    plt.figure(figsize=(10, 5))
    indices = np.arange(len(gt))
    width = 0.35
    
    plt.bar(indices - width/2, gt, width, label='Ground Truth')
    plt.bar(indices + width/2, pred, width, label='Prediction')
    
    plt.xlabel('Action Dimension')
    plt.ylabel('Value')
    plt.title('Action Comparison: GT vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()