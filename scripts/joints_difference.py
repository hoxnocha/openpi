import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools
import os
import sys

def compute_next_step_deviations(policy, dataset, episode_index):
    """
    Compute deviation between Predicted Action at frame t and Actual State at frame t+1.
    """
    episode_data_index = dataset.episode_data_index
    from_idx = episode_data_index["from"][episode_index].item()
    to_idx = episode_data_index["to"][episode_index].item()
    
    # We can only evaluate up to the second to last frame, 
    # because the last frame has no "next frame" state to compare against.
    episode_length = to_idx - from_idx
    eval_length = episode_length - 1
    
    deviations = []
    
    print(f"Processing episode {episode_index} with {episode_length} frames (evaluating {eval_length} transitions)...")
    
    for frame_idx in range(eval_length):
        global_frame_index = from_idx + frame_idx
        
        # --- 1. Load Current Frame (t) for Inference ---
        item_t = dataset[global_frame_index]
        
        # Prepare Observation
        def prepare_image(img_tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            return img_np

        left_image = prepare_image(item_t["observation.image"])
        wrist_image = prepare_image(item_t["observation.wrist_image"])
        state_t = item_t["observation.state"].numpy()
        
        # Get prompt (handle potential missing task field)
        prompt = item_t.get("task", "Press the pump dispenser on the bottle all the way down.")
        if isinstance(prompt, torch.Tensor):
            prompt = "Press the pump dispenser on the bottle all the way down."
        
        request_data = {
            "observation/image": image_tools.resize_with_pad(left_image, 224, 224),
            "observation/wrist_image": image_tools.resize_with_pad(wrist_image, 224, 224),
            "observation/state": state_t,
            "prompt": prompt,
        }
        
        # Infer Policy to get Action(t)
        # Note: Depending on your chunking strategy, you might want pred_action_chunk[0] 
        # which usually corresponds to the target for the immediate next step.
        result = policy.infer(request_data)
        pred_action_chunk = result["actions"]
        pred_action_t = pred_action_chunk[0]  # Shape: (8,)
        
        # --- 2. Load Next Frame (t+1) for Ground Truth State ---
        item_next = dataset[global_frame_index + 1]
        state_next = item_next["observation.state"].numpy() # Shape: (8,)
        
        # --- 3. Compute Deviation ---
        # Deviation = Predicted Target for Next Step - Actual State Reached at Next Step
        diff = pred_action_t - state_next
        deviations.append(diff)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{eval_length} transitions...")
    
    return np.array(deviations) # Shape: (N-1, 8)

def plot_joint_deviations(deviations, episode_index):
    """
    Plot deviations for each joint separately.
    deviations shape: (num_frames, 8)
    """
    num_frames, num_dims = deviations.shape
    # Assuming 8 dims: 7 Joints + 1 Gripper
    # We will create a 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    joint_names = [f"Joint {i+1}" for i in range(7)] + ["Gripper"]
    
    for i in range(min(num_dims, 8)):
        ax = axes[i]
        # Plot the deviation curve
        ax.plot(deviations[:, i], label='Pred - NextState', color='tab:blue', linewidth=1.5)
        
        # Add a zero line for reference
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_title(f'{joint_names[i]} Deviation')
        ax.set_ylabel('Diff (rad / m)')
        ax.grid(True, alpha=0.3)
        
        # Calculate stats for title/legend
        mean_abs_err = np.mean(np.abs(deviations[:, i]))
        ax.legend([f"MAE: {mean_abs_err:.4f}"], loc='upper right')

        if i >= 6: # Bottom plots get x-label
            ax.set_xlabel('Frame Index')

    plt.suptitle(f'Prediction Deviation (Pred Action - Next State) - Episode {episode_index}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle
    
    save_path = f'episode_{episode_index}_deviation_curve.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index to analyze")
    args = parser.parse_args()
    
    # Configuration
    CHECKPOINT_DIR = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/12000")
    CONFIG_NAME = "pi0_pump_bottle_lora" 
    DATASET_REPO_ID = "ty/pi0_pump_bottle" 
    DATASET_ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/ty/pi0_pump_bottle")

    # Load Policy
    print(f"Loading policy: {CONFIG_NAME} from {CHECKPOINT_DIR}...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("Policy loaded successfully!")

    # Load Dataset
    print(f"Loading dataset: {DATASET_REPO_ID}...")
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT)
    
    episode_index = args.episode_index
    num_episodes = len(dataset.episode_data_index["from"])
    
    if episode_index >= num_episodes:
        print(f"Episode {episode_index} not found. Total episodes: {num_episodes}")
        return
    
    # Compute Deviations
    deviations = compute_next_step_deviations(policy, dataset, episode_index)
    
    # Plot
    plot_joint_deviations(deviations, episode_index)

if __name__ == "__main__":
    main()