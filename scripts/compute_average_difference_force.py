import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

# ================= 配置区域 =================
# 1. 接触检测参数 (完全复刻 all_segment.py)
FS = 50                 # 采样率
WIN_S = 2.0             # 静默窗口长度 (秒)
STEP_S = 0.2            # 窗口滑动步长 (秒)
K_SIGMA = 10         # 阈值倍数 (根据你的数据调整)
USE_MAD = True          # 使用 MAD 算法抗噪

# 2. 力数据通道设置
# Observation State 结构: [Joints(8), Forces(6)]
# 原 m 文件 Col 12 (1-based) -> Index 11
# 去掉 Col 0 (Time) 后 -> Index 10
# 也就是倒数第 4 维 (Fz)
FORCE_INDEX_IN_STATE = 10 

# 3. 绘图设置
HIGHLIGHT_COLOR = 'green'
HIGHLIGHT_ALPHA = 0.15  # 透明度
# ===========================================

def find_quietest_window_1d(x: np.ndarray, win_len: int, step: int, use_mad: bool = True):
    """
    [核心算法] 复刻自 all_segment.py: 寻找最平稳的基准线
    """
    N = x.shape[0]
    if win_len >= N:
        bias = float(np.median(x))
        noise_std = float(np.std(x))
        return bias, 0, N, noise_std

    best_i0, best_score = 0, np.inf
    
    for i0 in range(0, N - win_len + 1, step):
        seg = x[i0: i0 + win_len]
        if use_mad:
            med = np.median(seg)
            mad = np.median(np.abs(seg - med))
            score = float(mad)
        else:
            score = float(np.std(seg))
        
        if score < best_score:
            best_score = score
            best_i0 = i0

    i1 = best_i0 + win_len
    baseline = x[best_i0:i1]
    bias = float(np.median(baseline))
    noise_std = float(np.std(baseline))
    return bias, best_i0, i1, noise_std

def detect_contact_segment(force_signal, fs=50):
    """
    对单维力信号进行分段
    返回: start_idx, end_idx, threshold, bias
    """
    win_len = int(WIN_S * fs)
    step = max(1, int(STEP_S * fs))
    
    # 1. 寻找基准线
    bias, i0, i1, noise_std = find_quietest_window_1d(force_signal, win_len, step, USE_MAD)
    
    # 2. 计算阈值
    x_db = force_signal - bias
    threshold = K_SIGMA * noise_std
    
    # 3. 判定 (原脚本逻辑: x_db > threshold)
    # 如果你也需要检测负向力(比如拉力)，建议改为 np.abs(x_db) > threshold
    # 这里保持和你原脚本一致的单向逻辑，或者你可以根据实际情况修改
    mask = x_db > threshold 
    # mask = np.abs(x_db) > threshold # 双向检测建议用这个
    
    idx = np.where(mask)[0]
    
    if idx.size > 0:
        return idx[0], idx[-1], threshold, bias
    else:
        return None, None, threshold, bias

def evaluate_episode(policy, dataset, episode_index):
    """
    运行推理，计算误差，并提取力信号
    """
    episode_data_index = dataset.episode_data_index
    from_idx = episode_data_index["from"][episode_index].item()
    to_idx = episode_data_index["to"][episode_index].item()
    total_len = to_idx - from_idx
    
    # --- 1. 提取力信号 (用于分段) ---
    force_trace = []
    for i in range(total_len):
        # 假设 transform 不改变数值
        state = dataset[from_idx + i]["observation.state"].numpy()
        force_trace.append(state[FORCE_INDEX_IN_STATE])
    force_trace = np.array(force_trace)
    
    # --- 2. 逐帧推理 ---
    joint_deviations = []
    eval_len = total_len - 1 # 最后一帧没有 Next State
    
    print(f"Processing Episode {episode_index} ({eval_len} frames)...")
    
    for i in range(eval_len):
        global_idx = from_idx + i
        item = dataset[global_idx]
        
        # 图像预处理
        def prepare_img(img):
            np_img = img.permute(1, 2, 0).numpy()
            if np_img.max() <= 1.0: np_img = (np_img * 255).astype(np.uint8)
            return np_img

        full_state = item["observation.state"].numpy()
        model_input_state = full_state[:8]  
        req = {
            "observation/image": image_tools.resize_with_pad(prepare_img(item["observation.image"]), 224, 224),
            "observation/wrist_image": image_tools.resize_with_pad(prepare_img(item["observation.wrist_image"]), 224, 224),
            "observation/state": model_input_state, 
            "prompt": "Press the pump dispenser on the bottle all the way down.",
        }
        
        res = policy.infer(req)
        
        # 预测动作 (通常是纯关节)
        pred_action = res["actions"][0]
        
        # 真实动作 (下一帧状态，包含力，需要切分)
        next_item = dataset[global_idx + 1]
        next_full_state = next_item["observation.state"].numpy()
        
        # 自动推断关节维度: Pred Action 的长度
        joint_dim = len(pred_action)
        
        # 获取 GT 的关节部分
        gt_joint_pos = next_full_state[:joint_dim]
        
        diff = pred_action - gt_joint_pos
        joint_deviations.append(diff)
        
    return np.array(joint_deviations), force_trace, joint_dim

def plot_with_highlight(deviations, force_trace, segment_info, episode_idx):
    """
    绘制带有 Contact Rich 高亮区域的误差图
    """
    start_idx, end_idx, threshold, bias = segment_info
    num_frames, num_joints = deviations.shape
    
    # 布局: 顶部画力信号，下面画关节误差
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2) # 5行2列
    
    # --- 1. 力信号与分段验证 (Top Row) ---
    ax_force = fig.add_subplot(gs[0, :]) 
    # 对齐长度 (force_trace 比 deviations 多 1 帧，切掉最后一帧以便对齐 x 轴)
    plot_len = len(deviations)
    ax_force.plot(force_trace[:plot_len], color='black', linewidth=1.5, label=f'Force Trace (Idx {FORCE_INDEX_IN_STATE})')
    
    # 画阈值线
    ax_force.axhline(bias, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax_force.axhline(bias + threshold, color='red', linestyle=':', label='Threshold')
    
    # 标记检测到的区域
    if start_idx is not None:
        ax_force.axvspan(start_idx, end_idx, color=HIGHLIGHT_COLOR, alpha=HIGHLIGHT_ALPHA, label='Contact Rich Segment')
        
    ax_force.set_title(f"Episode {episode_idx}: Force-based Contact Segmentation", fontsize=14)
    ax_force.legend(loc='upper right')
    ax_force.grid(True, alpha=0.3)
    ax_force.set_xlim(0, plot_len)

    # --- 2. 关节误差曲线 (Bottom Rows) ---
    # 假设 8 个关节 (7 + Gripper)
    joint_names = [f"Joint {i+1}" for i in range(num_joints-1)] + ["Gripper"]
    
    for i in range(min(num_joints, 8)):
        row = (i // 2) + 1
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # 误差曲线
        ax.plot(deviations[:, i], label='Prediction Error', color='tab:blue')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        
        # [关键] 添加 Contact Rich 辅助高亮
        if start_idx is not None:
            # 1. 绿色背景
            ax.axvspan(start_idx, end_idx, color=HIGHLIGHT_COLOR, alpha=HIGHLIGHT_ALPHA)
            # 2. 垂直边界线
            ax.axvline(start_idx, color='green', linestyle='--', linewidth=1)
            ax.axvline(end_idx, color='green', linestyle='--', linewidth=1)
            
            # 3. 计算并标注 Contact 区域内的平均误差 (Contact MAE)
            # 限制索引不越界
            s = max(0, start_idx)
            e = min(len(deviations), end_idx)
            if e > s:
                seg_err = deviations[s:e, i]
                contact_mae = np.mean(np.abs(seg_err))
                # 在图内标注
                ax.text(0.02, 0.92, f"Contact MAE: {contact_mae:.4f}", transform=ax.transAxes, 
                        color='darkgreen', fontweight='bold', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_title(joint_names[i])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, plot_len)
        
        if row == 4: ax.set_xlabel("Frame Index")

    plt.suptitle(f"Analysis of Episode {episode_idx}: Prediction Error & Contact Segmentation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_name = f"eval_ep{episode_idx}_contact_highlight.png"
    plt.savefig(save_name, dpi=300)
    print(f"\n[INFO] Plot saved to: {save_name}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_index", type=int, default=0)
    args = parser.parse_args()
    
    # 路径配置 (请确保与 convert 脚本一致)
    CHECKPOINT_DIR = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/12000")
    CONFIG_NAME = "pi0_pump_bottle_lora" 
    DATASET_REPO_ID = "ty/pi0_pump_bottle_force"  # 你新制作的数据集
    DATASET_ROOT = os.path.expanduser(f"~/.cache/huggingface/lerobot/{DATASET_REPO_ID}")

    print(f"[INFO] Loading Policy...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    print(f"[INFO] Loading Dataset...")
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT)
    
    # 1. 运行评估
    deviations, force_trace, j_dim = evaluate_episode(policy, dataset, args.episode_index)
    
    # 2. 运行分段算法
    print("[INFO] Detecting contact segment...")
    start_idx, end_idx, thresh, bias = detect_contact_segment(force_trace, fs=FS)
    
    if start_idx is not None:
        print(f"  -> Contact found: Frame {start_idx} to {end_idx} (Threshold: {thresh:.4f})")
    else:
        print(f"  -> No contact detected (Max val: {np.max(force_trace):.4f}). Plotting without highlight.")

    # 3. 绘图
    plot_with_highlight(deviations, force_trace, (start_idx, end_idx, thresh, bias), args.episode_index)

if __name__ == "__main__":
    main()