import numpy as np
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib
from tqdm import tqdm

# 强制使用交互式后端 (支持窗口弹出)
matplotlib.use('TkAgg') 

# --- 1. 配置参数 ---
REPO_ID = "ty/pi0_insert_plug_force" 
FORCE_START_IDX = 8   # observation.state [8:14] 为六轴力/力矩

def print_usage():
    """在终端打印操作说明，避开 Q 键冲突"""
    print("\n" + "="*55)
    print("   LeRobot Residual Dataset 标注工具 ")
    print("="*55)
    print(" 【帧级移动】")
    print("  ← / → (方向键) :  逐帧 后退/前进")
    print("  A / D         :  逐帧 后退/前进 (同上)")
    print("\n 【Episode 切换】")
    print("  ↑ (方向键 Up)  :  载入 上一个 Episode")
    print("  ↓ (方向键 Down):  载入 下一个 Episode")
    print("\n 【高效段落标注】")
    print("  [ (左中括号)   :  标记 Contact 区域起点")
    print("  ] (右中括号)   :  标记 Contact 区域终点并自动填充")
    print("  Space (空格)   :  切换当前单帧的状态 (用于精修)")
    print("\n 【系统功能】")
    print("  S             :  保存当前全集标注进度到 .npy 文件")
    print("  Esc           :  关闭窗口并退出")
    print("="*55)
    print(" 💡 提示：若按键无效，请先点击一下弹出窗口中心以激活焦点。")
    print("="*55 + "\n")

# --- 2. 数据加载 ---
print(f"正在读取数据集: {REPO_ID} ...")
dataset = LeRobotDataset(REPO_ID)

# 自动兼容不同版本的 Episode 索引获取
if hasattr(dataset, 'episode_data_index'):
    ep_idx_table = dataset.episode_data_index
else:
    ep_idx_table = dataset.meta.episode_data_index

num_episodes = len(ep_idx_table['from'])

# 全局状态字典
state = {
    'ep_idx': 0,        
    'frame_in_ep': 0,   
    'abs_idx': 0,       
    'ep_frames': [],    
    'ep_forces': [],    
    'range_start': None
}

# 初始化全局掩码
contact_mask = np.zeros(len(dataset), dtype=int)

# --- 3. 核心加载逻辑 ---
def load_episode(ep_idx):
    print(f"\n[数据切换] 正在预加载 Episode {ep_idx}/{num_episodes-1} ...")
    
    from_idx = int(ep_idx_table['from'][ep_idx])
    to_idx = int(ep_idx_table['to'][ep_idx])
    
    state['ep_frames'] = []
    
    # 解码图像 (AV1 解码可能较慢，建议一次性载入内存)
    print(f"  -> 正在解码视频帧 (共 {to_idx - from_idx} 帧)...")
    for i in tqdm(range(from_idx, to_idx), desc="Decoding"):
        item = dataset[i]
        # (3, 480, 640) -> (480, 640, 3)
        img = item['observation.image'].numpy().transpose(1, 2, 0)
        state['ep_frames'].append(img)
    
    # 从底层 Arrow 表批量获取力矩数据
    obs_states = np.array(dataset.hf_dataset.select(range(from_idx, to_idx))['observation.state'])
    state['ep_forces'] = obs_states[:, FORCE_START_IDX : FORCE_START_IDX+6]
    
    # 重置 Episode 内索引
    state['from_idx'] = from_idx
    state['to_idx'] = to_idx
    state['frame_in_ep'] = 0
    state['abs_idx'] = from_idx
    print(f"  -> Episode {ep_idx} 载入完成。")

# --- 4. UI 绘制与交互控制 ---
fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(15, 7.5))
plt.subplots_adjust(bottom=0.15, top=0.9)

# 载入初始 Episode
load_episode(0)
img_obj = ax_img.imshow(state['ep_frames'][0])
ax_img.axis('off')

# 力矩曲线配置
colors = ['#FF4500', '#32CD32', '#1E90FF', '#FFD700', '#FF1493', '#00FFFF']
labels = ['Fx','Fy','Fz','Mx','My','Mz']
lines = [ax_plot.plot([], [], color=colors[i], label=labels[i], lw=1.8)[0] for i in range(6)]
ax_plot.set_ylim(-20, 20) 
ax_plot.grid(True, alpha=0.3, ls='--')
ax_plot.legend(loc='upper right', fontsize='9', ncol=2)
ax_plot.set_title("Real-time 6-DoF Forces (observation.state[8:14])")

def update_ui():
    f_idx = state['frame_in_ep']
    abs_idx = state['abs_idx']
    
    # 更新当前帧图像
    img_obj.set_data(state['ep_frames'][f_idx])
    
    # 更新波形滑动窗口 (最近 60 帧)
    start_v = max(0, f_idx - 60)
    history = state['ep_forces'][start_v : f_idx + 1]
    for i, line in enumerate(lines):
        line.set_data(range(len(history)), history[:, i])
    
    ax_plot.set_xlim(0, 60)
    
    # 更新 UI 标签与颜色
    is_contact = contact_mask[abs_idx]
    status_msg = "CONTACT-RICH (COMPUTE RESIDUAL)" if is_contact else "FREE-MOTION (NOMINAL)"
    status_clr = '#D32F2F' if is_contact else '#388E3C' # 深红/深绿
    
    title_text = f"Episode: {state['ep_idx']} | Frame: {f_idx} | Global Index: {abs_idx}\nStatus: {status_msg}"
    if state['range_start'] is not None:
        title_text += f" | [RECORDING START: {state['range_start']}]"
        
    fig.suptitle(title_text, color=status_clr, fontsize=14, fontweight='bold')
    fig.canvas.draw_idle()

def on_key(event):
    # 帧跳转
    if event.key in ['right', 'd']:
        if state['frame_in_ep'] < (len(state['ep_frames']) - 1):
            state['frame_in_ep'] += 1; state['abs_idx'] += 1
    elif event.key in ['left', 'a']:
        if state['frame_in_ep'] > 0:
            state['frame_in_ep'] -= 1; state['abs_idx'] -= 1
            
    # Episode 切换 (使用方向键避开 Q 键系统冲突)
    elif event.key == 'up':
        if state['ep_idx'] > 0:
            state['ep_idx'] -= 1; load_episode(state['ep_idx'])
    elif event.key == 'down':
        if state['ep_idx'] < num_episodes - 1:
            state['ep_idx'] += 1; load_episode(state['ep_idx'])
            
    # 状态标记
    elif event.key == ' ':
        contact_mask[state['abs_idx']] = 1 - contact_mask[state['abs_idx']]
    elif event.key == '[':
        state['range_start'] = state['abs_idx']
        print(f" [M] 标记段落起点: {state['abs_idx']}")
    elif event.key == ']':
        if state['range_start'] is not None:
            s, e = sorted([state['range_start'], state['abs_idx']])
            contact_mask[s:e+1] = 1
            print(f" [M] 段落填充完成: {s} -> {e}")
            state['range_start'] = None
        else:
            print(" [!] 错误：请先按 [ 设定起点")
            
    # 数据保存
    elif event.key == 's':
        out_name = "contact_mask_final.npy"
        np.save(out_name, contact_mask)
        print(f"\n ✔️ 标注已成功保存至本地: {out_name}")
    elif event.key == 'escape':
        plt.close()
        
    update_ui()

# 启动流程
print_usage()
fig.canvas.mpl_connect('key_press_event', on_key)
update_ui()
plt.show()