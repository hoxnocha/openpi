import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset



RAW_DATA_ROOT = "/home/ty/Downloads/dataset/insert_the_USB"  

REPO_ID = "ty/pi0_insert_USB"

ROBOT_STATE_FREQ = 1000  
TARGET_FPS = 30         

M_FILE_NAME = "DATA_follower.m"
EXT_VIDEO_NAME = "cam1.mp4"      
WRIST_VIDEO_NAME = "cam2.mp4"    
Prompt_DESCRIPTION = "Pick up the USB and plug it into the port."

def load_m_file_robust(filepath):
    """parse DATA_follower.m """
    with open(filepath, 'r') as f:
        content = f.read()
    
    start_index = content.find('[') + 1
    if start_index == 0:
        data_str = content
    else:
        data_str = content[start_index:]
        
    try:
        raw_data = np.fromstring(data_str, sep=' ').reshape(-1, 15)
    except ValueError:
        raw_values = np.fromstring(data_str, sep=' ')
        num_rows = len(raw_values) // 15
        raw_data = raw_values[:num_rows*15].reshape(-1, 15)
        
    state = raw_data[:, 1:9].astype(np.float32)
    return state

def get_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, (0, 0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return count, (height, width)

def main():
    root_path = Path(RAW_DATA_ROOT)
    if not root_path.exists():
        print(f"Cannot find {RAW_DATA_ROOT}")
        return
    all_m_files = list(root_path.rglob(M_FILE_NAME))
    
    valid_trials = []
    video_resolution = None 

    for m_file in all_m_files:
        trial_dir = m_file.parent
        wrist_path = trial_dir / WRIST_VIDEO_NAME  # cam2.mp4
        ext_path = trial_dir / EXT_VIDEO_NAME      # cam1.mp4
        
        if wrist_path.exists() and ext_path.exists():
            valid_trials.append({
                "path": trial_dir,
                "m_file": m_file,
                "wrist": wrist_path,
                "ext": ext_path
            })
            if video_resolution is None:
                _, res = get_video_info(wrist_path)
                if res != (0, 0):
                    video_resolution = res
        else:
            pass

   
    found_folders = set(f.parent.parent.name for f in all_m_files) 
    print(f"Find folder: {found_folders}")
    print(f"Find {len(valid_trials)} Trial")
    print(f"Find video resolution (H, W): {video_resolution}")

    output_dir = Path(os.path.expanduser(f"~/.cache/huggingface/lerobot/{REPO_ID}"))
    if output_dir.exists():
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=TARGET_FPS,
        robot_type="panda",
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["joint_position"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["joint_position"],
            },
            "observation.wrist_image": { # cam2
                "dtype": "video",
                "shape": (3, video_resolution[0], video_resolution[1]),
                "names": ["channel", "height", "width"],
            },
            "observation.image": { # cam1
                "dtype": "video",
                "shape": (3, video_resolution[0], video_resolution[1]),
                "names": ["channel", "height", "width"],
            },
        }
    )

    for trial in tqdm(valid_trials, desc="Converting Trials"):
        
        raw_states = load_m_file_robust(trial["m_file"])
        
        wrist_cap = cv2.VideoCapture(str(trial["wrist"])) # cam2
        ext_cap = cv2.VideoCapture(str(trial["ext"]))   # cam1
        
        n_frames_w = int(wrist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames_e = int(ext_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        num_frames = min(n_frames_w, n_frames_e)
        
        for i in range(num_frames):
            ret_w, frame_wrist = wrist_cap.read()
            ret_e, frame_ext = ext_cap.read()
            
            if not ret_w or not ret_e:
                break 
                
            frame_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
            frame_ext = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
            
        
            t_current = i / TARGET_FPS
            t_next = (i + 1) / TARGET_FPS
            
            
            idx_curr = int(t_current * ROBOT_STATE_FREQ)
            idx_next = int(t_next * ROBOT_STATE_FREQ)
            
            idx_curr = min(idx_curr, len(raw_states) - 1)
            idx_next = min(idx_next, len(raw_states) - 1)
            
            current_state = raw_states[idx_curr]
            action = raw_states[idx_next]

            dataset.add_frame({
                "observation.state": current_state,
                "action": action,
                "observation.wrist_image": frame_wrist,   # cam2
                "observation.image": frame_ext,  # cam1
                "task": Prompt_DESCRIPTION,
            })
            
        
        
        
        dataset.save_episode()

        wrist_cap.release()
        ext_cap.release()

    #dataset.consolidate()
    print(f"Dataset saved in: {output_dir}")

if __name__ == "__main__":
    main()