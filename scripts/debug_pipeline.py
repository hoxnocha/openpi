# debug_pipeline.py
from openpi.training import config as cfg

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import openpi.transforms as T


def _get_prompt_from_task_flag(data_cfg) -> bool:
    # 兼容两种命名：openpi/config.py 里是 promptfromtask；你贴的 dataloader 是 prompt_from_task
    return bool(getattr(data_cfg, "promptfromtask", False) or getattr(data_cfg, "prompt_from_task", False))


# 1) 拿到你的训练 config（用你最终要训练的那个 name）
train_cfg = cfg.get_config("pi0_finetune_pumpe_bottle")  # 按实际改 [file:91]

# 2) 创建 DataConfig（里面会把 repack/data/model transforms 组装好）
data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)  # [file:91]
print("data_cfg.repo_id =", data_cfg.repo_id)  # [file:91]
print("prompt_from_task flag =", _get_prompt_from_task_flag(data_cfg))  # [file:91]

# 3) 打开 LeRobot 数据集
ds = LeRobotDataset(repo_id=data_cfg.repo_id)  # [file:91]
sample = ds[0]  # 或 ds.get_frame(0)，看你 lerobot 版本 [file:91]

print("RAW keys:", sorted(sample.keys()))  # [file:91]
print("RAW has task_index?", "task_index" in sample)  # [file:91]
print("RAW has prompt?", "prompt" in sample)  # [file:91]
print("RAW task:", sample.get("task", None))  # [file:91]

# 3.5) 模拟 dataloader：如果 prompt_from_task=True，则在 repack 前注入 prompt
if _get_prompt_from_task_flag(data_cfg):
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_cfg.repo_id)  # 你 dataloader 里就是这么写的
    sample = T.PromptFromLeRobotTask(dataset_meta.tasks)(sample)  # 需要 sample 里有 task_index [file:90]
    print("AFTER PromptFromLeRobotTask has prompt?", "prompt" in sample)  # [file:90]
    print("AFTER PromptFromLeRobotTask prompt:", sample.get("prompt", None))  # [file:90]

# 4) 跑 repack（注意：这里用 compose 来跑 Group.inputs，避免依赖你本地 Group 的 apply_inputs 方法名）
repack_fn = T.compose(data_cfg.repack_transforms.inputs)  # [file:90]
x = repack_fn(sample)  # [file:90]
print("AFTER REPACK keys:", sorted(x.keys()))  # [file:90]

# 5) 跑 data_transforms.inputs
data_fn = T.compose(data_cfg.data_transforms.inputs)  # [file:90]
x2 = data_fn(x)  # [file:90]
print("AFTER data_transforms keys:", sorted(x2.keys()))  # [file:90]
print("prompt(after data_transforms):", x2.get("prompt", None))  # [file:90]

model_fn = T.compose(data_cfg.model_transforms.inputs)  
x3 = model_fn(x2)  
print("AFTER model_transforms keys:", sorted(x3.keys()))  
print("has tokenizedprompt?", "tokenized_prompt" in x3)  
