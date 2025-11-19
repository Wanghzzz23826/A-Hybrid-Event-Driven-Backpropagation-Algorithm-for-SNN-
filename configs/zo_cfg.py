# configs/zo_cfg.py
from dataclasses import dataclass
from typing import Optional
from .train_cfg import TrainConfig


@dataclass
class ZOTrainConfig(TrainConfig):
    """
    基于 TrainConfig 的 ZO 版本：
    - W_out 用显式 SGD
    - W_in / W_rec 用全维随机方向的零阶更新
    """

    # 训练轮数可以单独设
    epochs: int = 50

    # 读出层的学习率单独命名，避免和之前 cfg.lr 混淆
    lr_out: float = 5e-3
    wd: float = 1e-4

    # ZO 超参数
    zo_epsilon: float = 1e-2   # 扰动幅度 ε
    zo_lr: float = 1e-2        # θ 更新步长
    zo_queries: int = 4        # 每个 batch 采样方向数，越大越平滑

    # 不用 SPSA
    use_spsa: bool = False

    # 输出目录名单独一份，避免和 SPSA 跑到同一个目录
    run_dir_name: str = "runs_ZO"
