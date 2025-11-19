# configs/train_cfg.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    # 数据
    npz_path: str = "nmini_0v1_T50_singlech.npz"

    # 模型
    hidden_dim: int = 512
    tau_m: float = 4.0
    out_tau: float = 5.0
    thr: float = 0.35

    # 训练
    batch_size: int = 64
    epochs: int = 50
    lr: float = 5e-3
    wd: float = 1e-4

    # SPSA 超参
    steps: int = 300
    a0: float = 2e-1
    c0: float = 2e-2
    A: float = 38.0
    alpha: float = 0.602
    gamma: float = 0.101
    spsa_grad_clip: Optional[float] = None
    spsa_clip_norm: float = 0.1
    wrec_param_clip: float = 4.5
    use_spsa: bool = True

    # 同步监测
    topk_pairs: int = 15  #######

    # 可视化
    enable_plots: bool = True
    raster_max_neurons: int = 128

    # 输出路径名
    run_dir_name: str = "runs_SPSA"
