# trainer/zo_trainer.py
from typing import List
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from configs.zo_cfg import ZOTrainConfig
from data.dataloader import make_dataloaders_from_npz
from models.srnn import EP_SRNNAdapter
from utils.seed import set_seed
from utils.device import auto_select_device
from utils.plot import (
    get_out_dirs,
    savefig_safe,
    raster_coords_from_spk_seq,
    plot_multi_epoch_rasters,
)
from utils.metrics import evaluate, evaluate_loss, estimate_forward_flops
from optim.sgd_ops import sgd_update_wout
from optim.spsa_ops import ce_loss_no_grad   # 里面的 ce_loss_no_grad 可以直接复用


class ZOTrainer:
    def __init__(self, cfg: ZOTrainConfig):
        self.cfg = cfg
        set_seed(310)
        self.device = auto_select_device("auto")

        # ---- 路径 ----
        self.run_dir, self.fig_dir = get_out_dirs(cfg.run_dir_name)
        print("[paths] RUN_DIR =", self.run_dir)
        print("[paths] FIG_DIR =", self.fig_dir)

        # ---- 数据 ----
        self.dl_tr, self.dl_va, self.dl_te, self.meta = make_dataloaders_from_npz(
            cfg.npz_path,
            batch_size=cfg.batch_size,
            device=self.device,
        )
        D, C = self.meta["D"], self.meta["num_classes"]
        self.T = self.meta["T"]
        self.D = D
        self.C = C
        self.X_ref, self.y_ref = next(iter(self.dl_va))

        print(
            f"[DATA] N_tr={self.meta['N_tr']}, batch_size={cfg.batch_size}, "
            f"len(dl_tr)={len(self.dl_tr)}, epochs={cfg.epochs}"
        )

        # ---- 模型 ----
        self.model = EP_SRNNAdapter(
            n_in=D,
            n_rec=cfg.hidden_dim,
            n_out=C,
            n_t=self.T,
            thr=cfg.thr,
            tau_m=cfg.tau_m,
            tau_o=cfg.out_tau,
            b_o=0.0,
            gamma=0.3,
            dt=1.0,
            classif=False,
            w_init_gain=(1.0, 1.0, 1.0),
            device=self.device,
        ).to(self.device)

        # 关闭 autograd：所有参数都只用手写更新
        for p in self.model.parameters():
            p.requires_grad = False

        self.ce = nn.CrossEntropyLoss()

        # 记录用
        self.global_step = 0
        self._cached_forward_flops = None
        self._cached_param_cnt = None

        self.train_loss_hist: List[float] = []
        self.val_loss_hist: List[float] = []
        self.val_acc_hist: List[float] = []

        self.raster_cache = []
        self.epochs_to_plot = [cfg.epochs]

    # ----------------- 内部小工具 -----------------
    def _estimate_flops_once(self, X: torch.Tensor):
        if self._cached_forward_flops is None:
            with torch.no_grad():
                fwd_flops, n_params = estimate_forward_flops(self.model, X)
            self._cached_forward_flops = fwd_flops
            self._cached_param_cnt = n_params

    def _plot_epoch_curves(self, epoch: int):
        epochs_axis = range(1, epoch + 1)
        plt.figure()
        plt.plot(epochs_axis, self.train_loss_hist, label="Train Loss")
        plt.plot(epochs_axis, self.val_loss_hist, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        savefig_safe(self.fig_dir / "loss_curve.png")

        if len(self.raster_cache):
            plot_multi_epoch_rasters(
                self.raster_cache,
                self.fig_dir / "rasters_epochs.png",
            )

    # ----------------- 核心训练循环 -----------------
    def train(self):
        cfg = self.cfg

        for ep in range(1, cfg.epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_cnt = 0

            for X, y in self.dl_tr:
                self.global_step += 1
                X = X.to(self.device)
                y = y.to(self.device)
                B = X.size(0)

                self._estimate_flops_once(X)

                # ===== 1) 用当前参数跑一遍，更新 W_out（显式 SGD）=====
                with torch.no_grad():
                    logits, spk_seq = self.model(X, return_spk=True)   # logits: [B,C]
                    loss = self.ce(logits, y)
                    train_loss = float(loss.item())

                    running_loss += train_loss * B
                    running_cnt += B

                    num_classes = logits.size(1)

                    probs = F.softmax(logits, dim=1)
                    y_onehot = F.one_hot(y, num_classes=num_classes).float()
                    grad_logits = (probs - y_onehot) / B

                    # 最近 K 个时间步的平均隐藏状态
                    K_spk = min(10, spk_seq.size(1))
                    h = spk_seq[:, -K_spk:, :].float().mean(dim=1)   # [B,H]

                    # 手写 SGD 更新 W_out
                    sgd_update_wout(
                        self.model.W_out,
                        grad_logits,
                        h,
                        lr=cfg.lr_out,
                        wd=cfg.wd,
                    )

                # ===== 2) 零阶更新 W_in + W_rec（全维扰动）=====
                with torch.no_grad():
                    theta0 = self.model.get_flatten_theta().clone()
                    g_hat = torch.zeros_like(theta0)

                    for _ in range(cfg.zo_queries):
                        # 随机方向 u，并 L2 归一化
                        u = torch.randn_like(theta0)
                        u = u / (u.norm() + 1e-8)

                        # θ+ = θ0 + εu
                        self.model.set_theta_from_flatten(theta0 + cfg.zo_epsilon * u)
                        loss_pos = ce_loss_no_grad(self.model, X, y)

                        # θ- = θ0 - εu
                        self.model.set_theta_from_flatten(theta0 - cfg.zo_epsilon * u)
                        loss_neg = ce_loss_no_grad(self.model, X, y)

                        # g ≈ (L+ - L-) / (2ε) * u
                        g_hat += (loss_pos - loss_neg) / (2.0 * cfg.zo_epsilon) * u

                    g_hat /= float(cfg.zo_queries)

                    # θ ← θ - η * g_hat
                    theta_new = theta0 - cfg.zo_lr * g_hat
                    self.model.set_theta_from_flatten(theta_new)

            # ===== epoch 结束：验证 =====
            epoch_loss = running_loss / max(running_cnt, 1)
            self.model.eval()
            val_acc = evaluate(self.model, self.dl_va, self.device)
            val_loss = evaluate_loss(self.model, self.dl_va, self.device)

            self.train_loss_hist.append(epoch_loss)
            self.val_loss_hist.append(val_loss)
            self.val_acc_hist.append(val_acc)

            print(
                f"[ZO] Epoch {ep}: Train Loss = {epoch_loss:.4f}, "
                f"Val Acc = {val_acc:.2f}%, Val Loss = {val_loss:.4f}"
            )

            # 画 raster（可选）
            if ep in self.epochs_to_plot:
                with torch.no_grad():
                    _ = self.model(self.X_ref.to(self.device))
                    t_idx, n_idx, TT, Hsel = raster_coords_from_spk_seq(
                        self.model._last_spk_seq,
                        max_neurons=cfg.raster_max_neurons,
                    )
                    self.raster_cache.append((ep, (t_idx, n_idx, TT, Hsel)))

            # 保存历史 & 图
            hist_row = {
                "epoch": ep,
                "train_loss": float(epoch_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "flops": float(self._cached_forward_flops or 0.0),
                "params": float(self._cached_param_cnt or 0.0),
            }
            df = pd.DataFrame([hist_row])
            hist_path = self.run_dir / "zo_history.csv"
            if hist_path.exists():
                # 追加模式：读旧的再 concat
                old = pd.read_csv(hist_path)
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(hist_path, index=False)

            if cfg.enable_plots:
                self._plot_epoch_curves(ep)

        # ===== 最后评估 Train / Val / Test =====
        acc_tr = evaluate(self.model, self.dl_tr, self.device)
        acc_va = evaluate(self.model, self.dl_va, self.device)
        acc_te = evaluate(self.model, self.dl_te, self.device)
        print(
            f"[ZO-SRNN] Train: {acc_tr:.2f}% | "
            f"Val: {acc_va:.2f}% | Test: {acc_te:.2f}%"
        )
