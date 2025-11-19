# trainer/trainer.py
from typing import List, Tuple, Dict, Any
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from configs.train_cfg import TrainConfig
from data.dataloader import make_dataloaders_from_npz
from models.srnn import EP_SRNNAdapter
from monitors.synchrony_monitor import SynchronyMoniter
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
from optim.spsa_ops import spsa_update_Wrec, spsa_update_Win, ce_loss_no_grad


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(310)
        self.device = auto_select_device("auto")

        # 路径
        self.run_dir, self.fig_dir = get_out_dirs(cfg.run_dir_name)
        print("[paths] RUN_DIR =", self.run_dir)
        print("[paths] FIG_DIR =", self.fig_dir)

        # 数据
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
            f"len(dl_tr)={len(self.dl_tr)}, epochs={cfg.epochs}, "
            f"max_steps={len(self.dl_tr) * cfg.epochs}"
        )

        # 模型
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

        for p in self.model.parameters():
            p.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()

        self.sync_mon = SynchronyMoniter(
            mode="hybrid",
            sigma=3.0,
            window=5,
            norm="min",
            alpha=0.85,
            local_sigma=1.5,
            local_window=5,
            local_kernel="gaussian",
        ).to(self.device)

        self.TOPK_PAIRS = self.cfg.topk_pairs

        # 训练状态
        self.trig_idx = 0
        self.global_step = 0
        self.history: List[Dict[str, Any]] = []
        self._cached_forward_flops = None
        self._cached_param_cnt = None

        self.train_loss_hist: List[float] = []
        self.val_loss_hist: List[float] = []
        self.val_acc_hist: List[float] = []

        self.trig_x_train: List[int] = []
        self.trig_y_train: List[float] = []
        self.trig_x_val: List[int] = []
        self.trig_y_val: List[float] = []

        self.epochs_to_plot = [cfg.epochs]
        self.raster_cache: List[Any] = []

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

        if len(self.trig_x_train) > 0:
            plt.scatter(self.trig_x_train, self.trig_y_train,
                        marker="o", s=35, label="Trigger (Train)")
            plt.scatter(self.trig_x_val, self.trig_y_val,
                        marker="x", s=35, label="Trigger (Val)")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        savefig_safe(self.fig_dir / "loss_curve_with_trigger.png")

        if len(self.raster_cache):
            plot_multi_epoch_rasters(
                self.raster_cache,
                self.fig_dir / "rasters_epochs.png",
            )

    def train(self):
        cfg = self.cfg
        A, a0, c0, alpha, gamma = cfg.A, cfg.a0, cfg.c0, cfg.alpha, cfg.gamma

        for ep in range(1, cfg.epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_cnt = 0
            epoch_triggered = False

            for X, y in self.dl_tr:
                self.global_step += 1
                X = X.to(self.device)
                y = y.to(self.device)

                self._estimate_flops_once(X)

                # ---- W_out 的手动 SGD ----
                with torch.no_grad():
                    logits, spk_seq = self.model(X, return_spk=True)
                    train_loss = float(self.criterion(logits, y).item())
                    running_loss += train_loss * X.size(0)
                    running_cnt += X.size(0)

                    B = X.size(0)
                    num_classes = logits.size(1)

                    probs = F.softmax(logits, dim=1)
                    y_onehot = F.one_hot(y, num_classes=num_classes).float()
                    grad_logits = (probs - y_onehot) / B

                    K_spk = min(10, spk_seq.size(1))
                    h = spk_seq[:, -K_spk:, :].float().mean(dim=1)

                    sgd_update_wout(self.model.W_out, grad_logits, h,
                                    lr=cfg.lr, wd=cfg.wd)

                # ---- 同步检测 + 局部 SPSA ----
                mon = {"trigger": False}
                if spk_seq.size(1) >= 2:
                    pre_spk = spk_seq[:, :-1, :]
                    post_spk = spk_seq[:, 1:, :]
                    mon = self.sync_mon.update_and_check(pre_spk, post_spk)

                if mon["trigger"]:
                    epoch_triggered = True
                    self.trig_idx += 1
                    print(
                        f"[TRIGGER] step={self.global_step} ep={ep} "
                        f"layer_sync={mon['layer_sync']:.4f} "
                        f"peak_tau={mon['peak_tau']:.4f} "
                        f"win_sum={mon['win_sum']} "
                        f"sub|post={len(mon['sub_idx']['post_idx'])} "
                        f"pre={len(mon['sub_idx']['pre_idx'])}"
                    )

                    S0 = mon["S"][0]
                    K = min(self.TOPK_PAIRS, S0.numel())
                    vals, flat_idx = torch.topk(S0.flatten(), k=K)
                    rows = (flat_idx // S0.size(1)).tolist()
                    cols = (flat_idx % S0.size(1)).tolist()
                    pairs = [
                        {"post": int(r), "pre": int(c), "score": float(v)}
                        for r, c, v in zip(
                            rows, cols, vals.detach().cpu().tolist()
                        )
                    ]

                    self.history.append(
                        {
                            "step": self.global_step,
                            "epoch": ep,
                            "phase": "sync_diag",
                            "top_pairs": pairs,
                            "layer_sync": float(mon["layer_sync"]),
                            "sync_ema": float(mon["sync_ema"]),
                            "peak_tau": float(mon["peak_tau"]),
                            "is_peak": bool(mon["is_peak"]),
                            "streak": int(mon["streak"]),
                            "win_sum": int(mon["win_sum"]),
                            "strong_cnt": int(mon["strong_cnt"]),
                            "trigger": True,
                            "strong_ratio": float(mon["strong_ratio"]),
                        }
                    )

                    if cfg.use_spsa:
                        post_idx = mon["sub_idx"]["post_idx"]
                        pre_idx = mon["sub_idx"]["pre_idx"]

                        # ---- W_rec 的局部掩码 ----
                        mask_rec = torch.zeros_like(self.model.W_rec, dtype=torch.bool)
                        post_idx_t = torch.as_tensor(post_idx, device=mask_rec.device, dtype=torch.long)
                        pre_idx_t = torch.as_tensor(pre_idx, device=mask_rec.device, dtype=torch.long)
                        mask_rec[post_idx_t[:, None], pre_idx_t[None, :]] = True
                        mask_rec &= self.model.core.rec_mask.bool()

                        # ---- W_in 的局部掩码（按输入参与度选）----
                        with torch.no_grad():
                            X_BTD = self.model._last_x_seq.to(self.model.W_in.device)
                            pre_part_in = (X_BTD > 0).float().sum(dim=(0, 1))
                            D_in = pre_part_in.numel()
                            k_pre_in = max(
                                self.sync_mon.min_sub,
                                int(np.ceil((1.0 - self.sync_mon.sub_pct) * D_in)),
                            )
                            _, idx_pre_in = torch.topk(pre_part_in, k=k_pre_in)

                            mask_in = torch.zeros_like(self.model.W_in, dtype=torch.bool)
                            mask_in[
                                torch.tensor(post_idx, device=mask_in.device)[:, None],
                                torch.tensor(idx_pre_in, device=mask_in.device)[None, :],
                            ] = True

                        a_k = a0 / ((self.trig_idx + A) ** alpha)
                        c_k = c0 / (max(1, self.trig_idx) ** gamma)

                        # ---- W_rec 更新 ----
                        with torch.no_grad():
                            W0_rec = self.model.W_rec.data.clone()
                            print("[DBG] nnz(mask_rec)=", int(mask_rec.sum().item()))

                        mon_spsa_rec = spsa_update_Wrec(
                            self.model,
                            X,
                            y,
                            ce_loss=ce_loss_no_grad,
                            a_k=a_k,
                            c_k=c_k,
                            grad_clip=cfg.spsa_grad_clip,
                            clip_norm=cfg.spsa_clip_norm,
                            auto_gain=False,
                            target_update_norm=0.25,
                            gain_clip=(0.2, 5.0),
                            mask=mask_rec,
                        )

                        with torch.no_grad():
                            dW_rec = self.model.W_rec.data - W0_rec
                            print(
                                "[SPSA W_rec] ||ΔW||2 =",
                                float(dW_rec.norm().item()),
                                " ||ΔW(mask)||2 =",
                                float((dW_rec * mask_rec).norm().item()),
                            )

                        # ---- W_in 更新 ----
                        with torch.no_grad():
                            W0_in = self.model.W_in.data.clone()
                            print("[DBG] nnz(mask_in)=", int(mask_in.sum().item()))

                        mon_spsa_in = spsa_update_Win(
                            self.model,
                            X,
                            y,
                            ce_loss=ce_loss_no_grad,
                            a_k=a_k,
                            c_k=c_k,
                            grad_clip=cfg.spsa_grad_clip,
                            clip_norm=cfg.spsa_clip_norm,
                            auto_gain=False,
                            target_update_norm=0.15,
                            gain_clip=(0.2, 5.0),
                            mask=mask_in,
                        )

                        with torch.no_grad():
                            dW_in = self.model.W_in.data - W0_in
                            print(
                                "[SPSA W_in] ||ΔW||2 =",
                                float(dW_in.norm().item()),
                                " ||ΔW(mask)||2 =",
                                float((dW_in * mask_in).norm().item()),
                            )

            # ---- epoch 结束：验证 ----
            epoch_loss = running_loss / max(running_cnt, 1)
            self.model.eval()
            val_acc = evaluate(self.model, self.dl_va, self.device)
            val_loss = evaluate_loss(self.model, self.dl_va, self.device)

            self.history.append(
                {
                    "step": self.global_step,
                    "epoch": ep,
                    "phase": "epoch",
                    "train_loss": float(epoch_loss),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "trig_in_epoch": bool(epoch_triggered),
                    "flops": float(self._cached_forward_flops or 0.0),
                    "params": float(self._cached_param_cnt or 0.0),
                }
            )

            self.train_loss_hist.append(epoch_loss)
            self.val_loss_hist.append(val_loss)
            self.val_acc_hist.append(val_acc)

            if epoch_triggered:
                self.trig_x_train.append(ep)
                self.trig_y_train.append(epoch_loss)
                self.trig_x_val.append(ep)
                self.trig_y_val.append(val_loss)

            print(
                f"Epoch {ep}: Train Loss = {epoch_loss:.4f}, "
                f"Val Acc = {val_acc:.2f}%, Val Loss = {val_loss:.4f}"
            )

            if ep in self.epochs_to_plot:
                with torch.no_grad():
                    _ = self.model(self.X_ref.to(self.device))
                    t_idx, n_idx, TT, Hsel = raster_coords_from_spk_seq(
                        self.model._last_spk_seq,
                        max_neurons=self.cfg.raster_max_neurons,
                    )
                    self.raster_cache.append((ep, (t_idx, n_idx, TT, Hsel)))

            if cfg.enable_plots:
                pd.DataFrame(self.history).to_csv(
                    self.run_dir / "rsnn_spsa_history.csv",
                    index=False,
                )
                self._plot_epoch_curves(ep)

        # ---- 最后评估 ----
        acc_tr = evaluate(self.model, self.dl_tr, self.device)
        acc_va = evaluate(self.model, self.dl_va, self.device)
        acc_te = evaluate(self.model, self.dl_te, self.device)
        tag = "SRNN+SPSA(on-trigger)" if cfg.use_spsa else "SRNN(no-learning)"

        print(
            f"[TRAIN] global_step={self.global_step}, trig_idx={self.trig_idx}, "
            f"trigger_ratio={self.trig_idx / max(self.global_step, 1):.3f}"
        )
        print(
            f"{tag} | Train: {acc_tr:.2f}% | Val: {acc_va:.2f}% | Test: {acc_te:.2f}%"
        )
        print(f"total trigger num:{self.trig_idx:d}")
