# monitors/synchrony_monitor.py
from typing import Optional, Dict, Any
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SynchronyMoniter(nn.Module):
    """
    计算突触对 (post_j <- pre_i) 的同步分数：
    mode="all","first","hybrid"
    """
    def __init__(
        self,
        sigma: float = 2.0,
        window: int = 5,
        kernel: str = "gaussian",
        norm: str = "min",
        mode: str = "all",
        alpha: float = 0.8,
        local_sigma: Optional[float] = None,
        local_window: Optional[int] = None,
        local_kernel: str = "gaussian",
    ):
        super().__init__()
        assert kernel in ("gaussian", "box")
        assert norm in ("min", "max", "none")
        assert mode in ("all", "first", "hybrid")
        assert local_kernel in ("gaussian", "box")
        assert 0.0 <= alpha <= 1.0

        self.mode = mode
        self.kernel = kernel
        self.norm = norm
        self.sigma = float(sigma)
        self.window = int(window)

        self.alpha = float(alpha)
        self.local_kernel = local_kernel
        self.local_sigma = float(local_sigma) if local_sigma is not None else max(1.0, self.sigma / 2.0)
        self.local_window = int(local_window) if local_window is not None else max(1, self.window // 2)

        self._streak = 0                                #检测连续“强同步帧”次数

        self.hist_len = 200                             #同步历史缓存长度     
        self.sync_hist = deque(maxlen=self.hist_len)    #同步历史layer_sync的滑动窗口
        self.ema_beta = 0.9                             #EMA平滑系数
        self.sync_ema = 0.0                             #同步EMA均值
        self.cooldown = 0                               #冷却计数器
        self._cool = 0                                  #内部冷却计数器
        self.need_streak = 1                            #需要的连续“强同步帧”次数

        self.win_len_pairs = 40                         #强同步对窗口长度
        self.strong_tau = 0.08                          #强同步阈值
        self.strong_hist = deque(maxlen=self.win_len_pairs)  #强同步历史缓存
        self.nsync_thresh = 50                          #同步阈值

        self.sub_pct = 0.80                             #局部更新参与度百分比
        self.min_sub = 3                                #局部更新最小参与数

    def _build_kernel(self, T: int, device, *, kernel: str, sigma: float, window: int):
        if kernel == "gaussian":
            half = max(1, int(4 * sigma))
            t = torch.arange(-half, half + 1, device=device).float()
            k = torch.exp(-(t ** 2) / (2 * (sigma ** 2)))
            k = k / k.sum().clamp_min(1e-8)
        else:
            half = window
            k = torch.ones(2 * half + 1, device=device).float()
            k = k / k.sum()
        return k.view(1, 1, -1)

    @staticmethod
    def _first_times(spk: torch.Tensor) -> torch.Tensor:
        B, T, C = spk.shape
        big = torch.full((B, 1, 1), T, device=spk.device)
        t_idx = torch.arange(T, device=spk.device).view(1, T, 1)
        t_first = torch.where(spk.bool(), t_idx, big).min(dim=1).values.float()
        return t_first

    def _score_first(self, pre_spk, post_spk, *, sigma: float, norm: str):
        B, T, C_in = pre_spk.shape
        _, _, C_out = post_spk.shape

        pre_first = self._first_times(pre_spk)
        post_first = self._first_times(post_spk)

        valid_pre = (pre_first < T)
        valid_post = (post_first < T)

        dt = (post_first.unsqueeze(2) - pre_first.unsqueeze(1)).abs()
        dt = torch.where(
            valid_post.unsqueeze(2) & valid_pre.unsqueeze(1),
            dt,
            torch.full_like(dt, float("inf")),
        )

        S = torch.exp(-(dt ** 2) / (2 * sigma ** 2))

        if norm != "none":
            pre_cnt = pre_spk.sum(1).clamp_min(1e-6)
            post_cnt = post_spk.sum(1).clamp_min(1e-6)
            if norm == "min":
                z = torch.minimum(
                    pre_cnt.unsqueeze(1).expand(-1, C_out, -1),
                    post_cnt.unsqueeze(2).expand(-1, -1, C_in),
                )
            else:
                z = torch.maximum(
                    pre_cnt.unsqueeze(1).expand(-1, C_out, -1),
                    post_cnt.unsqueeze(2).expand(-1, -1, C_in),
                )
            S = S / z
        return S

    def _score_all(self, pre_spk, post_spk, *, kernel: str, sigma: float, window: int, norm: str):
        B, T, C_in = pre_spk.shape
        _, _, C_out = post_spk.shape
        k = self._build_kernel(T, pre_spk.device, kernel=kernel, sigma=sigma, window=window)

        pre = pre_spk.transpose(1, 2)
        pre_f = F.conv1d(
            pre,
            k.expand(C_in, 1, -1),
            padding=k.size(-1) // 2,
            groups=C_in,
        ).transpose(1, 2)
        post = post_spk.transpose(1, 2)
        S = torch.einsum("bjt,bti->bji", post, pre_f)

        if norm != "none":
            pre_count = pre_spk.sum(1).clamp_min(1e-6)
            post_count = post_spk.sum(1).clamp_min(1e-6)
            if norm == "min":
                z = torch.minimum(
                    pre_count.unsqueeze(1).expand(-1, C_out, -1),
                    post_count.unsqueeze(2).expand(-1, -1, C_in),
                )
            else:
                z = torch.maximum(
                    pre_count.unsqueeze(1).expand(-1, C_out, -1),
                    post_count.unsqueeze(2).expand(-1, -1, C_in),
                )
            S = S / z

        valid = (pre_spk.sum(1) > 0).unsqueeze(1) & (post_spk.sum(1) > 0).unsqueeze(2)
        S = torch.where(valid, S, torch.zeros_like(S))
        return S

    @torch.no_grad()
    def forward(self, pre_spk, post_spk):
        if self.mode == "first":
            S = self._score_first(pre_spk, post_spk, sigma=self.sigma, norm=self.norm)
        elif self.mode == "all":
            S = self._score_all(
                pre_spk,
                post_spk,
                kernel=self.kernel,
                sigma=self.sigma,
                window=self.window,
                norm=self.norm,
            )
        else:  # hybrid
            S_first = self._score_first(pre_spk, post_spk, sigma=self.sigma, norm=self.norm)
            B, T, C_in = pre_spk.shape
            _, _, C_out = post_spk.shape
            pre_first = self._first_times(pre_spk)
            post_first = self._first_times(post_spk)

            t_idx = torch.arange(T, device=pre_spk.device).view(1, T, 1)

            pre_valid = (pre_first < T)
            pre_low = (pre_first - self.local_window).clamp_min(0).floor().long()
            pre_high = (pre_first + self.local_window).clamp_max(T - 1).ceil().long()
            pre_mask = (t_idx >= pre_low.unsqueeze(1)) & (t_idx <= pre_high.unsqueeze(1))
            pre_mask = pre_mask & pre_valid.unsqueeze(1)

            post_valid = (post_first < T)
            post_low = (post_first - self.local_window).clamp_min(0).floor().long()
            post_high = (post_first + self.local_window).clamp_max(T - 1).ceil().long()
            post_mask = (t_idx >= post_low.unsqueeze(1)) & (t_idx <= post_high.unsqueeze(1))
            post_mask = post_mask & post_valid.unsqueeze(1)

            pre_local = pre_spk * pre_mask.float()
            post_local = post_spk * post_mask.float()

            S_local = self._score_all(
                pre_local,
                post_local,
                kernel=self.local_kernel,
                sigma=self.local_sigma,
                window=self.local_window,
                norm=self.norm,
            )
            S = self.alpha * S_first + (1.0 - self.alpha) * S_local

        layer_sync = S.mean(dim=(1, 2))
        post_particip = S.mean(dim=2)
        pre_particip = S.mean(dim=1)
        agg = {
            "layer_sync": layer_sync,
            "post_particip": post_particip,
            "pre_particip": pre_particip,
        }
        return S, agg

    @torch.no_grad()
    def update_and_check(self, pre_spk, post_spk) -> Dict[str, Any]:
        S, agg = self(pre_spk, post_spk)
        layer_sync = float(agg["layer_sync"].mean().item())
        self.sync_ema = self.ema_beta * self.sync_ema + (1.0 - self.ema_beta) * layer_sync
        self.sync_hist.append(layer_sync)

        S0 = S[0]
        strong_cnt = int((S0 > self.strong_tau).sum().item())
        self.strong_hist.append(strong_cnt)
        win_sum = int(sum(self.strong_hist))

        if len(self.sync_hist) >= max(20, int(0.25 * self.hist_len)):
            peak_tau = float(np.quantile(np.array(self.sync_hist), 0.9))
        else:
            peak_tau = 0.08

        is_peak = (layer_sync > peak_tau)
        self._streak = self._streak + 1 if is_peak else 0
        win_over = True

        if self._cool > 0:
            self._cool -= 1

        trigger = (self._streak >= self.need_streak) and win_over and (self._cool == 0)

        post_part = agg["post_particip"][0]
        pre_part = agg["pre_particip"][0]
        H = post_part.numel()
        k_post = max(self.min_sub, int(np.ceil((1.0 - self.sub_pct) * H)))
        k_pre = max(self.min_sub, int(np.ceil((1.0 - self.sub_pct) * H)))
        _, idx_post = torch.topk(post_part, k=k_post)
        _, idx_pre = torch.topk(pre_part, k=k_pre)

        sub = {
            "post_idx": idx_post.detach().cpu().tolist(),
            "pre_idx": idx_pre.detach().cpu().tolist(),
        }

        if trigger:
            self._cool = self.cooldown
            self._streak = 0

        strong_ratio = strong_cnt / float(S0.numel())

        info = {
            "S": S,
            "agg": agg,
            "layer_sync": layer_sync,
            "sync_ema": self.sync_ema,
            "peak_tau": peak_tau,
            "is_peak": bool(is_peak),
            "streak": int(self._streak),
            "win_sum": win_sum,
            "trigger": bool(trigger),
            "strong_cnt": strong_cnt,
            "sub_idx": sub,
            "strong_ratio": float(strong_ratio),
        }
        return info


@torch.no_grad()
def hybrid_decompose(sync_mon: SynchronyMoniter, pre_spk, post_spk):
    S_first = sync_mon._score_first(pre_spk, post_spk,
                                    sigma=sync_mon.sigma, norm=sync_mon.norm)
    B, T, C_in = pre_spk.shape
    _, _, C_out = post_spk.shape
    pre_first = sync_mon._first_times(pre_spk)
    post_first = sync_mon._first_times(post_spk)
    t_idx = torch.arange(T, device=pre_spk.device).view(1, T, 1)

    pre_low = (pre_first - sync_mon.local_window).clamp_min(0).floor().long()
    pre_high = (pre_first + sync_mon.local_window).clamp_max(T - 1).ceil().long()
    post_low = (post_first - sync_mon.local_window).clamp_min(0).floor().long()
    post_high = (post_first + sync_mon.local_window).clamp_max(T - 1).ceil().long()

    pre_mask = (t_idx >= pre_low.unsqueeze(1)) & (t_idx <= pre_high.unsqueeze(1))
    post_mask = (t_idx >= post_low.unsqueeze(1)) & (t_idx <= post_high.unsqueeze(1))
    pre_mask = pre_mask & (pre_first < T).unsqueeze(1)
    post_mask = post_mask & (post_first < T).unsqueeze(1)

    pre_local = pre_spk * pre_mask.float()
    post_local = post_spk * post_mask.float()
    S_local = sync_mon._score_all(
        pre_local,
        post_local,
        kernel=sync_mon.local_kernel,
        sigma=sync_mon.local_sigma,
        window=sync_mon.local_window,
        norm=sync_mon.norm,
    )
    S_h = sync_mon.alpha * S_first + (1.0 - sync_mon.alpha) * S_local

    return {
        "first": float(S_first.mean(dim=(1, 2)).mean().item()),
        "local": float(S_local.mean(dim=(1, 2)).mean().item()),
        "hybrid": float(S_h.mean(dim=(1, 2)).mean().item()),
    }
