# models/srnn.py
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRNN(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        n_t: int,
        thr: float = 1.0,
        tau_m: float = 20.0,
        tau_o: float = 20.0,
        b_o: float = 0.0,
        gamma: float = 0.3,
        dt: float = 1.0,
        classif: bool = True,
        w_init_gain=(1.0, 1.0, 1.0),
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_t = n_t
        self.thr = thr
        self.dt = dt
        self.alpha = np.exp(-dt / tau_m)
        self.kappa = np.exp(-dt / tau_o)
        self.gamma = gamma
        self.b_o = b_o
        self.classif = classif
        self.device = device

        self.w_in = nn.Parameter(torch.Tensor(n_rec, n_in))
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))

        self.register_buffer(
            "rec_mask",
            torch.ones(n_rec, n_rec, device=device)
            - torch.eye(n_rec, n_rec, device=device),
        )

        self.reset_parameters(w_init_gain)

    def reset_parameters(self, gain):
        nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0] * self.w_in.data
        nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1] * self.w_rec.data
        nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data

    def init_net(self, n_b: int):
        T, H, O = self.n_t, self.n_rec, self.n_out
        self.v = torch.zeros(T, n_b, H, device=self.device)
        self.vo = torch.zeros(T, n_b, O, device=self.device)
        self.z = torch.zeros(T, n_b, H, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, D]
        return yo: [T, B, O]
        """
        T, B, _ = x.shape
        assert T == self.n_t, f"X T={T} 与模型 n_t={self.n_t} 不一致"
        self.init_net(B)

        w_rec_eff = self.w_rec * self.rec_mask
        for t in range(T - 1):
            self.v[t + 1] = (
                self.alpha * self.v[t]
                + torch.mm(self.z[t], w_rec_eff.t())
                + torch.mm(x[t], self.w_in.t())
            ) - self.z[t] * self.thr
            self.z[t + 1] = (self.v[t + 1] > self.thr).float()
            self.vo[t + 1] = (
                self.kappa * self.vo[t]
                + torch.mm(self.z[t + 1], self.w_out.t())
                + self.b_o
            )

        yo = F.softmax(self.vo, dim=2) if self.classif else self.vo
        return yo

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.n_in} -> {self.n_rec} -> {self.n_out})"


class EP_SRNNAdapter(nn.Module):
    """
    适配 e-prop 的 SRNN 到现在架构：
    - inputs: X [B, T, D]
    - 暴露 W_in/W_rec/W_out
    """
    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        n_t: int,
        thr: float = 1.0,
        tau_m: float = 20.0,
        tau_o: float = 20.0,
        b_o: float = 0.0,
        gamma: float = 0.3,
        dt: float = 1.0,
        classif: bool = True,
        w_init_gain=(1.0, 1.0, 1.0),
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.core = SRNN(
            n_in=n_in,
            n_rec=n_rec,
            n_out=n_out,
            n_t=n_t,
            thr=thr,
            tau_m=tau_m,
            tau_o=tau_o,
            b_o=b_o,
            gamma=gamma,
            dt=dt,
            classif=classif,
            w_init_gain=w_init_gain,
            device=device,
        )
        self.n_in, self.n_rec, self.n_out, self.n_t = n_in, n_rec, n_out, n_t
        self.device = device

        self.W_in = self.core.w_in
        self.W_rec = self.core.w_rec
        self.W_out = self.core.w_out

        self.register_buffer("_last_x_seq", torch.zeros(1, n_t, n_in))
        self.register_buffer("_last_spk_seq", torch.zeros(1, n_t, n_rec))
        self.register_buffer("_last_mem_seq", torch.zeros(1, n_t, n_rec))
        self.register_buffer("_last_forward_macs", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_spike_rate_for_reg", torch.tensor(0.0))

        self.cached_pre_any = None
        self.cached_post_any = None
        self.cached_delta_t = None

    @staticmethod
    def _to_TBN(X_BTD: torch.Tensor) -> torch.Tensor:
        return X_BTD.transpose(0, 1)  # [T,B,D]
    
        # ==== 零阶优化专用接口 ====
    def get_flatten_theta(self) -> torch.Tensor:
        """
        把 W_in 和 W_rec 展平成一个 1D 向量，用于 ZO/SPSA 等零阶优化。
        """
        return torch.cat([
            self.W_in.view(-1),
            self.W_rec.view(-1),
        ], dim=0)

    @torch.no_grad()
    def set_theta_from_flatten(self, theta: torch.Tensor):
        """
        从扁平化参数向量 theta 还原回 W_in 和 W_rec。
        theta.numel() 必须等于 W_in.numel() + W_rec.numel().
        """
        n_in = self.W_in.numel()
        n_rec = self.W_rec.numel()
        assert theta.numel() == n_in + n_rec, \
            f"theta.len={theta.numel()} != W_in+W_rec={n_in + n_rec}"

        theta_in  = theta[:n_in].view_as(self.W_in)
        theta_rec = theta[n_in:].view_as(self.W_rec)

        self.W_in.data.copy_(theta_in)
        self.W_rec.data.copy_(theta_rec)


    def forward(self, X_BTD: torch.Tensor, return_spk: bool = False):
        T = self.n_t
        assert (
            X_BTD.shape[1] == T
        ), f"X T={X_BTD.shape[1]} 与模型 n_t={T} 不一致"

        x_TBD = self._to_TBN(X_BTD).contiguous()
        yo = self.core(x_TBD)

        K = min(10, T)
        logits_BC = yo[-K:].mean(dim=0)

        # 缓存时序
        self._last_x_seq = X_BTD.detach()
        self._last_spk_seq = self.core.z.detach().transpose(0, 1)
        self._last_mem_seq = self.core.v.detach().transpose(0, 1)
        self._spike_rate_for_reg = self._last_spk_seq.float().mean()

        with torch.no_grad():
            batch_size = X_BTD.size(0)
            seq_length = X_BTD.size(1)
            input_dim = X_BTD.size(2)
            hidden_dim = self._last_spk_seq.size(2)

            input_spike_sequence = (X_BTD > 0).to(X_BTD.dtype)
            spike_sequence = self._last_spk_seq.to(X_BTD.dtype)

            pre_any_per_neuron = (input_spike_sequence.sum(dim=1) > 0).float()
            post_any_per_neuron = (spike_sequence.sum(dim=1) > 0).float()

            pre_time_mask = (input_spike_sequence.sum(dim=2) > 0)
            post_time_mask = (spike_sequence.sum(dim=2) > 0)

            time_idx = torch.arange(seq_length, device=X_BTD.device).unsqueeze(0).expand(batch_size, -1)
            big = torch.full_like(time_idx, fill_value=seq_length)

            t_pre_scalar = torch.where(pre_time_mask, time_idx, big).min(dim=1).values.float()
            t_post_scalar = torch.where(post_time_mask, time_idx, big).min(dim=1).values.float()
            delta_t_scalar = (t_post_scalar - t_pre_scalar).abs()

            valid_pre = pre_time_mask.any(dim=1)
            valid_post = post_time_mask.any(dim=1)
            valid_pair = valid_pre & valid_post

            inf_like = torch.full_like(delta_t_scalar, float("inf"))
            delta_t_scalar = torch.where(valid_pair, delta_t_scalar, inf_like)

            self.cached_pre_any = pre_any_per_neuron.detach()
            self.cached_post_any = post_any_per_neuron.detach()
            self.cached_delta_t = delta_t_scalar.detach()

        if return_spk:
            return logits_BC, self._last_spk_seq
        return logits_BC
