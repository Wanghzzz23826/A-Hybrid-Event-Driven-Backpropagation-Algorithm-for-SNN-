# rsnn_with_spsa.py
#only W_out use SGD W_in and W_rec use SPSA do empirical gradient
#test modify the parameters 
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt 
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import torch.nn.functional as F
from collections import deque
from spsa import spsa_update_Wrec,ce_loss_no_grad,spsa_update_Win
USE_SPSA = True 
try:
    from thop import profile
    _HAVE_THOP = True
except Exception:
    profile = None
    _HAVE_THOP = False
#-------------------------------------------------------------------------------------------
def get_out_dirs():
    try:
        base = Path(__file__).parent
    except NameError:
        base = Path.cwd()
    run_dir = (base / "runs_SPSA").resolve()
    fig_dir = (run_dir / "figs").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, fig_dir

RUN_DIR, FIG_DIR = get_out_dirs()
print("[paths] RUN_DIR =", RUN_DIR)
print("[paths] FIG_DIR =", FIG_DIR)

def savefig_safe(pathlike):
    p = Path(pathlike)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
# ---------------------------------------------------------------------

# 随机种子设置和设备选择
def set_seed(seed: int = 310):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auto_select_device(req: str = "auto") -> torch.device:
    if req != "auto": return torch.device(req)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# 数据集加载和预处理
#x[N,T,D]
class NMiniDataset(Dataset):
    """X: [N,T,D], y: [N]"""
    def __init__(self, X: np.ndarray, y: np.ndarray, dtype=torch.float32):
        assert X.ndim==3 and len(X)==len(y)
        self.X = torch.from_numpy(X).to(dtype)
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_nmini_npz(npz_path: str):  #训练集测试集验证集划分
    data = np.load(npz_path)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    T = Xtr.shape[1]; D = Xtr.shape[2]
    C = int(np.max(np.concatenate([ytr,yva,yte]))+1)
    meta = {"T":T,"D":D,"num_classes":C,"N_tr":len(ytr),"N_va":len(yva),"N_te":len(yte)}
    return (Xtr,ytr,Xva,yva,Xte,yte,meta)

def make_dataloaders_from_npz(npz_path: str, batch_size=256, device=torch.device("cpu"), num_workers=0):
    Xtr,ytr,Xva,yva,Xte,yte,meta = load_nmini_npz(npz_path)
    ds_tr, ds_va, ds_te = NMiniDataset(Xtr,ytr), NMiniDataset(Xva,yva), NMiniDataset(Xte,yte)
    pin = (device.type=="cuda")
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return dl_tr, dl_va, dl_te, meta

# ---------------------------------------------------------------------
# RSNN 

class SRNN(nn.Module):
    def __init__(self, n_in, n_rec, n_out, n_t,
                 thr=1.0, tau_m=20.0, tau_o=20.0, b_o=0.0,
                 gamma=0.3, dt=1.0, classif=True,
                 w_init_gain=(1.0, 1.0, 1.0),
                 device=torch.device("cpu")):
        super().__init__()
        self.n_in = n_in; self.n_rec = n_rec; self.n_out = n_out
        self.n_t = n_t; self.thr = thr; self.dt = dt
        self.alpha = np.exp(-dt / tau_m)
        self.kappa = np.exp(-dt / tau_o)
        self.gamma = gamma
        self.b_o = b_o
        self.classif = classif
        self.device = device

        # 参数
        self.w_in  = nn.Parameter(torch.Tensor(n_rec, n_in))
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))

        # 去自环
        self.register_buffer(
            "rec_mask",
            torch.ones(n_rec, n_rec, device=device) - torch.eye(n_rec, n_rec, device=device)
        )

        self.reset_parameters(w_init_gain)

    def reset_parameters(self, gain):
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0] * self.w_in.data
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1] * self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data

    def init_net(self, n_b):
        T, H, O = self.n_t, self.n_rec, self.n_out
        # 隐状态
        self.v  = torch.zeros(T, n_b, H, device=self.device)
        self.vo = torch.zeros(T, n_b, O, device=self.device)
        # 可见脉冲
        self.z  = torch.zeros(T, n_b, H, device=self.device)

    def forward(self, x):
        """
        x: [T, B, D]
        return yo: [T, B, O]  (若 classif=True，则 softmax 概率)
        """
        T, B, _ = x.shape
        assert T == self.n_t, f"X T={T} 与模型 n_t={self.n_t} 不一致"
        self.init_net(B)

        w_rec_eff = self.w_rec * self.rec_mask
        for t in range(T - 1):
            # 膜电位与发放
            self.v[t+1]  = (self.alpha * self.v[t]
                           + torch.mm(self.z[t], w_rec_eff.t())
                           + torch.mm(x[t], self.w_in.t())) - self.z[t] * self.thr
            self.z[t+1]  = (self.v[t+1] > self.thr).float()
            # 读出--指数加权平滑的线性读出////同时可以尝试均值膜电位线性读出
            self.vo[t+1] = self.kappa * self.vo[t] + torch.mm(self.z[t+1], self.w_out.t()) + self.b_o

        yo = F.softmax(self.vo, dim=2) if self.classif else self.vo
        return yo

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.n_in} -> {self.n_rec} -> {self.n_out})"
           
class EP_SRNNAdapter(nn.Module):
    """
    适配 e-prop 的 SRNN 到现在架构：
    - inputs: X [B, T, D]
    - 暴露 W_in/W_rec/W_out/b_out 供使用
    """
    def __init__(self, n_in, n_rec, n_out, n_t,
                 thr=1.0, tau_m=20.0, tau_o=20.0, b_o=0.0, gamma=0.3, dt=1.0,
                 classif=True, w_init_gain=(1.0,1.0,1.0),
                 device=torch.device("cpu")):
        super().__init__()
        
        self.core = SRNN(
            n_in=n_in, n_rec=n_rec, n_out=n_out, n_t=n_t,
            thr=thr, tau_m=tau_m, tau_o=tau_o, b_o=b_o,
            gamma=gamma, dt=dt,
            classif=classif,
            w_init_gain=w_init_gain,
            device=device,
        )
        self.n_in, self.n_rec, self.n_out, self.n_t = n_in, n_rec, n_out, n_t
        self.device = device

        self.W_in  = self.core.w_in
        self.W_rec = self.core.w_rec
        self.W_out = self.core.w_out
        # self.b_out = nn.Parameter(torch.full((n_out,), float(b_o), device=device))


        self.register_buffer("_last_x_seq",   torch.zeros(1, n_t, n_in))
        self.register_buffer("_last_spk_seq", torch.zeros(1, n_t, n_rec))
        self.register_buffer("_last_mem_seq", torch.zeros(1, n_t, n_rec))
        self.register_buffer("_last_forward_macs", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_spike_rate_for_reg", torch.tensor(0.0))
        self.cached_pre_any  = None
        self.cached_post_any = None
        self.cached_delta_t  = None

    def _to_TBN(self, X_BTD):
  
        return X_BTD.transpose(0, 1)  # [T,B,D]

    def get_flatten_theta(self) -> torch.Tensor:
        """
        只包含 W_in 和 W_rec，用于零阶优化。
        返回: 1D tensor，放在对应 device 上。
        """
        return torch.cat([
            self.W_in.view(-1),
            self.W_rec.view(-1)
        ], dim=0)
    @torch.no_grad()
    def set_theta_from_flatten(self, theta: torch.Tensor):
        """
        theta: 1D tensor，长度 = W_in.numel + W_rec.numel
        """
        n_in = self.W_in.numel()
        n_rec = self.W_rec.numel()
        assert theta.numel() == n_in + n_rec

        theta_in  = theta[:n_in].view_as(self.W_in)
        theta_rec = theta[n_in:].view_as(self.W_rec)

        self.W_in.data.copy_(theta_in)
        self.W_rec.data.copy_(theta_rec)
    
    def forward(self, X_BTD: torch.Tensor, return_spk: bool = False):

        T = self.n_t
        assert X_BTD.shape[1] == T, f"X T={X_BTD.shape[1]} 与模型 n_t={T} 不一致"
        x_TBD = self._to_TBN(X_BTD).contiguous()
        yo = self.core(x_TBD)


        # 读出：默认取时间维平均
        # logits_BC = yo.mean(dim=0)
        K = min(10, T)
        logits_BC = yo[-K:].mean(dim=0)


        # 缓存时序
        self._last_x_seq = X_BTD.detach()
        self._last_spk_seq = self.core.z.detach().transpose(0, 1) 
        self._last_mem_seq = self.core.v.detach().transpose(0, 1)  
        self._spike_rate_for_reg = self._last_spk_seq.float().mean()
        #缓存发放与否和First-spike
        with torch.no_grad():
            batch_size = X_BTD.size(0)  
            seq_length = X_BTD.size(1)  #=self.n_t
            input_dim = X_BTD.size(2)   #=self.n_in
            hidden_dim = self._last_spk_seq.size(2) #=self.n_rec

            input_spike_sequence = (X_BTD > 0).to(X_BTD.dtype)  # [B, T, D] 
            spike_sequence       = self._last_spk_seq.to(X_BTD.dtype)  # [B, T, H]

            #是否发放
            pre_any_per_neuron  = (input_spike_sequence.sum(dim=1) > 0).float()   # [B, D]
            post_any_per_neuron = (spike_sequence.sum(dim=1) > 0).float()         # [B, H]

            #first spk
            pre_time_mask  = (input_spike_sequence.sum(dim=2) > 0)   # [B, T]
            post_time_mask = (spike_sequence.sum(dim=2) > 0)         

            time_idx = torch.arange(seq_length, device=X_BTD.device).unsqueeze(0).expand(batch_size, -1)  # [B, T]
            big = torch.full_like(time_idx, fill_value=seq_length)  # 用 seq_length 代表未发放

            t_pre_scalar  = torch.where(pre_time_mask,  time_idx, big).min(dim=1).values.float()   # [B]
            t_post_scalar = torch.where(post_time_mask, time_idx, big).min(dim=1).values.float()   
            delta_t_scalar = (t_post_scalar - t_pre_scalar).abs()   

            #掩码没发放
            valid_pre  = pre_time_mask.any(dim=1)   
            valid_post = post_time_mask.any(dim=1)  
            valid_pair = valid_pre & valid_post     

            inf_like = torch.full_like(delta_t_scalar, float('inf'))
            delta_t_scalar = torch.where(valid_pair, delta_t_scalar, inf_like)

            self.cached_pre_any  = pre_any_per_neuron.detach()   # [B, D]
            self.cached_post_any = post_any_per_neuron.detach()  # [B, H]
            self.cached_delta_t  = delta_t_scalar.detach()

        if return_spk:
            return logits_BC, self._last_spk_seq
        return logits_BC


# ---------------------------------------------------------------------
#检测
class SynchronyMoniter(nn.Module):
    """
    计算突触对 (post_j <- pre_i) 的同步分数：
    mode="all","first","hybrid"
    """
    def __init__(self,
                 sigma: float = 2.0,
                 window: int = 5,
                 kernel: str = "gaussian",
                 norm: str = "min",
                 mode: str = "all",
                 alpha: float = 0.8,
                 local_sigma: Optional[float] = None,
                 local_window: Optional[int] = None,
                 local_kernel: str = "gaussian"):
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

        # hybrid 
        self.alpha = float(alpha) #首发占权重
        self.local_kernel = local_kernel
        self.local_sigma = float(local_sigma) if local_sigma is not None else max(1.0, self.sigma / 2.0)
        self.local_window = int(local_window) if local_window is not None else max(1, self.window // 2)

        self._streak = 0
 
        self.hist_len=200
        self.sync_hist=deque(maxlen=self.hist_len)
        self.ema_beta=0.9
        self.sync_ema=0.0
        self.cooldown=0            # 冷却时间
        self._cool=0
        self.need_streak=1          #连续多少次才触发

        # Nsync的统计缓存
        self.win_len_pairs = 40
        self.strong_tau = 0.08             #判断强同步的阈值
        self.strong_hist = deque(maxlen=self.win_len_pairs)
        self.nsync_thresh = 50           # 窗内强同步对数阈（可变）

        #局部更新
        self.sub_pct = 0.80                 # 选前20%参与度的 post/pre
        self.min_sub = 3                    # 至少这么多个神经元才触发局部SPSA



    def _build_kernel(self, T: int, device, *, kernel: str, sigma: float, window: int):
        if kernel == "gaussian":
            half = max(1, int(4 * sigma))                     # 截断高斯 ±4σ
            t = torch.arange(-half, half + 1, device=device).float()
            k = torch.exp(-(t ** 2) / (2 * (sigma ** 2)))
            k = k / k.sum().clamp_min(1e-8)
        else:  # 'box'
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
        return t_first  # [B, C]

    def _score_first(self, pre_spk, post_spk, *, sigma: float, norm: str):

        B, T, C_in = pre_spk.shape
        _, _, C_out = post_spk.shape

        pre_first = self._first_times(pre_spk)    # [B,C_in]
        post_first = self._first_times(post_spk)  # [B,C_out]

        valid_pre = (pre_first < T)
        valid_post = (post_first < T)

        dt = (post_first.unsqueeze(2) - pre_first.unsqueeze(1)).abs() 
        dt = torch.where(valid_post.unsqueeze(2) & valid_pre.unsqueeze(1),
                         dt, torch.full_like(dt, float("inf")))

        S = torch.exp(-(dt ** 2) / (2 * sigma ** 2))             

        if norm != "none":
            pre_cnt = pre_spk.sum(1).clamp_min(1e-6)             
            post_cnt = post_spk.sum(1).clamp_min(1e-6)               
            if norm == "min":
                z = torch.minimum(pre_cnt.unsqueeze(1).expand(-1, C_out, -1),
                                   post_cnt.unsqueeze(2).expand(-1, -1, C_in))
            else:
                z = torch.maximum(pre_cnt.unsqueeze(1).expand(-1, C_out, -1),
                                   post_cnt.unsqueeze(2).expand(-1, -1, C_in))
            S = S / z
        return S  # [B,C_out,C_in]

    def _score_all(self, pre_spk, post_spk, *, kernel: str, sigma: float, window: int, norm: str):
      
        B, T, C_in = pre_spk.shape
        _, _, C_out = post_spk.shape
        k = self._build_kernel(T, pre_spk.device, kernel=kernel, sigma=sigma, window=window)
        pre = pre_spk.transpose(1, 2)  
        pre_f = F.conv1d(pre, k.expand(C_in, 1, -1), padding=k.size(-1) // 2, groups=C_in).transpose(1, 2)  
        post = post_spk.transpose(1, 2)  
        S = torch.einsum("bjt,bti->bji", post, pre_f)  # [B,C_out,C_in]

        if norm != "none":
            pre_count = pre_spk.sum(1).clamp_min(1e-6)   # [B,C_in]
            post_count = post_spk.sum(1).clamp_min(1e-6) # [B,C_out]
            if norm == "min":
                z = torch.minimum(pre_count.unsqueeze(1).expand(-1, C_out, -1),
                                   post_count.unsqueeze(2).expand(-1, -1, C_in))
            else:
                z = torch.maximum(pre_count.unsqueeze(1).expand(-1, C_out, -1),
                                   post_count.unsqueeze(2).expand(-1, -1, C_in))
            S = S / z

        # 屏蔽“双方都没发”
        valid = (pre_spk.sum(1) > 0).unsqueeze(1) & (post_spk.sum(1) > 0).unsqueeze(2)
        S = torch.where(valid, S, torch.zeros_like(S))
        return S  # [B,C_out,C_in] 

    @torch.no_grad()
    def forward(self, pre_spk, post_spk):
        """
        pre_spk : [B,T,C_in]  (0/1)
        post_spk: [B,T,C_out] (0/1)
        return:
          S:   [B,C_out,C_in]  同步分数矩阵
          agg: dict  {layer_sync:[B], post_particip:[B,C_out], pre_particip:[B,C_in]}
        """
        if self.mode == "first":
            S = self._score_first(pre_spk, post_spk, sigma=self.sigma, norm=self.norm)

        elif self.mode == "all":
            S = self._score_all(pre_spk, post_spk,
                                kernel=self.kernel, sigma=self.sigma, window=self.window, norm=self.norm)

        else:  # "hybrid"
 
            S_first = self._score_first(pre_spk, post_spk, sigma=self.sigma, norm=self.norm)

   
            B, T, C_in = pre_spk.shape
            _, _, C_out = post_spk.shape
            pre_first = self._first_times(pre_spk)    # [B,C_in]
            post_first = self._first_times(post_spk)  # [B,C_out]

     
            t_idx = torch.arange(T, device=pre_spk.device).view(1, T, 1)  
            # pre mask
            pre_valid = (pre_first < T)  # [B,C_in]
            pre_low  = (pre_first - self.local_window).clamp_min(0).floor().long()    
            pre_high = (pre_first + self.local_window).clamp_max(T - 1).ceil().long() 
            pre_mask = (t_idx >= pre_low.unsqueeze(1)) & (t_idx <= pre_high.unsqueeze(1)) 
            pre_mask = pre_mask & pre_valid.unsqueeze(1)                               

            # post mask
            post_valid = (post_first < T) # [B,C_out]
            post_low  = (post_first - self.local_window).clamp_min(0).floor().long()
            post_high = (post_first + self.local_window).clamp_max(T - 1).ceil().long()
            post_mask = (t_idx >= post_low.unsqueeze(1)) & (t_idx <= post_high.unsqueeze(1))  
            post_mask = post_mask & post_valid.unsqueeze(1)

            # 局部截取（与原脉冲相与）
            pre_local  = pre_spk * pre_mask.float()
            post_local = post_spk * post_mask.float()

            S_local = self._score_all(pre_local, post_local,
                                      kernel=self.local_kernel,
                                      sigma=self.local_sigma,
                                      window=self.local_window,
                                      norm=self.norm)

            
            S = self.alpha * S_first + (1.0 - self.alpha) * S_local

        # 聚合指标
        layer_sync = S.mean(dim=(1, 2))
        post_particip = S.mean(dim=2)
        pre_particip = S.mean(dim=1)
        agg = {
            "layer_sync": layer_sync,   
            "post_particip": post_particip, 
            "pre_particip": pre_particip  
        }
        return S, agg

    @torch.no_grad()
    def update_and_check(self,pre_spk,post_spk):
        """
        调用 forward 得到 S, agg 后，再调用本函数完成：
        对真正同步事件的检查
        1、层级同步显著高于近期水平(滤波)：layer_sync=mean(S),sync_ema(t)=beta*sync_ema(t-1)+(1-beta)*layer_sync
        2、层级同步达到峰值门：layer_sync > peak_tau (peak_tau 自适应)且连续发生3次才将峰值视为持续的强同步
        3、定义强同步对：S_ij > strong_tau (0.2)，统计窗口内强同步对数并求和
        4、给系统冷却时间cooldown，避免过于频繁触发
        5、局部更新：pre、post各自取top-k挑中的子块权重

        """
        S, agg = self(pre_spk, post_spk)     # 复用你已有的分数
        layer_sync = float(agg["layer_sync"].mean().item())
        self.sync_ema = self.ema_beta * self.sync_ema + (1. - self.ema_beta) * layer_sync
        self.sync_hist.append(layer_sync)
    

        # --- 强同步对统计（看第一个样本的 S 矩阵；也可用 batch 均值）---
        S0 = S[0]                            # [H,H]
        strong_cnt = int((S0 > self.strong_tau).sum().item())
        self.strong_hist.append(strong_cnt)
        win_sum = int(sum(self.strong_hist))

        # --- 自适应阈：用历史分位数做“峰值门” ---
        # 若历史还不够长，就退化用固定值 0.11
        import numpy as np
        if len(self.sync_hist) >= max(20, int(0.25*self.hist_len)):
            peak_tau = float(np.quantile(np.array(self.sync_hist), 0.9))  # P90
        else:
            peak_tau = 0.08

        is_peak = (layer_sync > peak_tau)
        self._streak = self._streak + 1 if is_peak else 0
        win_over = (win_sum >= self.nsync_thresh)
        # win_over = True
        # 冷却计数器
        if self._cool > 0:
            self._cool -= 1

        # --- 触发条件：峰值门(持续) AND 窗口门 AND 非冷却 ---
        trigger = (self._streak >= self.need_streak) and win_over and (self._cool == 0)
        # trigger = False
        # --- 为“局部-SPSA”挑子群：按参与度选前10% ---
        # post_particip: [B,C_out] 取第一个样本
        post_part = agg["post_particip"][0]         # [H]
        pre_part  = agg["pre_particip"][0]          # [H]
        H = post_part.numel()
        k_post = max(self.min_sub, int(np.ceil((1.0 - self.sub_pct) * H)))
        k_pre  = max(self.min_sub, int(np.ceil((1.0 - self.sub_pct) * H)))
        _, idx_post = torch.topk(post_part, k=k_post)
        _, idx_pre  = torch.topk(pre_part,  k=k_pre)


        sub = {
            "post_idx": idx_post.detach().cpu().tolist(),
            "pre_idx":  idx_pre.detach().cpu().tolist(),
        }

        if trigger:
            self._cool = self.cooldown      
            self._streak = 0           
             
        strong_ratio = strong_cnt / float(S0.numel())

        info = {
            "S": S, "agg": agg,
            "layer_sync": layer_sync, "sync_ema": self.sync_ema,
            "peak_tau": peak_tau, "is_peak": bool(is_peak),
            "streak": int(self._streak), "win_sum": win_sum,
            "trigger": bool(trigger), "strong_cnt": strong_cnt,
            "sub_idx": sub,"strong_ratio": float(strong_ratio),
        }
        return info
       

# ---------------------------------------------------------------------
# 评估
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, batch: int = 1024) -> float:
    ce = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss_sum = 0, 0, 0.0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        logits = model.forward(X)
        loss_sum += ce(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += y.numel()
    acc = 100.0*correct/total
    return acc

@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    ce = nn.CrossEntropyLoss(reduction="sum")
    total, loss_sum = 0, 0.0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        logits = model.forward(X)
        loss_sum += ce(logits, y).item()
        total += y.numel()
    return loss_sum / max(total, 1)

@torch.no_grad()
def estimate_forward_flops(model: nn.Module, x_batch: torch.Tensor) -> Tuple[float, float]:

    params = sum(p.numel() for p in model.parameters())
    if not _HAVE_THOP:
        return float("nan"), float(params)

    # thop 只测 forward
    macs, _ = profile(model, inputs=(x_batch,), verbose=False)
    flops = 2.0 * float(macs)
    return flops, float(params)

def estimate_snn_flops(T, H, D, O):
    step = 2 * (H*H + H*D + O*H)
    return step * T


#---------------------------------------------------------------------
# main

@dataclass
class MainCfg:
    npz_path: str = "nmini_0v1_T50_singlech.npz"
    hidden_dim: int = 512
    batch_size: int = 64
    tau_m: float = 4.0
    out_tau: float = 5.0
    steps: int = 300             
    spsa_a: float = 1e-2          # a_k（相当于学习率）
    spsa_c: float = 1e-2          # c_k（扰动幅度）
    spsa_grad_clip: Optional[float] = None
    spsa_clip_norm: float = 0.1
    zo_epsilon: float = 1e-2       # 扰动幅度 ε
    zo_lr: float = 1e-2            # 零阶更新学习率 a_k
    zo_queries: int = 4            # 每个 batch 采样几次方向（平均一下）
    wrec_param_clip: float = 4.5
    lr_out: float = 5e-3
    wd: float = 1e-4
    epochs: int = 100
    topk_pairs: int = 15
    enable_plots: bool = True  

    #用传递
    spsa_direction: str = "spsa"      # "spsa" 或 "gaussian"
    spsa_normalize_dir: bool = True  # True 时对方向做 L2 归一化

def main():
    set_seed(310)
    device = auto_select_device("auto")

    dl_tr, dl_va, dl_te, meta = make_dataloaders_from_npz(
        MainCfg.npz_path,
        batch_size=MainCfg.batch_size,
        device=device
    )
    D, C = meta["D"], meta["num_classes"]
    T = meta["T"]

    print(f"[DATA] N_tr={meta['N_tr']}, batch_size={MainCfg.batch_size}, "
          f"len(dl_tr)={len(dl_tr)}, epochs={MainCfg.epochs}")

    # --------- 构建模型 ---------
    model = EP_SRNNAdapter(
        n_in=D, n_rec=MainCfg.hidden_dim, n_out=C, n_t=T,
        thr=0.35, tau_m=MainCfg.tau_m, tau_o=MainCfg.out_tau, b_o=0.0,
        gamma=0.3, dt=1.0, classif=False, w_init_gain=(1.0, 1.0, 1.0),
        device=device
    ).to(device)

    # 关闭 autograd，全部手写更新
    for p in model.parameters():
        p.requires_grad = False

    ce = nn.CrossEntropyLoss()
    sync_mon = SynchronyMoniter(
        mode="hybrid",
        sigma=3.0,
        window=5,
        norm="min",
        alpha=0.85,
        local_sigma=1.5,
        local_window=5,
        local_kernel="gaussian"
    ).to(device)
    trig_idx = 0
    global_step = 0
    train_loss_hist = []
    val_loss_hist   = []
    val_acc_hist    = []

    # 触发点在 loss 曲线上的坐标（按 epoch 标）
    trig_x_train, trig_y_train = [], []
    trig_x_val,   trig_y_val   = [], []
    for ep in range(1, MainCfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_cnt  = 0

        epoch_triggered = False

        for X, y in dl_tr:
            global_step += 1
            X = X.to(device)
            y = y.to(device)
            B = X.size(0)

            # ==========================
            # 1) 显式更新 W_out（近似 SGD）
            # ==========================
            with torch.no_grad():
                logits, spk_seq = model(X, return_spk=True)   # logits: [B,C]
                loss = ce(logits, y)
                train_loss = float(loss.item())

                # 统计
                running_loss += train_loss * B
                running_cnt  += B

                num_classes = logits.size(1)
                hidden_dim  = spk_seq.size(2)

                probs    = F.softmax(logits, dim=1)
                y_onehot = F.one_hot(y, num_classes=num_classes).float()
                grad_logits = (probs - y_onehot) / B

                # 读出层最近 K 步的平均活动
                K_spk = min(10, spk_seq.size(1))
                h = spk_seq[:, -K_spk:, :].float().mean(dim=1)    # [B,H]

                # dL/dW_out ≈ grad_logits^T @ h
                dW_out = grad_logits.t() @ h                     # [C,H]
                dW_out = dW_out + MainCfg.wd * model.W_out.data

                # SGD step for W_out
                model.W_out.data -= MainCfg.lr_out * dW_out

            # ==========================
            # 2) 零阶更新 W_in + W_rec
            # ==========================
            with torch.no_grad():
                # 当前 theta
                theta0 = model.get_flatten_theta().clone()
                g_hat = torch.zeros_like(theta0)

                for q in range(MainCfg.zo_queries):
                    # 采样扰动方向 u
                    u = torch.randn_like(theta0)
                    # 可选：归一化，避免尺度过大
                    u = u / (u.norm() + 1e-8)

                    # θ+ = θ0 + εu
                    model.set_theta_from_flatten(theta0 + MainCfg.zo_epsilon * u)
                    loss_pos = ce_loss_no_grad(model, X, y)

                    # θ- = θ0 - εu
                    model.set_theta_from_flatten(theta0 - MainCfg.zo_epsilon * u)
                    loss_neg = ce_loss_no_grad(model, X, y)

                    # SPSA / ZO 梯度估计： (L+ - L-) / (2ε) * u
                    g_hat += (loss_pos - loss_neg) / (2.0 * MainCfg.zo_epsilon) * u

                g_hat /= float(MainCfg.zo_queries)

                # SGD-like 更新：θ ← θ - η * g_hat
                theta_new = theta0 - MainCfg.zo_lr * g_hat
                model.set_theta_from_flatten(theta_new)
            # with torch.no_grad():
            #     # 用 spk_seq 做同步检测
            #     if spk_seq.size(1) >= 2:
            #         pre_spk  = spk_seq[:, :-1, :]  # [B,T-1,H]
            #         post_spk = spk_seq[:,  1:, :]  # [B,T-1,H]
            #         mon = sync_mon.update_and_check(pre_spk, post_spk)
            #     else:
            #         mon = {"trigger": False}

            #     if USE_SPSA and mon["trigger"]:
            #         trig_idx += 1
            #         epoch_triggered = True 
            #         print(f"[TRIGGER] step={global_step} ep={ep} "
            #             f"layer_sync={mon['layer_sync']:.4f} "
            #             f"peak_tau={mon['peak_tau']:.4f} "
            #             f"win_sum={mon['win_sum']} "
            #             f"sub|post={len(mon['sub_idx']['post_idx'])} "
            #             f"pre={len(mon['sub_idx']['pre_idx'])}")

            #         post_idx = mon["sub_idx"]["post_idx"]  # list[int]
            #         pre_idx  = mon["sub_idx"]["pre_idx"]

            #         # ---- 构造 W_rec 的局部 mask: 只更新选中的 post x pre ----
            #         mask_rec = torch.zeros_like(model.W_rec, dtype=torch.bool)
            #         post_idx_t = torch.as_tensor(post_idx, device=mask_rec.device, dtype=torch.long)
            #         pre_idx_t  = torch.as_tensor(pre_idx,  device=mask_rec.device, dtype=torch.long)
            #         mask_rec[post_idx_t[:, None], pre_idx_t[None, :]] = True
            #         mask_rec &= model.core.rec_mask.bool()

            #         # ---- 构造 W_in 的局部 mask: post 同一组，pre 挑激活度最高的一部分输入维度 ----
            #         X_BTD = model._last_x_seq.to(model.W_in.device)  # [B,T,D]
            #         pre_part_in = (X_BTD > 0).float().sum(dim=(0, 1))  # [D]
            #         D_in = pre_part_in.numel()
            #         k_pre_in = max(sync_mon.min_sub, int(np.ceil((1.0 - sync_mon.sub_pct) * D_in)))
            #         _, idx_pre_in = torch.topk(pre_part_in, k=k_pre_in)  # [k_pre_in]

            #         mask_in = torch.zeros_like(model.W_in, dtype=torch.bool)
            #         mask_in[post_idx_t[:, None], idx_pre_in[None, :]] = True

            #         # ---- 设置 a_k, c_k（这里简单用常数）----
            #         a_k = MainCfg.spsa_a
            #         c_k = MainCfg.spsa_c

            #         # ---- 更新 W_rec ----
            #         mon_spsa_rec = spsa_update_Wrec(
            #             model, X, y, ce_loss=ce_loss_no_grad,
            #             a_k=a_k, c_k=c_k,
            #             grad_clip=MainCfg.spsa_grad_clip,
            #             clip_norm=MainCfg.spsa_clip_norm,
            #             auto_gain=False,
            #             target_update_norm=None,
            #             gain_clip=None,
            #             mask=mask_rec,
            #             direction=MainCfg.spsa_direction,
            #             normalize_dir=MainCfg.spsa_normalize_dir,
            #         )

            #         # ---- 更新 W_in ----
            #         mon_spsa_in = spsa_update_Win(
            #             model, X, y, ce_loss=ce_loss_no_grad,
            #             a_k=a_k, c_k=c_k,
            #             grad_clip=MainCfg.spsa_grad_clip,
            #             clip_norm=MainCfg.spsa_clip_norm,
            #             auto_gain=False,
            #             target_update_norm=None,
            #             gain_clip=None,
            #             mask=mask_in,
            #             direction=MainCfg.spsa_direction,
            #             normalize_dir=MainCfg.spsa_normalize_dir,
            #         )           

        # ------- epoch 结束：打印 loss & 验证 -------
        epoch_loss = running_loss / max(running_cnt, 1)
        model.eval()
        val_acc  = evaluate(model, dl_va, device)
        val_loss = evaluate_loss(model, dl_va, device)

        # 记录历史
        train_loss_hist.append(epoch_loss)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # 如果本 epoch 发生过 trigger，就记录在曲线上的坐标
        if epoch_triggered:
            trig_x_train.append(ep)
            trig_y_train.append(epoch_loss)
            trig_x_val.append(ep)
            trig_y_val.append(val_loss)

        print(f"Epoch {ep}: Train Loss = {epoch_loss:.4f}, "
              f"Val Acc = {val_acc:.2f}%, Val Loss = {val_loss:.4f}")

        # ------- 画图 -------
        if MainCfg.enable_plots:
            epochs_axis = range(1, ep + 1)

            # 1) loss 曲线 + trigger 点
            plt.figure()
            plt.plot(epochs_axis, train_loss_hist, label="Train Loss")
            plt.plot(epochs_axis, val_loss_hist,   label="Val Loss")

            # 画出 trigger 发生时对应的点（在 loss 曲线上的位置）
            if len(trig_x_train) > 0:
                plt.scatter(trig_x_train, trig_y_train,
                            marker="o", s=35, label="Trigger (Train)")
                plt.scatter(trig_x_val, trig_y_val,
                            marker="x", s=35, label="Trigger (Val)")

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            savefig_safe(FIG_DIR / "loss_curve_with_trigger.png")

    # -------- 训练结束，评估 --------
    acc_tr = evaluate(model, dl_tr, device)
    acc_va = evaluate(model, dl_va, device)
    acc_te = evaluate(model, dl_te, device)
    print(f"ZO-SRNN | Train: {acc_tr:.2f}% | Val: {acc_va:.2f}% | Test: {acc_te:.2f}%")

if __name__ == "__main__":
    main()
