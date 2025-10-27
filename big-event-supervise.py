# rsnn_with_spsa.py
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
from spsa import SPSAConfig, spsa_update_Wrec, ce_loss_no_grad
USE_SPSA = False 
try:
    from thop import profile
    _HAVE_THOP = True
except Exception:
    profile = None
    _HAVE_THOP = False

def get_out_dirs():
    try:
        base = Path(__file__).parent
    except NameError:
        base = Path.cwd()
    run_dir = (base / "runs").resolve()
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
    
    def __init__(self, n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, gamma, dt, model, classif, w_init_gain, lr_layer, t_crop, visualize, visualize_light, device):    
        
        super(SRNN, self).__init__()
        self.n_in     = n_in
        self.n_rec    = n_rec
        self.n_out    = n_out
        self.n_t      = n_t
        self.thr      = thr
        self.dt       = dt
        self.alpha    = np.exp(-dt/tau_m)
        self.kappa    = np.exp(-dt/tau_o)
        self.gamma    = gamma
        self.b_o      = b_o
        self.model    = model
        self.classif  = classif
        self.lr_layer = lr_layer
        self.t_crop   = t_crop  
        self.visu     = visualize
        self.visu_l   = visualize_light
        self.device   = device
        
        #Parameters
        self.w_in  = nn.Parameter(torch.Tensor(n_rec, n_in ))
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))
        self.reg_term = torch.zeros(self.n_rec).to(self.device)
        self.B_out = torch.Tensor(n_out, n_rec).to(self.device)
        self.reset_parameters(w_init_gain)
        self.register_buffer("rec_mask", (torch.ones(n_rec, n_rec, device=device) 
                                                  - torch.eye(n_rec, n_rec, device=device)))
        
        #Visualization
        if self.visu:
            plt.ion()
            self.fig, self.ax_list = plt.subplots(2+self.n_out+5, sharex=True)

    def reset_parameters(self, gain):
        
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0]*self.w_in.data
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1]*self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2]*self.w_out.data
        
    def init_net(self, n_b, n_t, n_in, n_rec, n_out):
        
        #Hidden state
        self.v  = torch.zeros(n_t,n_b,n_rec).to(self.device)
        self.vo = torch.zeros(n_t,n_b,n_out).to(self.device)
        #Visible state
        self.z  = torch.zeros(n_t,n_b,n_rec).to(self.device)
        #Weight gradients
        self.w_in.grad  = torch.zeros_like(self.w_in)
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)
    
    def forward(self, x, yt, do_training):
        
        self.n_b = x.shape[1]    # Extracting batch size
        self.init_net(self.n_b, self.n_t, self.n_in, self.n_rec, self.n_out)    # Network reset
         
        # self.w_rec *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))         # Making sure recurrent self excitation/inhibition is cancelled
        w_rec_eff = self.w_rec * self.rec_mask
        for t in range(self.n_t-1):     # Computing the network state and outputs for the whole sample duration
        
            # Forward pass - Hidden state:  v: recurrent layer membrane potential
            #                Visible state: z: recurrent layer spike output, vo: output layer membrane potential (yo incl. activation function)
            # 
            self.v[t+1]  = (self.alpha * self.v[t] + torch.mm(self.z[t], w_rec_eff.t()) + torch.mm(x[t], self.w_in.t())) - self.z[t]*self.thr
            self.z[t+1]  = (self.v[t+1] > self.thr).float()
            self.vo[t+1] = self.kappa * self.vo[t] + torch.mm(self.z[t+1], self.w_out.t()) + self.b_o

        if self.classif:        #Apply a softmax function for classification problems based on the task
            yo = F.softmax(self.vo,dim=2)
        else:
            yo = self.vo

        if do_training:
            self.grads_batch(x, yo, yt)
            
        return yo
    
    def grads_batch(self, x, yo, yt):
        
        # Surrogate derivatives
        h = self.gamma*torch.max(torch.zeros_like(self.v), 1-torch.abs((self.v-self.thr)/self.thr))
   
        # Input and recurrent eligibility vectors for the 'LIF' model (vectorized computation, model-dependent)
        assert self.model == "LIF", "Nice try, but model " + self.model + " is not supported. ;-)"
        alpha_conv  = torch.tensor([self.alpha ** (self.n_t-i-1) for i in range(self.n_t)]).float().view(1,1,-1).to(self.device)
        trace_in    = F.conv1d(x.permute(1,2,0), alpha_conv.expand(self.n_in ,-1,-1), padding=self.n_t, groups=self.n_in )[:,:,1:self.n_t+1].unsqueeze(1).expand(-1,self.n_rec,-1,-1)  #n_b, n_rec, n_in , n_t 
        trace_in    = torch.einsum('tbr,brit->brit', h, trace_in )                                                                                                                          #n_b, n_rec, n_in , n_t 
        trace_rec   = F.conv1d(self.z.permute(1,2,0), alpha_conv.expand(self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec)[:,:, :self.n_t  ].unsqueeze(1).expand(-1,self.n_rec,-1,-1)  #n_b, n_rec, n_rec, n_t
        trace_rec   = torch.einsum('tbr,brit->brit', h, trace_rec)                                                                                                                          #n_b, n_rec, n_rec, n_t    
        trace_reg   = trace_rec

        # Output eligibility vector (vectorized computation, model-dependent)
        kappa_conv = torch.tensor([self.kappa ** (self.n_t-i-1) for i in range(self.n_t)]).float().view(1,1,-1).to(self.device)
        trace_out  = F.conv1d(self.z.permute(1,2,0), kappa_conv.expand(self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec)[:,:,1:self.n_t+1]  #n_b, n_rec, n_t

        # Eligibility traces
        trace_in     = F.conv1d(   trace_in.reshape(self.n_b,self.n_in *self.n_rec,self.n_t), kappa_conv.expand(self.n_in *self.n_rec,-1,-1), padding=self.n_t, groups=self.n_in *self.n_rec)[:,:,1:self.n_t+1].reshape(self.n_b,self.n_rec,self.n_in ,self.n_t)   #n_b, n_rec, n_in , n_t  
        trace_rec    = F.conv1d(  trace_rec.reshape(self.n_b,self.n_rec*self.n_rec,self.n_t), kappa_conv.expand(self.n_rec*self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec*self.n_rec)[:,:,1:self.n_t+1].reshape(self.n_b,self.n_rec,self.n_rec,self.n_t)   #n_b, n_rec, n_rec, n_t
        
        # Learning signals
        err = yo - yt
        L = torch.einsum('tbo,or->brt', err, self.w_out)
        
        # Update network visualization
        if self.visu:
            self.update_plot(x, self.z, yo, yt, L, trace_reg, trace_in, trace_rec, trace_out)
        
        # Compute network updates taking only the timesteps where the target is present
        if self.t_crop != 0:
            L         =          L[:,:,-self.t_crop:]
            err       =        err[-self.t_crop:,:,:]
            trace_in  =   trace_in[:,:,:,-self.t_crop:]
            trace_rec =  trace_rec[:,:,:,-self.t_crop:]
            trace_out =  trace_out[:,:,-self.t_crop:]
        
        # Weight gradient updates
        self.w_in.grad  += self.lr_layer[0]*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_in ,-1) * trace_in , dim=(0,3)) 
        self.w_rec.grad += self.lr_layer[1]*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_rec,-1) * trace_rec, dim=(0,3))
        self.w_out.grad += self.lr_layer[2]*torch.einsum('tbo,brt->or', err, trace_out)
        if self.w_rec.grad is not None:
            self.w_rec.grad *= self.rec_mask

    def update_plot(self, x, z, yo, yt, L, trace_reg, trace_in, trace_rec, trace_out):
        """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""
    
        # Clear the axis to print new plots
        for k in range(self.ax_list.shape[0]):
            ax = self.ax_list[k]
            ax.clear()
    
        # Plot input signals
        for k, spike_ref in enumerate(zip(['In spikes','Rec spikes'],[x,z])):
            spikes = spike_ref[1][:,0,:].cpu().numpy()
            ax = self.ax_list[k]
    
            ax.imshow(spikes.T, aspect='auto', cmap='hot_r', interpolation="none")
            ax.set_xlim([0, self.n_t])
            ax.set_ylabel(spike_ref[0])
    
        for i in range(self.n_out):
            ax = self.ax_list[i + 2]
            if self.classif:
                ax.set_ylim([-0.05, 1.05])
            ax.set_ylabel('Output '+str(i))
    
            ax.plot(np.arange(self.n_t), yo[:,0,i].cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            if self.t_crop != 0:
                ax.plot(np.arange(self.n_t)[-self.t_crop:], yt[-self.t_crop:,0,i].cpu().numpy(), linestyle='solid', label='Target', alpha=0.8)
            else:
                ax.plot(np.arange(self.n_t), yt[:,0,i].cpu().numpy(), linestyle='solid' , label='Target', alpha=0.8)
    
            ax.set_xlim([0, self.n_t])
    
        for i in range(5):
            ax = self.ax_list[i + 2 + self.n_out]
            ax.set_ylabel("Trace reg" if i==0 else "Traces out" if i==1 else "Traces rec" if i==2 else "Traces in" if i==3 else "Learning sigs")
            
            if i==0:
                if self.visu_l:
                    ax.plot(np.arange(self.n_t), trace_reg[0,:,0,:].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
                else:
                    ax.plot(np.arange(self.n_t), trace_reg[0,:,:,:].reshape(self.n_rec*self.n_rec,self.n_t).T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            elif i<4:
                if self.visu_l:
                    ax.plot(np.arange(self.n_t), trace_out[0,:,:].T.cpu().numpy() if i==1 \
                                            else trace_rec[0,:,0,:].T.cpu().numpy() if i==2 \
                                            else trace_in[0,:,0,:].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
                else:
                    ax.plot(np.arange(self.n_t), trace_out[0,:,:].T.cpu().numpy() if i==1 \
                                        else trace_rec[0,:,:,:].reshape(self.n_rec*self.n_rec,self.n_t).T.cpu().numpy() if i==2 \
                                        else trace_in[0,:,:,:].reshape(self.n_rec*self.n_in,self.n_t).T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            elif self.t_crop != 0:
                ax.plot(np.arange(self.n_t)[-self.t_crop:], L[0,:,-self.t_crop:].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            else:
                ax.plot(np.arange(self.n_t), L[0,:,:].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
        
        ax.set_xlim([0, self.n_t])
        ax.set_xlabel('Time in ms')
        
        # Short wait time to draw with interactive python
        plt.draw()
        plt.pause(0.1)
        
        
    def __repr__(self):
        
        return self.__class__.__name__ + ' (' \
            + str(self.n_in) + ' -> ' \
            + str(self.n_rec) + ' -> ' \
            + str(self.n_out) + ') '
           
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
        
        self.core = SRNN(n_in=n_in, n_rec=n_rec, n_out=n_out, n_t=n_t,
                         thr=thr, tau_m=tau_m, tau_o=tau_o, b_o=b_o,
                         gamma=gamma, dt=dt, model="LIF", classif=classif,
                         w_init_gain=w_init_gain, lr_layer=(1.,1.,1.),
                         t_crop=0, visualize=False, visualize_light=True,
                         device=device)
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

    @torch.no_grad()
    def _make_time_targets(self, y_B, T, C):
        # e-prop 要求逐时刻监督信号
        B = y_B.numel()
        yt = torch.zeros(T, B, C, device=y_B.device)
        yt[:, torch.arange(B), y_B] = 1.0
        return yt

    @torch.no_grad()
    def forward(self, X_BTD: torch.Tensor, return_spk: bool = False):

        T = self.n_t
        assert X_BTD.shape[1] == T, f"X T={X_BTD.shape[1]} 与模型 n_t={T} 不一致"
        x_TBD = self._to_TBN(X_BTD).contiguous()

        # yt 随便给个占位但 do_training=False
        dummy_yt = torch.zeros(T, X_BTD.size(0), self.n_out, device=X_BTD.device)
        yo = self.core(x_TBD, dummy_yt, do_training=False) 

        # 读出：默认取时间维平均
        logits_BC = yo.mean(dim=0)

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

    def eprop_train_step(self, X_BTD: torch.Tensor, y_B: torch.Tensor):

        T = self.n_t; C = self.n_out
        x_TBD = self._to_TBN(X_BTD)
        yt_TBC = self._make_time_targets(y_B, T, C)
        _ = self.core(x_TBD, yt_TBC, do_training=True)  
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

        # 缓存 kernel 原型
        self.register_buffer("_kernel", torch.empty(0))

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

# ---------------------------------------------------------------------
# 评估
@torch.no_grad()
def hybrid_decompose(sync_mon: SynchronyMoniter, pre_spk, post_spk):
    # S_first
    S_first = sync_mon._score_first(pre_spk, post_spk,
                                    sigma=sync_mon.sigma, norm=sync_mon.norm)
    # 局部 mask（照 forward 里的实现）
    B,T,C_in = pre_spk.shape
    _,_,C_out= post_spk.shape
    pre_first  = sync_mon._first_times(pre_spk)
    post_first = sync_mon._first_times(post_spk)
    t_idx = torch.arange(T, device=pre_spk.device).view(1,T,1)
    pre_low  = (pre_first - sync_mon.local_window).clamp_min(0).floor().long()
    pre_high = (pre_first + sync_mon.local_window).clamp_max(T-1).ceil().long()
    post_low  = (post_first - sync_mon.local_window).clamp_min(0).floor().long()
    post_high = (post_first + sync_mon.local_window).clamp_max(T-1).ceil().long()
    pre_mask  = (t_idx >= pre_low.unsqueeze(1)) & (t_idx <= pre_high.unsqueeze(1))
    post_mask = (t_idx >= post_low.unsqueeze(1)) & (t_idx <= post_high.unsqueeze(1))
    pre_mask  = pre_mask & (pre_first<T).unsqueeze(1)
    post_mask = post_mask & (post_first<T).unsqueeze(1)

    pre_local  = pre_spk  * pre_mask.float()
    post_local = post_spk * post_mask.float()
    S_local = sync_mon._score_all(pre_local, post_local,
                                  kernel=sync_mon.local_kernel,
                                  sigma=sync_mon.local_sigma,
                                  window=sync_mon.local_window,
                                  norm=sync_mon.norm)
    S_h = sync_mon.alpha * S_first + (1.0 - sync_mon.alpha) * S_local
    # 返回 batch 平均的层均值
    return {
        "first": float(S_first.mean(dim=(1,2)).mean().item()),
        "local": float(S_local.mean(dim=(1,2)).mean().item()),
        "hybrid": float(S_h.mean(dim=(1,2)).mean().item()),
    }

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
#群体动力学
@torch.no_grad()
def _raster_coords_from_spk_seq(spk_seq_BTH: torch.Tensor, max_neurons: int = 128):
    spk = spk_seq_BTH[0].detach().cpu().numpy()  
    T, H = spk.shape
    import numpy as np
    H_sel = min(H, max_neurons)
    idx = np.arange(H_sel)
    spk_sel = spk[:, idx]                      
    t_idx, n_idx = np.nonzero(spk_sel)           
    return t_idx, n_idx, T, H_sel


def _plot_multi_epoch_rasters(raster_data: List[Tuple[int, Tuple[np.ndarray, np.ndarray, int, int]]],
                              save_path: str):

    ncols = len(raster_data)
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, (ep, (t_idx, n_idx, T, H_sel)) in zip(axes, raster_data):
        ax.scatter(t_idx, n_idx, s=3)
        ax.set_xlim(0, T)
        ax.set_ylim(-0.5, H_sel - 0.5)
        ax.set_xlabel("time")
        ax.set_ylabel(f"neuron (mod {H_sel})")

        if ax is axes[0]:
            ax.set_ylabel(f"neuron (mod {H_sel})")
    plt.tight_layout()
    savefig_safe(save_path)

@torch.no_grad()
def estimate_forward_flops(model: nn.Module, x_batch: torch.Tensor) -> Tuple[float, float]:

    params = sum(p.numel() for p in model.parameters())
    if not _HAVE_THOP:
        return float("nan"), float(params)

    # thop 只测 forward
    macs, _ = profile(model, inputs=(x_batch,), verbose=False)
    flops = 2.0 * float(macs)
    return flops, float(params)

#---------------------------------------------------------------------
# main

@dataclass
class MainCfg:
    npz_path: str = "nmini_0v1_T50_singlech.npz"
    hidden_dim: int = 128
    batch_size: int = 32
    tau_m: float = 4.0
    tau_n: float = 4.0
    input_scale: float = 1.0
    rec_density: float = 0.05
    spectral_radius: float = 0.9
    spsa_grad_clip: Optional[float] = 0.1   # 逐元素裁剪
    spsa_clip_norm: float = 0.1             # 梯度 L2 裁剪
    wrec_param_clip: float = 5.0            # 参数 L2 裁剪
    readout_agg: str = "mem_peak"#last_spike/mean_mem/spike_trace/first_latency/mem_peak
    out_tau     = 5.0
    #readout
    hist_bins: int = 5
    trace_tau: Optional[float] = None
    kernel_tau: float = 5.0
    # SPSA
    steps: int = 300
    a: float = 5e-2
    c: float = 1e-2
    A: float = 10.0
    alpha: float = 0.602
    gamma: float = 0.101
    batch_for_obj: int = 512
    grad_clip: Optional[float] = None
    # event driven
    event_driven_rec: bool = True
    min_active_to_dense: int = 8
    # 脉冲稀疏正则系数
    spike_reg_lambda: float = 5e-2
    target_spike_rate: float = 0.05

def main():
    set_seed(310)
    device = auto_select_device("auto")

    dl_tr, dl_va, dl_te, meta = make_dataloaders_from_npz(MainCfg.npz_path, batch_size=MainCfg.batch_size, device=device)
    D, C = meta["D"], meta["num_classes"]
    X_ref, y_ref = next(iter(dl_va))

    T = meta["T"]; D = meta["D"]; C = meta["num_classes"]
    model = EP_SRNNAdapter(
        n_in=D, n_rec=MainCfg.hidden_dim, n_out=C, n_t=T,
        thr=0.35, tau_m=MainCfg.tau_m, tau_o=MainCfg.out_tau, b_o=0.0,
        gamma=0.3, dt=1.0, classif=True, w_init_gain=(1.0,1.0,1.0),
        device=device
    ).to(device)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  


    sync_mon = SynchronyMoniter(
        mode="all",
        sigma=3.0,
        window=5,
        norm="min",
        alpha=0.85,
        local_sigma=1.5,
        local_window=3,
        local_kernel="gaussian"
    ).to(device)

    sync_ema = 0.0         
    ema_beta = 0.9
    SYNC_TRIGGER = 0.02
    PLOT_EVERY = 200
    TOPK_PAIRS = 20


    # 混合训练循环
    epochs = 10
    A, a0, c0, alpha, gamma = 10.0, 5e-2, 1e-2, 0.602, 0.101
    global_step = 0
    history = []

    #可视化和统计缓存
    _cached_forward_flops = None
    _cached_param_cnt     = None
    _sync_streak = 0
    _sync_q = deque(maxlen=50)

    epochs_to_plot = [10]
    raster_cache = [] 
    raster_max_neurons = 128



    for ep in range(1, epochs + 1):
        model.train()
        for X, y in dl_tr:
            global_step += 1
            X = X.to(device); y = y.to(device)

            # 仅首次估 FLOPs
            if _cached_forward_flops is None:
                fwd_flops, n_params = estimate_forward_flops(model, X)
                _cached_forward_flops = fwd_flops
                _cached_param_cnt     = n_params

            # --------  e-prop 段（取代 BP）--------
            optimizer.zero_grad()
            model.eprop_train_step(X, y)   # e-prop 写入 .grad（w_in/w_rec/w_out）
            optimizer.step()



            with torch.no_grad():
                logits, spk_seq = model(X, return_spk=True)
                train_loss = criterion(logits, y).item()
          
                if spk_seq.size(1) >= 2:
                    pre_spk  = spk_seq[:, :-1, :]        # [B,T-1,H]
                    post_spk = spk_seq[:,  1:, :]       

                    if sync_mon.mode == "hybrid":
                        comp = hybrid_decompose(sync_mon, pre_spk, post_spk)
                        history.append({
                            "step": global_step, "epoch": ep, "phase": "hybrid_comp",
                            **comp
                        })

                    S, agg = sync_mon(pre_spk, post_spk) 
                    layer_sync_batch = agg["layer_sync"].mean().item()

                    sync_ema = ema_beta * sync_ema + (1.0 - ema_beta) * layer_sync_batch
                    history.append({
                        "step": global_step, "epoch": ep, "phase": "sync",
                        "sync_layer": float(layer_sync_batch),
                        "sync_ema": float(sync_ema),
                    })

                    spk_t_h = spk_seq[0].float()            
                    spikes_per_t = spk_t_h.sum(dim=1)        
                    win_t = 5
                    kernel = torch.ones(win_t, device=spk_t_h.device).view(1,1,-1)
                    pad = win_t // 2
                    spikes_win = F.conv1d(spikes_per_t.view(1,1,-1), kernel, padding=pad).view(-1)
                    history.append({
                        "step": global_step, "epoch": ep, "phase": "spk_window",
                        "spk_win_mean": float(spikes_win.mean().item()),
                        "spk_win_max":  float(spikes_win.max().item()),
                    })


                    # 强同步对
                    SYNC_TAU = 0.20    # 建议先用 0.18~0.20，0.25 往往太高
                    S0 = S[0].detach().cpu()
                    strong = (S0 > SYNC_TAU)
                    strong_cnt  = int(strong.sum().item())
                    strong_ratio = float(strong.float().mean().item())

                    # 连续超阈值（用层均值作为“峰值”）
                    PEAK_TAU = 0.11
                    is_peak = (layer_sync_batch > PEAK_TAU)
                    _sync_streak = _sync_streak + 1 if is_peak else 0

                    # 滑动窗口里累计强同步对数
                    _sync_q.append(strong_cnt)
                    window_sum = int(sum(_sync_q))
                    Nsync = 1500
                    win_over = (window_sum >= Nsync)

                    # 记日志（phase='sync_diag'）
                    history.append({
                        "step": global_step, "epoch": ep, "phase": "sync_diag",
                        "layer_sync": float(layer_sync_batch),
                        "strong_cnt": strong_cnt, "strong_ratio": strong_ratio,
                        "streak": int(_sync_streak), "win_sum": int(window_sum),
                        "win_over": bool(win_over),
                    })


                    # Top-K 对（看第 1 个样本）
                    S0 = S[0].detach().cpu()                  # [H,H]
                    H  = S0.size(0)
                    K  = min(TOPK_PAIRS, H*H)
                    vals, idx = torch.topk(S0.reshape(-1), k=K)
                    top_pairs = [(int(i//H), int(i%H), float(v)) for i, v in zip(idx, vals)]
                    history.append({
                        "step": global_step, "epoch": ep, "phase": "sync_topk",
                        "top_pairs": top_pairs
                    })

                    # 可视化：每隔 PLOT_EVERY 步画一张热图
                    if (global_step % PLOT_EVERY) == 0:
                        plt.figure(figsize=(5,4))
                        plt.imshow(S0.numpy(), aspect="auto", interpolation="nearest")
                        plt.colorbar(label="Synchrony score")
                        plt.xlabel("pre (i)"); plt.ylabel("post (j)")
                        plt.title(f"S[0] heatmap @ step {global_step}")
                        savefig_safe(FIG_DIR / f"sync_heatmap_step{global_step}.png")

                    # （可选）触发 SPSA：只在 USE_SPSA=True 且达到阈值时
                    # if USE_SPSA and (sync_ema > SYNC_TRIGGER):
                    #     a_k = a0 / ((global_step + A) ** alpha)
                    #     c_k = c0 / (global_step ** gamma)
                    #     mon = spsa_update_Wrec(
                    #         model, X, y, ce_loss=ce_loss_no_grad,
                    #         a_k=a_k, c_k=c_k,
                    #         grad_clip=MainCfg.spsa_grad_clip,
                    #         clip_norm=MainCfg.spsa_clip_norm,
                    #         auto_gain=True, target_update_norm=1e-2, gain_clip=(0.2, 5.0)
                    #     )
                    #     history.append({**mon, "step": global_step, "epoch": ep, "phase": "spsa_stat"})
                
            history.append({
                "step": global_step , "epoch": ep, "phase": "eprop",
                "train_loss": float(train_loss),
                "flops": float(2*getattr(model, "_last_forward_macs", 0)) if _HAVE_THOP else None,
                "params": float(_cached_param_cnt),
            })


        # ===== 每个 epoch 结束后验证一次 =====
        model.eval()
        val_acc  = evaluate(model, dl_va, device)
        val_loss = evaluate_loss(model, dl_va, device)
        history.append({
            "step": global_step, "epoch": ep, "phase": "val",
            "val_loss": float(val_loss), "val_acc": float(val_acc),
            "flops": None, "params": float(_cached_param_cnt),
        })
        print(f"Epoch {ep}: Val Acc = {val_acc:.2f}% | Val Loss = {val_loss:.4f}")

        # ===  raster ===
        if ep in epochs_to_plot:
            with torch.no_grad():
                _ = model(X_ref.to(device))
                t_idx, n_idx, TT, Hsel = _raster_coords_from_spk_seq(
                    model._last_spk_seq, max_neurons=raster_max_neurons
                )
                raster_cache.append((ep, (t_idx, n_idx, TT, Hsel)))


        # === 保存 CSV + 画图 ===
        pd.DataFrame(history).to_csv(RUN_DIR / "rsnn_spsa_history.csv", index=False)
    
        hist_df = pd.DataFrame(history)

        df_sync = hist_df[hist_df["phase"] == "sync"]
        if len(df_sync):
            plt.figure()
            plt.plot(df_sync["step"], df_sync["sync_layer"], label="batch layer_sync", alpha=0.35)
            plt.plot(df_sync["step"], df_sync["sync_ema"],   label="EMA", linewidth=2)
            plt.xlabel("Step"); plt.ylabel("Synchrony score")
            plt.title("Layer Synchrony (batch & EMA)")
            plt.legend()
            savefig_safe(FIG_DIR / "sync_timeseries.png")
    
        df_ep = hist_df[hist_df["phase"] == "eprop"]
        if len(df_ep):
            plt.figure()
            plt.plot(df_ep["step"], df_ep["train_loss"])
            plt.xlabel("Step"); plt.ylabel("Train Loss"); plt.title("Training Loss per Step (e-prop)")
            savefig_safe(FIG_DIR / "train_loss_per_step_eprop.png")

  
        # hybrid 分解（first/local/hybrid）
        df_hc = hist_df[hist_df["phase"] == "hybrid_comp"]
        if len(df_hc):
            plt.figure()
            for k in ["first", "local", "hybrid"]:
                plt.plot(df_hc["step"], df_hc[k], label=k)
            plt.xlabel("Step"); plt.ylabel("Layer-mean synchrony")
            plt.title("Hybrid decomposition of synchrony")
            plt.legend()
            savefig_safe(FIG_DIR / "sync_hybrid_decompose.png")

        # layer_sync、strong_cnt/ratio
        df_sd = hist_df[hist_df["phase"] == "sync_diag"]
        if len(df_sd):
            # 时序：层均值与强同步比例
            plt.figure()
            plt.plot(df_sd["step"], df_sd["layer_sync"], label="layer_sync")
            plt.plot(df_sd["step"], df_sd["strong_ratio"], label="strong_ratio")
            plt.xlabel("Step"); plt.ylabel("value"); plt.legend()
            plt.title("Synchrony diagnostics (layer mean & strong ratio)")
            savefig_safe(FIG_DIR / "sync_diag_timeseries.png")

            # 峰值连击长度 + 滑动窗口计数
            plt.figure()
            plt.plot(df_sd["step"], df_sd["streak"], label="streak (consecutive > PEAK_TAU)")
            plt.plot(df_sd["step"], df_sd["win_sum"], label="window strong-pairs sum")
            plt.xlabel("Step"); plt.ylabel("count"); plt.legend()
            plt.title("Streak & windowed strong-pair count")
            savefig_safe(FIG_DIR / "sync_streak_window.png")

            # 每个 epoch 的平均同步（便于观察是否稳定）
            df_epmean = df_sd.groupby("epoch", as_index=False)["layer_sync"].mean()
            plt.figure()
            plt.plot(df_epmean["epoch"], df_epmean["layer_sync"], marker="o")
            plt.xlabel("Epoch"); plt.ylabel("mean layer_sync")
            plt.title("Mean synchrony per epoch")
            savefig_safe(FIG_DIR / "layer_sync_per_epoch.png")

        # 滑动时间窗口内尖峰数（整层）
        df_sw = hist_df[hist_df["phase"] == "spk_window"]
        if len(df_sw):
            plt.figure()
            plt.plot(df_sw["step"], df_sw["spk_win_mean"], label="mean spikes in window")
            plt.plot(df_sw["step"], df_sw["spk_win_max"],  label="max spikes in window")
            plt.xlabel("Step"); plt.ylabel("spikes")
            plt.title("Spikes per sliding time-window")
            plt.legend()
            savefig_safe(FIG_DIR / "spikes_sliding_window.png")

        # 把 Top-K 强同步对保存成 JSON，方便排查（可选）
        df_top = hist_df[hist_df["phase"] == "sync_topk"][["step", "epoch", "top_pairs"]]
        if len(df_top):
            df_top.to_json(RUN_DIR / "top_sync_pairs.json", orient="records", lines=True)

        if len(raster_cache):
            _plot_multi_epoch_rasters(raster_cache, FIG_DIR / "rasters_epochs.png")



    # ---- 评估 ----
    acc_tr = evaluate(model, dl_tr, device)
    acc_va = evaluate(model, dl_va, device)
    acc_te = evaluate(model, dl_te, device)
    tag = "SRNN+e-prop+SPSA" if USE_SPSA else "SRNN+e-prop"
    print(f"{tag} | Train: {acc_tr:.2f}% | Val: {acc_va:.2f}% | Test: {acc_te:.2f}%")



if __name__ == "__main__":
    main()
