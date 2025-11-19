# make_nmini.py
import numpy as np, torch, torchvision as tv, torch.nn.functional as F

# ===== 生成 N-MNIST 风格小数据集 =====
def make_nmini(n_train=1000, n_val=200, n_test=200,
               digits=(0,1),           # 二分类 0 vs 1；改成 None 则用 10 类
               T=50,                   # 时间步
               img_size=34, pad=3,     # 28->34 与 N-MNIST 一致
               n_saccades=3,           # 眼跳段数
               max_pix_per_step=1,     # 每步最大像素移动（CPU 友好）
               diff_thresh=0.15,       # 事件阈值（0~1），越小事件越密
               bipolar=True,           # True=双极性两路；False=合成一路
               seed=310, download=True, root="./data"):
    """
    返回: (Xtr, ytr), (Xva, yva), (Xte, yte)
    X*: [N, T, D]  (D=34*34*(2 if bipolar else 1))
    y*: [N]
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # 1) 取 MNIST，并筛选类
    ds_train = tv.datasets.MNIST(root, train=True, download=download, transform=None)
    ds_test  = tv.datasets.MNIST(root, train=False, download=download, transform=None)

    def filt(ds):
        X, y = [], []
        for img, lab in ds:
            if digits is None or lab in digits:
                X.append(np.array(img, dtype=np.float32)/255.0)
                y.append(int(lab if digits is None else (0 if lab==digits[0] else 1)))
        return np.stack(X), np.array(y, dtype=np.int64)
    Xtr_raw, ytr = filt(ds_train)
    Xte_raw, yte = filt(ds_test)

    # 合并再切分，确保随机性一致
    Xall = np.concatenate([Xtr_raw, Xte_raw], 0)
    yall = np.concatenate([ytr,    yte],    0)
    idx  = rng.permutation(len(yall))
    Xall, yall = Xall[idx], yall[idx]

    n_tot = n_train + n_val + n_test
    Xall, yall = Xall[:n_tot], yall[:n_tot]
    Xtr, Xva, Xte = np.split(Xall, [n_train, n_train+n_val], axis=0)
    ytr, yva, yte = np.split(yall, [n_train, n_train+n_val], axis=0)

    # 2) 填充到 34×34
    def pad34(x):  # x:[N,28,28]
        x = torch.from_numpy(x).unsqueeze(1)               # [N,1,28,28]
        x = F.pad(x, (pad,pad,pad,pad), mode='constant', value=0.0)  # [N,1,34,34]
        return x.squeeze(1)                                # [N,34,34]

    Xtr = pad34(Xtr); Xva = pad34(Xva); Xte = pad34(Xte)

    # 3) 构造眼跳轨迹（每个样本不同轨迹，保证多样性）
    def sample_path(T, n_seg, max_step, rng_local):
        # 返回每个 t 的 (dx, dy) 位移；分段随机方向，小步移动
        steps_per = T // n_seg
        traj = []
        for s in range(n_seg):
            dx = rng_local.integers(-max_step, max_step+1)
            dy = rng_local.integers(-max_step, max_step+1)
            for _ in range(steps_per):
                traj.append((dx, dy))
        while len(traj)<T: traj.append((0,0))
        return np.array(traj[:T], dtype=np.int32)  # [T,2]

    # 4) 生成事件序列
    def imgs_to_events(X34, label_array):
        N = X34.shape[0]
        Cmult = 2 if bipolar else 1
        D = img_size*img_size*Cmult
        X_out = np.zeros((N, T, D), dtype=np.float32)

        for i in range(N):
            rng_i = np.random.default_rng(int(seed + i*13))
            traj  = sample_path(T, n_saccades, max_pix_per_step, rng_i)

            # 逐时刻生成移动后的帧
            frames = []
            base = torch.from_numpy(X34[i])  # [34,34]
            x, y = 0, 0
            for t in range(T):
                dx, dy = traj[t]
                x = int(np.clip(x+dx, -pad, pad))
                y = int(np.clip(y+dy, -pad, pad))
                # 平移：通过 pad+crop 实现（无插值）
                f = F.pad(base.unsqueeze(0).unsqueeze(0),
                          (pad,pad,pad,pad), value=0.0)            # [1,1,40,40]
                f = f[:,:, (pad+y):(pad+y+img_size), (pad+x):(pad+x+img_size)]  # [1,1,34,34]
                frames.append(f.squeeze().numpy())
            frames = np.stack(frames, 0)  # [T,34,34]

            # 帧差 -> 事件
            diff = frames[1:] - frames[:-1]                 # [T-1,34,34]
            pos = (diff >=  diff_thresh).astype(np.float32)
            neg = (diff <= -diff_thresh).astype(np.float32)
            if bipolar:
                spikes = np.concatenate([pos, neg], axis=0) # [2*(T-1),34,34]
                # 统一到 T 长度（首帧无事件，用全 0 填在前）
                spikes = np.concatenate([np.zeros((1,34,34),np.float32),
                                          spikes[:(T-1)]], axis=0)  # 先放正极性
                spikes_neg = np.concatenate([np.zeros((1,34,34),np.float32),
                                             neg[:(T-1)]], axis=0)
                spikes = np.stack([spikes, spikes_neg], axis=-1)     # [T,34,34,2]
                spikes = spikes.reshape(T, -1)                       # [T, 34*34*2]
            else:
                spikes_mag = ((np.abs(diff) >= diff_thresh).astype(np.float32))
                spikes = np.concatenate([np.zeros((1,34,34),np.float32),
                                         spikes_mag[:(T-1)]], axis=0) # [T,34,34]
                spikes = spikes.reshape(T, -1)                         # [T, 34*34]

            X_out[i] = spikes
        return X_out, label_array

    Xtr_s, ytr = imgs_to_events(Xtr.numpy(), ytr)
    Xva_s, yva = imgs_to_events(Xva.numpy(), yva)
    Xte_s, yte = imgs_to_events(Xte.numpy(), yte)
    return (Xtr_s, ytr), (Xva_s, yva), (Xte_s, yte)

if __name__ == "__main__":
    (Xtr,ytr),(Xva,yva),(Xte,yte)=make_nmini(
        n_train=800, n_val=100, n_test=100,
        digits=(0,1), T=50, diff_thresh=0.12,
        bipolar=False,   # 先用单通道，兼容你现在的 input_dim=1156
        max_pix_per_step=1
    )
    # 保存为 npz（推荐），也可改成你喜欢的格式
    np.savez("nmini_0v1_T50_singlech.npz",
             Xtr=Xtr,ytr=ytr, Xva=Xva,yva=yva, Xte=Xte,yte=yte)
    print("Saved:", Xtr.shape, Xva.shape, Xte.shape)
