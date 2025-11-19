import numpy as np

def make_gaussian_binary(n_per_class=200, dim=64, 
                         mean0=0.3, mean1=0.7, 
                         sigma=1.0, seed=42, 
                         p0=0.5):
    """
    生成二分类高斯数据：
    - 类0: N(mean0 * 1_vec, sigma^2 I)
    - 类1: N(mean1 * 1_vec, sigma^2 I)
    """

    rng = np.random.default_rng(seed)
    n0 = int(n_per_class if isinstance(n_per_class, int) else n_per_class[0])
    n1 = int(n_per_class if isinstance(n_per_class, int) else n_per_class)  # 简单写法
    
    # 也可用先按先验 p0 采样整体数量 n 的标签
    # 这里默认每类各 n_per_class
    mu0 = np.full(dim, mean0, dtype=float)
    mu1 = np.full(dim, mean1, dtype=float)
    cov = (sigma ** 2) * np.eye(dim)

    X0 = rng.multivariate_normal(mu0, cov, size=n0)
    X1 = rng.multivariate_normal(mu1, cov, size=n1)
    y0 = np.zeros(n0, dtype=int)
    y1 = np.ones(n1, dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # 打乱
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

# 示例：1维、各类200个样本，类间均值差=1，方差=1
X, y = make_gaussian_binary(n_per_class=200, dim=64, mean0=0.3, mean1=0.7, sigma=1.0)

# 保存为 CSV
np.savetxt("gauss_binary_X.csv", X, delimiter=",")
np.savetxt("gauss_binary_y.csv", y, fmt="%d", delimiter=",")
print(X.shape, y.shape)  # (400, 64) (400,)


#USE:
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split

# class GaussianBinaryDataset(Dataset):
#     def __init__(self, n_per_class=200, dim=2, mean0=0.0, mean1=1.0, sigma=1.0, seed=42):
#         X, y = make_gaussian_binary(n_per_class, dim, mean0, mean1, sigma, seed)
#         self.X = torch.from_numpy(X).float()
#         self.y = torch.from_numpy(y).long()
#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx): return self.X[idx], self.y[idx]

# # 构造数据集与划分
# full_ds = GaussianBinaryDataset(n_per_class=500, dim=2, mean0=0.0, mean1=1.0, sigma=1.0, seed=0)
# n_train = int(0.8 * len(full_ds))
# n_val = len(full_ds) - n_train
# train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
