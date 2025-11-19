# spsa.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

@torch.no_grad()
def _spsa_update_tensor(
    W: torch.Tensor,
    model,
    X, y,
    ce_loss,
    a_k: float,
    c_k: float,
    grad_clip: Optional[float] = None,
    clip_norm: Optional[float] = None,
    auto_gain: bool = False,
    target_update_norm: Optional[float] = None,
    gain_clip: Optional[Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    proj: Optional[torch.Tensor] = None,   # 子空间基，形状 [k, D]
    direction: str = "spsa",               # 新增：扰动分布类型： "spsa" 或 "gaussian"
    normalize_dir: bool = False,           # 新增：是否对方向做 L2 归一化
):
    """
    通用的 SPSA 更新函数，对给定权重张量 W 做一次 SPSA 更新。
    注意：这个函数是就地修改 W 的（in-place）。
    """
    device = W.device
    dtype = W.dtype

    # 生成扰动 Delta
    if proj is None:
        # --- 全维空间 ---
        if direction == "spsa":
            # Rademacher ±1
            Delta = torch.empty_like(W, device=device, dtype=dtype).bernoulli_(0.5).mul_(2.).sub_(1.)
        elif direction == "gaussian":
            # 高斯方向
            Delta = torch.randn_like(W, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown direction mode: {direction}")
    else:
        # --- 子空间 proj: [k, D_flat] ---
        flat_dim = W.numel()
        k, D_flat = proj.shape
        assert D_flat == flat_dim, f"proj.shape[1]={D_flat} 与 W.numel()={flat_dim} 不一致"

        proj = proj.to(device=device, dtype=dtype)

        if direction == "spsa":
            # 子空间坐标上的 ±1
            z = torch.empty(k, device=device, dtype=dtype).bernoulli_(0.5).mul_(2.).sub_(1.)
        elif direction == "gaussian":
            # 子空间坐标上的高斯
            z = torch.randn(k, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown direction mode: {direction}")

        Delta_flat = torch.matmul(proj.transpose(0, 1), z)     
        Delta = Delta_flat.view_as(W)

    if mask is not None:
        Delta = torch.where(mask, Delta, torch.zeros_like(Delta))

    # 可选：对方向做 L2 归一化（只看非零元素）
    if normalize_dir:
        flat = Delta.view(-1)
        if mask is not None:
            flat = flat[mask.view(-1)]
        norm = flat.norm().item()
        if norm > 1e-8:
            Delta = Delta / norm

    c = float(c_k)


    W_orig = W.clone()

    #计算正负梯度
    W_plus = W_orig + c * Delta
    W_minus = W_orig - c * Delta


    W.copy_(W_plus)
    loss_plus = float(ce_loss(model, X, y))


    W.copy_(W_minus)
    loss_minus = float(ce_loss(model, X, y))


    W.copy_(W_orig)


    scale = (loss_plus - loss_minus) / (2.0 * c + 1e-12)
    ghat = Delta * scale

    if mask is not None:
        ghat = torch.where(mask, ghat, torch.zeros_like(ghat))


    if grad_clip is not None:
        ghat.clamp_(min=-grad_clip, max=grad_clip)


    if clip_norm is not None:
        n = ghat.norm().item()
        if n > clip_norm and n > 0:
            ghat.mul_(clip_norm / (n + 1e-12))


    update = -a_k * ghat


    if auto_gain and (target_update_norm is not None):
        un = update.norm().item()
        if un > 0:
            gain = target_update_norm / (un + 1e-12)
            if gain_clip is not None:
                gain = max(gain_clip[0], min(gain, gain_clip[1]))
            update.mul_(gain)

    if mask is not None:
        update = torch.where(mask, update, torch.zeros_like(update))


    W.add_(update)


    return {
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "est_grad_norm": float(ghat.norm().item()),
        "update_norm": float(update.norm().item()),
        "c_k": float(c_k),
        "a_k": float(a_k),
        "masked_frac": float(mask.float().mean().item()) if mask is not None else 1.0,
    }


#  W_rec 
@torch.no_grad()
def spsa_update_Wrec(
    model,
    X, y,
    ce_loss,
    a_k: float,
    c_k: float,
    grad_clip: Optional[float] = None,
    clip_norm: Optional[float] = None,
    auto_gain: bool = False,
    target_update_norm: Optional[float] = None,
    gain_clip: Optional[Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    proj: Optional[torch.Tensor] = None,
    direction: str = "spsa",              # 新增：方向模式
    normalize_dir: bool = False,          # 新增：是否归一化
):
    return _spsa_update_tensor(
        model.W_rec, model, X, y, ce_loss,
        a_k, c_k,
        grad_clip=grad_clip,
        clip_norm=clip_norm,
        auto_gain=auto_gain,
        target_update_norm=target_update_norm,
        gain_clip=gain_clip,
        mask=mask,
        proj=proj,
        direction=direction,
        normalize_dir=normalize_dir,
    )

# W_in
@torch.no_grad()
def spsa_update_Win(
    model,
    X, y,
    ce_loss,
    a_k: float,
    c_k: float,
    grad_clip: Optional[float] = None,
    clip_norm: Optional[float] = None,
    auto_gain: bool = False,
    target_update_norm: Optional[float] = None,
    gain_clip: Optional[Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    proj: Optional[torch.Tensor] = None,
    direction: str = "spsa",
    normalize_dir: bool = False,
):
    return _spsa_update_tensor(
        model.W_in, model, X, y, ce_loss,
        a_k, c_k,
        grad_clip=grad_clip,
        clip_norm=clip_norm,
        auto_gain=auto_gain,
        target_update_norm=target_update_norm,
        gain_clip=gain_clip,
        mask=mask,
        proj=proj,
        direction=direction,
        normalize_dir=normalize_dir,
    )

@torch.no_grad()
def ce_loss_no_grad(model, X, y):
    """
    用于 SPSA 的 CE 损失计算
    """
    logits = model(X)              
    loss = F.cross_entropy(logits, y, reduction="mean")
    return float(loss.item())
