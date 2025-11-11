# spsa.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


@torch.no_grad()
def _spsa_update_tensor(
    W: torch.Tensor,
    model,
    X, y,
    ce_loss,                # 函数句柄，输入 (model, X, y)，输出一个标量 loss（float 或 0-dim tensor）
    a_k: float,
    c_k: float,
    grad_clip: Optional[float] = None,       # 元素级裁剪
    clip_norm: Optional[float] = None,      # 全局范数裁剪
    auto_gain: bool = False,                # 是否根据 target_update_norm 自动缩放 update
    target_update_norm: Optional[float] = None,
    gain_clip: Optional[Tuple[float, float]] = None,  # (min_gain, max_gain)
    mask: Optional[torch.Tensor] = None,    # 可选的布尔 mask，限制哪些位置可以被更新
):
    """
    通用的 SPSA 更新函数，对给定权重张量 W 做一次 SPSA 更新。
    注意：这个函数是就地修改 W 的（in-place）。
    """
    device = W.device
    dtype = W.dtype

    # 生成扰动 Delta
    Delta = torch.empty_like(W, device=device, dtype=dtype).bernoulli_(0.5).mul_(2.).sub_(1.)
    if mask is not None:
        Delta = torch.where(mask, Delta, torch.zeros_like(Delta))

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
):
    """
    对 model.W_rec 做 SPSA 更新。
    """
    return _spsa_update_tensor(
        model.W_rec, model, X, y, ce_loss,
        a_k, c_k,
        grad_clip=grad_clip,
        clip_norm=clip_norm,
        auto_gain=auto_gain,
        target_update_norm=target_update_norm,
        gain_clip=gain_clip,
        mask=mask,
    )


# W_in 的

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
):
    """
    对 model.W_in 做 SPSA 更新。
    """
    return _spsa_update_tensor(
        model.W_in, model, X, y, ce_loss,
        a_k, c_k,
        grad_clip=grad_clip,
        clip_norm=clip_norm,
        auto_gain=auto_gain,
        target_update_norm=target_update_norm,
        gain_clip=gain_clip,
        mask=mask,
    )

@torch.no_grad()
def ce_loss_no_grad(model, X, y):
    """
    用于 SPSA 的 CE 损失计算
    """
    logits = model(X)              
    loss = F.cross_entropy(logits, y, reduction="mean")
    return float(loss.item())
