# spsa.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

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

    W = model.W_rec
    device = W.device
    dtype  = W.dtype


    Delta = torch.empty_like(W, device=device, dtype=dtype).bernoulli_(0.5).mul_(2.).sub_(1.)
    if mask is not None:

        Delta = torch.where(mask, Delta, torch.zeros_like(Delta))

    c = float(c_k)


    W_orig = W.clone()


    W_plus  = W_orig + c * Delta
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
@torch.no_grad()
def ce_loss_no_grad(model, X, y):
    # 如果你的 forward 有可选返回 (logits, spk_seq)，确保只拿 logits：
    logits = model(X)                 # 等价于 model.forward(X)
    loss = F.cross_entropy(logits, y, reduction="mean")
    return float(loss.item())