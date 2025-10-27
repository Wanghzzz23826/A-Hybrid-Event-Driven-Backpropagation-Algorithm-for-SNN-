# spsa.py
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn

@dataclass
class SPSAConfig:
    steps: int = 300
    a: float = 5e-2
    c: float = 1e-2
    A: float = 10.0
    alpha: float = 0.602
    gamma: float = 0.101
    batch_size_for_obj: int = 512
    grad_clip: Optional[float] = 0.1
    clip_norm: float = 0.1
    auto_gain: bool = True
    target_update_norm: float = 1e-2
    gain_clip: Tuple[float,float] = (0.2, 5.0)

@torch.no_grad()
def ce_loss_no_grad(model, X, y):
    logits = model(X)
    return nn.functional.cross_entropy(logits, y, reduction="mean")

@torch.no_grad()
def spsa_update_Wrec(
    model, X, y, ce_loss, a_k, c_k,
    grad_clip: Optional[float] = 0.1,
    clip_norm: float = 0.1,
    auto_gain: bool = True,
    target_update_norm: float = 1e-2,
    gain_clip: Tuple[float,float] = (0.2, 5.0),
):
    W = model.W_rec
    W0 = W.data.clone()
    delta = (torch.randint_like(W, 2) * 2 - 1).to(torch.float32)

    W.data = W0 + c_k * delta; Jp = ce_loss(model, X, y)
    macs_jp = int(getattr(model, "_last_forward_macs", 0))
    W.data = W0 - c_k * delta; Jm = ce_loss(model, X, y)
    macs_jm = int(getattr(model, "_last_forward_macs", 0))
    W.data = W0

    g_coeff = (Jp - Jm) / (2.0 * c_k)
    g_hat   = g_coeff * (1.0 / (delta + 1e-12))
    if grad_clip is not None:
        g_hat = torch.clamp(g_hat, -grad_clip, grad_clip)

    gn = torch.linalg.norm(g_hat)
    if gn > clip_norm:
        g_hat.mul_(clip_norm / (gn + 1e-12))

    if auto_gain:
        pred_upd_norm = a_k * (torch.linalg.norm(g_hat) + 1e-12)
        scale = float(target_update_norm / pred_upd_norm)
        scale = max(gain_clip[0], min(gain_clip[1], scale))
        a_k = a_k * scale

    W.add_(-a_k * g_hat)

    dW = -a_k * g_hat
    upd_norm = float(torch.linalg.norm(dW))
    rel_upd  = upd_norm / (float(torch.linalg.norm(W)) + 1e-12)
    jm_gap   = float(abs(Jp.item() - Jm.item()))
    return {
        "Jp": float(Jp.item()), "Jm": float(Jm.item()),
        "g_norm": float(torch.linalg.norm(g_hat).item()),
        "a_k": float(a_k), "macs_spsa": int(macs_jp + macs_jm),
        "upd_norm": upd_norm, "rel_upd": rel_upd, "jm_gap": jm_gap
    }
