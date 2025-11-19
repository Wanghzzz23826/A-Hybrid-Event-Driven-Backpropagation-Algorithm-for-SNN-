# utils/metrics.py
from typing import Tuple
import torch
import torch.nn as nn

try:
    from thop import profile
    _HAVE_THOP = True
except Exception:
    profile = None
    _HAVE_THOP = False

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> float:
    ce = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss_sum = 0, 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss_sum += ce(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    acc = 100.0 * correct / max(total, 1)
    return acc

@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader, device: torch.device) -> float:
    ce = nn.CrossEntropyLoss(reduction="sum")
    total, loss_sum = 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss_sum += ce(logits, y).item()
        total += y.numel()
    return loss_sum / max(total, 1)

@torch.no_grad()
def estimate_forward_flops(model: torch.nn.Module, x_batch: torch.Tensor) -> Tuple[float, float]:
    params = sum(p.numel() for p in model.parameters())
    if not _HAVE_THOP:
        return float("nan"), float(params)
    macs, _ = profile(model, inputs=(x_batch,), verbose=False)
    flops = 2.0 * float(macs)
    return flops, float(params)
