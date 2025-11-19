# optim/sgd_ops.py
import torch

def sgd_update_wout(
    W_out: torch.Tensor,
    grad_logits: torch.Tensor,
    h: torch.Tensor,
    lr: float,
    wd: float,
):
    """
    dL/dW_out ≈ grad_logits^T @ h + wd * W_out
    """
    dW_out = grad_logits.t() @ h
    dW_out = dW_out + wd * W_out.data
    W_out.data -= lr * dW_out
