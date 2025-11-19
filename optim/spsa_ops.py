# optim/spsa_ops.py
from typing import Dict, Any
import torch
from spsa import spsa_update_Wrec, spsa_update_Win, ce_loss_no_grad  # 复用你原来的实现

__all__ = ["spsa_update_Wrec", "spsa_update_Win", "ce_loss_no_grad"]
