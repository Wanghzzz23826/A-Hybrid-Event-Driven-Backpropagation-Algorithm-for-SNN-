# utils/plot.py
from typing import List, Tuple
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

def get_out_dirs(run_dir_name: str = "runs_SPSA"):
    try:
        base = Path(__file__).parent.parent  # project 根目录
    except NameError:
        base = Path.cwd()
    run_dir = (base / run_dir_name).resolve()
    fig_dir = (run_dir / "figs").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, fig_dir

def savefig_safe(pathlike):
    p = Path(pathlike)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()

@torch.no_grad()
def raster_coords_from_spk_seq(spk_seq_BTH: torch.Tensor, max_neurons: int = 128):
    spk = spk_seq_BTH[0].detach().cpu().numpy()
    T, H = spk.shape
    H_sel = min(H, max_neurons)
    idx = np.arange(H_sel)
    spk_sel = spk[:, idx]
    t_idx, n_idx = np.nonzero(spk_sel)
    return t_idx, n_idx, T, H_sel

def plot_multi_epoch_rasters(
    raster_data: List[Tuple[int, Tuple[np.ndarray, np.ndarray, int, int]]],
    save_path: str,
):
    ncols = len(raster_data)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, (ep, (t_idx, n_idx, T, H_sel)) in zip(axes, raster_data):
        ax.scatter(t_idx, n_idx, s=3)
        ax.set_xlim(0, T)
        ax.set_ylim(-0.5, H_sel - 0.5)
        ax.set_xlabel("time")
        if ax is axes[0]:
            ax.set_ylabel(f"neuron (mod {H_sel})")
    plt.tight_layout()
    savefig_safe(save_path)
