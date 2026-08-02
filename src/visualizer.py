from __future__ import annotations

import os
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

# Use a clean, modern plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_time_series_comparison(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    y_clean: Optional[np.ndarray] = None,
    y_adv: Optional[np.ndarray] = None,
    channel_idx: int = 0,
    title: str = "TSA-Bench: Adversarial Attack & Prediction Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plots historical time series (clean vs. adversarial) and predicted sequence (ground truth vs clean pred vs adv pred).

    Args:
        x_clean: Array of shape (seq_len, C) or (seq_len,)
        x_adv: Array of shape (seq_len, C) or (seq_len,)
        y_true: Ground truth future target, shape (pred_len, C) or (pred_len,)
        y_clean: Prediction under clean input, shape (pred_len, C) or (pred_len,)
        y_adv: Prediction under adversarial input, shape (pred_len, C) or (pred_len,)
        channel_idx: Channel index to visualize for multivariate series
        title: Figure main title
        save_path: Filepath to save the plot image (e.g. 'plots/sample_attack.png')
        show: Whether to call plt.show()
    """
    # Helper to squeeze and select channel
    def _get_ch(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr[:, channel_idx]
        elif arr.ndim == 1:
            return arr
        elif arr.ndim == 3:  # (B=1, L, F)
            return arr[0, :, channel_idx]
        return arr

    xc = _get_ch(x_clean)
    xa = _get_ch(x_adv)
    yt = _get_ch(y_true)
    ypc = _get_ch(y_clean)
    ypa = _get_ch(y_adv)

    seq_len = len(xc)
    t_hist = np.arange(seq_len)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False, gridspec_kw={'height_ratios': [2, 1]})
    ax_main, ax_delta = axes

    # Main plot: Time Series & Predictions
    ax_main.plot(t_hist, xc, label="Clean Input (x)", color="#1f77b4", lw=2)
    ax_main.plot(t_hist, xa, label="Adversarial Input (x_adv)", color="#d62728", linestyle="--", lw=1.8, alpha=0.85)

    if yt is not None or ypc is not None or ypa is not None:
        pred_len = len(yt) if yt is not None else (len(ypc) if ypc is not None else len(ypa))
        t_pred = np.arange(seq_len, seq_len + pred_len)

        # Draw a vertical divider between historical lookback and future prediction horizon
        ax_main.axvline(x=seq_len - 0.5, color="gray", linestyle=":", lw=1.5, label="Prediction Horizon Split")

        if yt is not None:
            ax_main.plot(t_pred, yt, label="Ground Truth (y)", color="#2ca02c", lw=2, marker="o", ms=3)
        if ypc is not None:
            ax_main.plot(t_pred, ypc, label="Clean Prediction (y_clean)", color="#17becf", linestyle="-.", lw=1.8)
        if ypa is not None:
            ax_main.plot(t_pred, ypa, label="Adversarial Prediction (y_adv)", color="#ff7f0e", linestyle="--", lw=2, marker="x", ms=4)

    ax_main.set_ylabel(f"Value (Channel {channel_idx})", fontsize=11)
    ax_main.set_title(title, fontsize=13, fontweight="bold")
    ax_main.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax_main.grid(True, linestyle="--", alpha=0.5)

    # Subplot: Added Perturbation (x_adv - x_clean)
    delta = xa - xc
    ax_delta.plot(t_hist, delta, label="Perturbation (x_adv - x)", color="#9467bd", lw=1.5)
    ax_delta.axhline(0, color="black", lw=0.8, linestyle="-")
    ax_delta.set_xlabel("Time Step", fontsize=11)
    ax_delta.set_ylabel("Perturbation $\\delta$", fontsize=11)
    ax_delta.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax_delta.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Saved comparison plot to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_layer_wise_divergence(
    layer_metrics: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Layer-wise Representation Divergence (Observer Diagnosis)",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plots the divergence metrics collected across layers by the Observer.

    Args:
        layer_metrics: Dictionary mapping layer_name -> {metric_name: float_value}
                       e.g. {'backbone.layer.0': {'mse': 0.12, 'cosine_distance': 0.05, ...}}
        metrics_to_plot: List of metric keys to visualize. Defaults to ['mse', 'cosine_distance', 'norm_ratio', 'linf_distance'].
        title: Plot title
        save_path: Filepath to save the plot image
        show: Whether to display plot
    """
    if not layer_metrics:
        raise ValueError("layer_metrics dictionary is empty")

    layer_names = list(layer_metrics.keys())
    # Shorten overly long layer names for cleaner x-axis display
    short_names = [name.replace("model.", "").replace("backbone.", "") for name in layer_names]

    if metrics_to_plot is None:
        metrics_to_plot = ["mse", "cosine_distance", "norm_ratio", "linf_distance"]

    # Filter metrics actually present in layer_metrics
    available_metrics = [m for m in metrics_to_plot if any(m in dict_val or (m == "cosine_distance" and "cos_dist" in dict_val) for dict_val in layer_metrics.values())]

    n_metrics = len(available_metrics)
    if n_metrics == 0:
        raise ValueError(f"None of requested metrics {metrics_to_plot} found in layer_metrics keys.")

    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    color_map = {
        "mse": "#d62728",
        "cosine_distance": "#1f77b4",
        "cos_dist": "#1f77b4",
        "norm_ratio": "#2ca02c",
        "linf_distance": "#ff7f0e",
        "l_inf": "#ff7f0e",
    }

    metric_labels = {
        "mse": "Representation MSE",
        "cosine_distance": "Cosine Similarity / Distance",
        "cos_dist": "Cosine Distance",
        "norm_ratio": "Norm Ratio ||h_adv|| / ||h_clean||",
        "linf_distance": "$L_\\infty$ Distance",
        "l_inf": "$L_\\infty$ Distance",
    }

    x_indices = np.arange(len(layer_names))

    for ax, metric in zip(axes, available_metrics):
        # Support aliases for keys
        raw_key = metric
        if metric == "cosine_distance" and "cos_dist" in list(layer_metrics.values())[0]:
            raw_key = "cos_dist"
        elif metric == "linf_distance" and "l_inf" in list(layer_metrics.values())[0]:
            raw_key = "l_inf"

        values = [np.mean(layer_metrics[name].get(raw_key, 0.0)) for name in layer_names]
        color = color_map.get(metric, "#333333")
        label = metric_labels.get(metric, metric)

        ax.plot(x_indices, values, marker="o", color=color, linewidth=2, label=label)
        if metric == "norm_ratio":
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (1.0)")

        ax.set_ylabel(metric.upper(), fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=11, loc="left")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    axes[-1].set_xticks(x_indices)
    axes[-1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    axes[-1].set_xlabel("Monitored Model Layers (Input -> Output)", fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Saved layer divergence plot to: {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    print("Testing visualizer module with dummy data...")
    seq_len, pred_len = 96, 24
    np.random.seed(42)

    # Dummy inputs
    t = np.linspace(0, 4 * np.pi, seq_len)
    x_clean = np.sin(t)[:, None] + 0.1 * np.random.randn(seq_len, 1)
    x_adv = x_clean + 0.15 * np.sign(np.cos(t))[:, None]

    y_true = np.sin(np.linspace(4 * np.pi, 5 * np.pi, pred_len))[:, None]
    y_clean = y_true + 0.05 * np.random.randn(pred_len, 1)
    y_adv = y_clean - 0.3 * np.random.randn(pred_len, 1)

    # Save time-series plot
    plot_time_series_comparison(
        x_clean, x_adv, y_true, y_clean, y_adv,
        channel_idx=0,
        save_path="scratch/plots/demo_time_series_comparison.png"
    )

    # Dummy layer metrics
    dummy_layer_metrics = {
        "model.backbone.enc.0": {"mse": np.array([0.02]), "cos_dist": np.array([0.98]), "norm_ratio": np.array([1.01]), "l_inf": np.array([0.05])},
        "model.backbone.enc.1": {"mse": np.array([0.08]), "cos_dist": np.array([0.91]), "norm_ratio": np.array([1.08]), "l_inf": np.array([0.12])},
        "model.backbone.enc.2": {"mse": np.array([0.25]), "cos_dist": np.array([0.76]), "norm_ratio": np.array([1.30]), "l_inf": np.array([0.35])},
        "model.head.linear":    {"mse": np.array([0.42]), "cos_dist": np.array([0.62]), "norm_ratio": np.array([1.55]), "l_inf": np.array([0.60])},
    }

    plot_layer_wise_divergence(
        dummy_layer_metrics,
        save_path="scratch/plots/demo_layer_divergence.png"
    )
    print("Done! Check scratch/plots/")

