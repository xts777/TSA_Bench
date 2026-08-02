from __future__ import annotations

import argparse
import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import gc
import random

import numpy as np
import torch
from torch import Tensor, nn
import torch.optim as optim

try:
    from model import load_tsfm_wrapper
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import load_tsfm_wrapper

@dataclass
class MetricAccumulator:
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    n_elem: int = 0

    def update(self, y_pred: Tensor, y_true: Tensor) -> None:
        diff = y_pred - y_true
        self.sum_sq_err += float((diff * diff).sum().item())
        self.sum_abs_err += float(diff.abs().sum().item())
        self.n_elem += int(diff.numel())

    def mse(self) -> float: return self.sum_sq_err / max(1, self.n_elem)
    def mae(self) -> float: return self.sum_abs_err / max(1, self.n_elem)

def set_seed(seed: int = 2024):
    """Fix all random seeds to improve experiment reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _auto_select_target_layers(model: nn.Module) -> List[str]:
    # Some HuggingFace RMSNorm variants do not inherit from nn.LayerNorm, so we also check by class name.
    preferred_types = (nn.Linear, nn.Conv1d, nn.MultiheadAttention, nn.LayerNorm)
    llama_types = ("LlamaRMSNorm", "LlamaAttention", "LlamaMLP")
    candidates = []

    # Pass 1: collect all layers outside encoder blocks.
    for name, m in model.named_modules():
        if ".encoder.block." not in name:
            if isinstance(m, preferred_types) or m.__class__.__name__ in llama_types:
                if name not in candidates:
                    candidates.append(name)

    # Pass 2: add model-specific suffixes that we want to monitor.
    must_watch_suffixes = [
        "backbone", "head.flatten", "head.linear",  # For PatchTST
        "encoder", "projection", "inner_attention", # For iTransformer
        "final_layer_norm",                            # After all MOMENT blocks
        "model.layers"                                 # For Llama / LLMTime
    ]

    # Extract attention, FFN, and block-output layers from the 24 MOMENT blocks.
    for i in range(24):
        must_watch_suffixes.append(f"block.{i}")                                 
        must_watch_suffixes.append(f"block.{i}.layer.0.SelfAttention.o")         
        must_watch_suffixes.append(f"block.{i}.layer.1.DenseReluDense.wo")       

    # Pass 3: collect layers whose names match the selected suffixes.
    for name, m in model.named_modules():
        for suffix in must_watch_suffixes:
            if name.endswith(suffix):
                if name not in candidates:
                    candidates.append(name)

    print(f"🔍 Automatically selected monitoring layers: {len(candidates)} layers")
    return candidates


def _parse_target_layers(target_layers: Optional[str]) -> Optional[List[str]]:
    if target_layers is None:
        return None

    value = target_layers.strip()
    if not value or value.lower() == "auto":
        return None

    parsed_layers = [layer.strip() for layer in value.split(",") if layer.strip()]
    if not parsed_layers:
        return None
    return parsed_layers


def _resolve_target_layers(model: nn.Module, target_layers: Optional[str]) -> List[str]:
    parsed_layers = _parse_target_layers(target_layers)
    if parsed_layers is None:
        return _auto_select_target_layers(model)

    available_layers = [name for name, _ in model.named_modules() if name]
    resolved_layers: List[str] = []
    unresolved_layers: List[str] = []

    for requested_layer in parsed_layers:
        exact_matches = [name for name in available_layers if name == requested_layer]
        suffix_matches = [name for name in available_layers if name.endswith(f".{requested_layer}") or name.endswith(requested_layer)]

        matches = exact_matches or suffix_matches
        if len(matches) == 1:
            resolved_layers.append(matches[0])
        elif len(matches) > 1:
            unresolved_layers.append(
                f"{requested_layer} (ambiguous: {', '.join(matches[:5])})"
            )
        else:
            unresolved_layers.append(requested_layer)

    if unresolved_layers:
        available_preview = ", ".join(available_layers[:30])
        raise ValueError(
            "Unknown target layer(s): "
            + ", ".join(unresolved_layers)
            + f"\nAvailable layer names include: {available_preview}"
        )

    print(f"🎯 Using manually selected monitoring layers: {len(resolved_layers)} layers")
    return resolved_layers


def _print_model_architecture(model: nn.Module) -> None:
    target_model = getattr(model, "model", model)
    print(f"=== {target_model.__class__.__name__} architecture (named_modules) ===")
    for name, module in target_model.named_modules():
        if name:
            indent = "  " * name.count(".")
            print(f"{indent}- {name} : {module.__class__.__name__}")


def train_model(
        wrapper: nn.Module,
        data_path: str,
        seq_len: int,
        pred_len: int,
        batch_size: int,
        epochs: int,
        lr: float,
        device: torch.device,
        is_moment: bool = False,
        use_wandb: bool = False,
    ):
    """
        Automatic training helper. For MOMENT, only the prediction head is trained via linear probing.

    Args:
            wrapper: Model wrapper.
            data_path: Training data path.
            seq_len: Input sequence length.
            pred_len: Prediction horizon.
            batch_size: Batch size.
            epochs: Number of epochs.
            lr: Learning rate.
            device: Device.
            is_moment: Whether the model is MOMENT.
            use_wandb: Whether to use WandB.
    Returns:
            nn.Module: Trained model.
    """
    try:
        from data import get_dataloader_and_scaler
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data import get_dataloader_and_scaler

    from tqdm import tqdm
    if use_wandb:
        import wandb

    print(f"\n🚀 [Auto-Train] Loading training data ({data_path})...")
    train_loader, _ = get_dataloader_and_scaler(data_path, seq_len, pred_len, batch_size, flag='train')

    wrapper.train()

    if is_moment:
        print("❄️ Freezing the MOMENT backbone and training only the prediction head (linear probing).")
        for name, param in wrapper.named_parameters():
            if "head" in name:
                param.requires_grad = True  # Train the head only.
            else:
                param.requires_grad = False # Freeze the backbone.
    else:
        for p in wrapper.parameters():
            p.requires_grad = True

    # Pass only trainable parameters to the optimizer.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, wrapper.parameters()), lr=lr)
    criterion = nn.MSELoss()

    print(f"=== Auto-Training Start ({epochs} Epochs) ===")
    for epoch in range(epochs):
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (batch_x, batch_y) in enumerate(pbar):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = wrapper(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss (MSE): {avg_loss:.4f}")

        if use_wandb:
            wandb.log({
                "train/loss": avg_loss,
                "train/epoch": epoch+1
            })

    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad = False
    print("=== Auto-Training Complete ===\n")
    return wrapper

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSFM Robustness Benchmark with WandB")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="PatchTST")
    parser.add_argument("--attack_method", type=str, default="TSA", choices=["TSA", "GWN", "FGSM", "PGD"])
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tau", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--pgd_alpha", type=float, default=0.02, help="Step size for PGD attack")
    parser.add_argument("--pgd_steps", type=int, default=10, help="Number of iterations for PGD attack")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--c_in",
        type=int,
        default=None,
        help="Number of input channels/features. Required for architecture inspection mode.",
    )
    
    # Auto-training and seed settings
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to pre-trained weights")
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--memo", type=str, default=None)
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer names to monitor. Use 'auto' or omit the argument to keep automatic selection.",
    )
    parser.add_argument(
        "--show_model_architecture",
        action="store_true",
        help="Print the unified model architecture and exit before running attacks.",
    )

    # WandB settings
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--project_name", type=str, default="TSFM-Robustness")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-3B", help="HuggingFace model name for LLMTime")

    # Visualization settings
    parser.add_argument("--plot_results", action="store_true", help="Generate and save comparison plots for time-series and layer divergence")
    parser.add_argument("--plot_dir", type=str, default="results/plots", help="Directory to save generated plot images")
    return parser

def main() -> None:
    args = build_arg_parser().parse_args()
    
    set_seed(args.seed)
    print(f"🌱 Random seed fixed to: {args.seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.show_model_architecture and args.c_in is None:
        raise ValueError("--c_in is required when using --show_model_architecture.")

    if args.show_model_architecture:
        c_in = int(args.c_in)
    else:
        try:
            from data import get_dataloader_and_scaler
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from data import get_dataloader_and_scaler

        test_loader, scaler = get_dataloader_and_scaler(args.data_path, args.seq_len, args.pred_len, args.batch_size, flag='test')
        c_in = int(np.asarray(scaler.mean).shape[0]) if np.asarray(scaler.mean).ndim > 0 else 1
    
    # Step 1: load the model.
    model = load_tsfm_wrapper(args.model_name, args.seq_len, args.pred_len, c_in, checkpoint_path=args.checkpoint_path).to(device)
    if hasattr(model.model, 'model_name') and args.llm_model:
        model.model.model_name = args.llm_model

    if args.show_model_architecture:
        _print_model_architecture(model)
        return

    try:
        from data import get_dataloader_and_scaler
        from attacker import GWNAttacker, TSAttacker, FGSMAttacker, PGDAttacker, TSFMObserver
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data import get_dataloader_and_scaler
        from attacker import GWNAttacker, TSAttacker, FGSMAttacker, PGDAttacker, TSFMObserver

    from tqdm import tqdm

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.project_name,
            config=vars(args),
            name=f"{args.model_name}_{args.attack_method}_{time.strftime('%m%d-%H%M')}"
        )
    
    # Step 2: decide whether to auto-train, including MOMENT linear probing.
    is_moment = args.model_name.lower().startswith("moment")
    is_llmtime = args.model_name.lower().startswith("llm")
    has_weights = args.checkpoint_path is not None and os.path.exists(args.checkpoint_path)

    if has_weights:
        print(f"ℹ️ Evaluating with the specified weights ({args.checkpoint_path}).")
    elif is_llmtime:
        print(f"ℹ️ {args.model_name} is a zero-shot forecasting model, so training is skipped.")
    elif is_moment:
        print(f"⚠️ {args.model_name} is pretrained, but its prediction head is untrained. Starting automatic head-only training.")
        model = train_model(model, args.data_path, args.seq_len, args.pred_len, args.train_batch_size, args.train_epochs, args.learning_rate, device, is_moment=True, use_wandb=args.use_wandb)
        os.makedirs("./weights", exist_ok=True)
        dataset_name = os.path.basename(args.data_path).split('.')[0]
        torch.save(model.model.state_dict(), f"./weights/{args.model_name}_{dataset_name}_head.pth")
    else:
        print(f"⚠️ No pretrained weights were found for {args.model_name}. Starting automatic training from scratch.")
        model = train_model(model, args.data_path, args.seq_len, args.pred_len, args.train_batch_size, args.train_epochs, args.learning_rate, device, is_moment=False, use_wandb=args.use_wandb)
        os.makedirs("./weights", exist_ok=True)
        dataset_name = os.path.basename(args.data_path).split('.')[0]
        if args.memo:
            torch.save(model.model.state_dict(), f"./weights/{args.model_name}_{dataset_name}_{args.memo}.pth")
        else:
            torch.save(model.model.state_dict(), f"./weights/{args.model_name}_{dataset_name}.pth")

    # Step 3: prepare the attacker.
    loss_fn = nn.MSELoss()
    if args.attack_method == "GWN":
        attacker = GWNAttacker(model, loss_fn)
    elif args.attack_method == "FGSM":
        attacker = FGSMAttacker(model, loss_fn, epsilon=args.epsilon)
    elif args.attack_method == "PGD":
        attacker = PGDAttacker(model, loss_fn, epsilon=args.epsilon, alpha=args.pgd_alpha, steps=args.pgd_steps)
    else:
        attacker = TSAttacker(model, loss_fn, args.tau, args.epsilon, 5, model.is_channel_independent)

    if is_llmtime:
        if hasattr(model.model, '_init_llm'):
            model.model._init_llm()

    observer = TSFMObserver(model, _resolve_target_layers(model, args.target_layers))
    clean_metrics, adv_metrics = MetricAccumulator(), MetricAccumulator()
    report = {}

    print("\n=== Benchmark attack test started ===")
    pbar = tqdm(test_loader, desc="Attacking")
    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        if args.max_batches and batch_idx >= args.max_batches: 
            pbar.close()
            break
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        with torch.no_grad(): y_hat_clean = model(batch_x)
        
        adv_list = []
        for i in range(batch_x.shape[0]):
            w = attacker.attack(batch_x[i:i+1], y_hat_clean[i:i+1])
            adv_list.append(batch_x[i:i+1] * (1 + w))
        
        batch_x_adv = torch.cat(adv_list, dim=0).detach()
        
        with torch.no_grad():
            report = observer.diagnose_divergence(batch_x, batch_x_adv)
            y_hat_adv = model(batch_x_adv)

        y_real, yc_real, ya_real = map(scaler.inverse_transform, [batch_y, y_hat_clean, y_hat_adv])
        clean_metrics.update(yc_real, y_real)
        adv_metrics.update(ya_real, y_real)

        clean_mse_val = (yc_real - y_real).pow(2).mean().item()
        adv_mse_val = (ya_real - y_real).pow(2).mean().item()

        if args.use_wandb:
            wandb.log({
                "batch/clean_mse": clean_mse_val,
                "batch/adv_mse": adv_mse_val,
                "batch/degradation_ratio": adv_mse_val / (clean_mse_val + 1e-8),
                "batch_idx": batch_idx
            })
        pbar.set_postfix({"Adv_MSE": f"{adv_mse_val:.4f}"})
        # print(f"  Progress: Batch {batch_idx+1}/{len(test_loader)}")

        torch.cuda.empty_cache()
        gc.collect()

    # Step 4: save and print the results.
    summary = {
        "final/clean_mse": clean_metrics.mse(), "final/clean_mae": clean_metrics.mae(),
        "final/adv_mse": adv_metrics.mse(), "final/adv_mae": adv_metrics.mae(),
        "final/degradation": (adv_metrics.mse() - clean_metrics.mse()) / max(clean_metrics.mse(), 1e-8)
    }
    
    print("\n" + "="*30 + " SUMMARY " + "="*30)
    for k, v in summary.items(): print(f"{k}: {v:.6f}")

    if args.plot_results:
        try:
            from visualizer import plot_time_series_comparison, plot_layer_wise_divergence
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from visualizer import plot_time_series_comparison, plot_layer_wise_divergence

        print(f"\n📊 Generating evaluation plots in directory: {args.plot_dir}...")
        os.makedirs(args.plot_dir, exist_ok=True)

        # Plot time series comparison for the last processed sample
        x_c_sample = batch_x[0].cpu().numpy()
        x_a_sample = batch_x_adv[0].cpu().numpy()
        y_t_sample = y_real[0].cpu().numpy()
        y_c_sample = yc_real[0].cpu().numpy()
        y_a_sample = ya_real[0].cpu().numpy()

        plot_time_series_comparison(
            x_clean=x_c_sample,
            x_adv=x_a_sample,
            y_true=y_t_sample,
            y_clean=y_c_sample,
            y_adv=y_a_sample,
            channel_idx=0,
            title=f"{args.model_name} - {args.attack_method} Attack & Prediction Comparison",
            save_path=os.path.join(args.plot_dir, f"{args.model_name}_{args.attack_method}_ts_comparison.png")
        )

        # Plot layer-wise divergence diagnostic report
        if report:
            plot_layer_wise_divergence(
                layer_metrics=report,
                title=f"{args.model_name} - Layer-wise Observer Divergence ({args.attack_method})",
                save_path=os.path.join(args.plot_dir, f"{args.model_name}_{args.attack_method}_layer_divergence.png")
            )

    if args.use_wandb:
        wandb.log(summary)
        # Include layer diagnostics in the W&B table as well.
        obs_table = wandb.Table(columns=["Layer", "Divergence_MSE", "Cos_Dist", "Norm_Ratio", "L_inf"])
        for layer, metrics in report.items():
            obs_table.add_data(
                layer, 
                np.mean(metrics["mse"]), 
                np.mean(metrics["cos_dist"]),
                np.mean(metrics.get("norm_ratio", 0)),
                np.mean(metrics.get("l_inf", 0))
            )
        wandb.log({"diagnostics/layer_impact": obs_table})
        wandb.finish()

if __name__ == "__main__":
    main()