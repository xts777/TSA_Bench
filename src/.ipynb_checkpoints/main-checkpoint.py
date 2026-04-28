# main.py
from __future__ import annotations

import argparse
import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import gc

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

# WandBのインポート
import wandb

try:
    from data import get_dataloader_and_scaler
    from model import load_tsfm_wrapper
    from attacker import GWNAttacker, TSAttacker, TSFMObserver
except ImportError:
    # 実行パスに応じたインポート処理
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data import get_dataloader_and_scaler
    from model import load_tsfm_wrapper
    from attacker import GWNAttacker, TSAttacker, TSFMObserver

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

def _auto_select_target_layers(model: nn.Module, max_layers: int = 10) -> List[str]:
    # (前述の自動レイヤー選択ロジック)
    preferred_types = (nn.Linear, nn.Conv1d, nn.MultiheadAttention, nn.LayerNorm)
    candidates = [name for name, m in model.named_modules() if isinstance(m, preferred_types)]
    return candidates

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSFM Robustness Benchmark with WandB")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="PatchTST")
    parser.add_argument("--attack_method", type=str, default="TSA", choices=["TSA", "GWN"])
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tau", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max_batches", type=int, default=0)

    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to pre-trained weights")

    # WandB設定
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--project_name", type=str, default="TSFM-Robustness")
    return parser

def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. WandBの初期化
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args),
            name=f"{args.model_name}_{args.attack_method}_{time.strftime('%m%d-%H%M')}"
        )

    # 2. セットアップ
    test_loader, scaler = get_dataloader_and_scaler(args.data_path, args.seq_len, args.pred_len, args.batch_size)
    c_in = int(np.asarray(scaler.mean).shape[0]) if np.asarray(scaler.mean).ndim > 0 else 1
    
    model = load_tsfm_wrapper(args.model_name, args.seq_len, args.pred_len, c_in, checkpoint_path=args.checkpoint_path).to(device)
    loss_fn = nn.MSELoss()
    
    if args.attack_method == "GWN":
        attacker = GWNAttacker(model, loss_fn)
    else:
        attacker = TSAttacker(model, loss_fn, args.tau, args.epsilon, 5, model.is_channel_independent)

    observer = TSFMObserver(model, _auto_select_target_layers(model))
    
    clean_metrics, adv_metrics = MetricAccumulator(), MetricAccumulator()
    repr_results = []

    # 3. メインループ
    for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
        if args.max_batches and batch_idx >= args.max_batches: break
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 攻撃 & 診断 (Accumulation Protocol)
        with torch.no_grad(): y_hat_clean = model(batch_x)
        
        adv_list = []
        for i in range(batch_x.shape[0]):
            w = attacker.attack(batch_x[i:i+1], y_hat_clean[i:i+1])
            adv_list.append(batch_x[i:i+1] * (1 + w))
        
        batch_x_adv = torch.cat(adv_list, dim=0)
        report = observer.diagnose_divergence(batch_x, batch_x_adv)
        
        with torch.no_grad(): y_hat_adv = model(batch_x_adv)

        # 逆変換 & 指標計算
        y_real, yc_real, ya_real = map(scaler.inverse_transform, [batch_y, y_hat_clean, y_hat_adv])
        clean_metrics.update(yc_real, y_real)
        adv_metrics.update(ya_real, y_real)

        # バッチごとのWandBログ
        if args.use_wandb:
            wandb.log({
                "batch/clean_mse": (yc_real - y_real).pow(2).mean().item(),
                "batch/adv_mse": (ya_real - y_real).pow(2).mean().item(),
                "batch/degradation_ratio": (ya_real - y_real).pow(2).mean().item() / (yc_real - y_real).pow(2).mean().item(),
                "batch_idx": batch_idx
            })
        print(f"Progress: {batch_idx}/{len(test_loader)}")

        torch.cuda.empty_cache()
        gc.collect()

    # 4. 結果保存と出力
    summary = {
        "final/clean_mse": clean_metrics.mse(), "final/clean_mae": clean_metrics.mae(),
        "final/adv_mse": adv_metrics.mse(), "final/adv_mae": adv_metrics.mae(),
        "final/degradation": (adv_metrics.mse() - clean_metrics.mse()) / clean_metrics.mse()
    }
    
    print("\n" + "="*30 + " SUMMARY " + "="*30)
    for k, v in summary.items(): print(f"{k}: {v:.6f}")

    if args.use_wandb:
        wandb.log(summary)
        # Observerの結果をTable形式で保存
        obs_table = wandb.Table(columns=["Layer", "Divergence_MSE", "Cos_Dist"])
        for layer, metrics in report.items():
            obs_table.add_data(layer, np.mean(metrics["mse"]), np.mean(metrics["cos_dist"]))
        wandb.log({"diagnostics/layer_impact": obs_table})
        wandb.finish()

if __name__ == "__main__":
    main()