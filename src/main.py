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
import pandas as pd
import torch
from torch import Tensor, nn
import torch.optim as optim

from tqdm import tqdm

import wandb

try:
    from data import get_dataloader_and_scaler
    from model import load_tsfm_wrapper
    from attacker import GWNAttacker, TSAttacker, TSFMObserver
except ImportError:
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

def set_seed(seed: int = 2024):
    """すべての乱数シードを固定し、実験の再現性を担保する"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _auto_select_target_layers(model: nn.Module) -> List[str]:
    preferred_types = (nn.Linear, nn.Conv1d, nn.MultiheadAttention, nn.LayerNorm)
    candidates = []

    # 1. Encoderブロック「以外」の層をすべて網羅（余裕があるため全取得）
    for name, m in model.named_modules():
        if ".encoder.block." not in name:
            if isinstance(m, preferred_types):
                if name not in candidates:
                    candidates.append(name)

    # 2. ピンポイントで狙い撃ちするサフィックス
    must_watch_suffixes = [
        "backbone", "head.flatten", "head.linear",  # PatchTST用
        "encoder", "projection", "inner_attention", # iTransformer用
        "final_layer_norm"                          # MOMENTの全ブロック通過後
    ]

    # MOMENTの24層の中身から「Attention」「FFN」「ブロック出口」を抽出
    for i in range(24):
        must_watch_suffixes.append(f"block.{i}")                                 
        must_watch_suffixes.append(f"block.{i}.layer.0.SelfAttention.o")         
        must_watch_suffixes.append(f"block.{i}.layer.1.DenseReluDense.wo")       

    # 3. 指定したサフィックスに一致する層を回収
    for name, m in model.named_modules():
        for suffix in must_watch_suffixes:
            if name.endswith(suffix):
                if name not in candidates:
                    candidates.append(name)

    print(f"🔍 自動選択された監視層: {len(candidates)} 層")
    return candidates


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
    自動学習用の関数。MOMENTの場合はHeadのみをLinear Probingします。

    Args:
        wrapper: モデルラッパー
        data_path: 訓練データのパス
        seq_len: 入力系列長
        pred_len: 予測系列長
        batch_size: バッチサイズ
        epochs: エポック数
        lr: 学習率
        device: デバイス
        is_moment: MOMENTかどうか
        use_wandb: WandBを使用するかどうか
    Returns:
        nn.Module: 学習済みモデル
    """
    print(f"\n🚀 [Auto-Train] 訓練データ ({data_path}) をロード中...")
    train_loader, _ = get_dataloader_and_scaler(data_path, seq_len, pred_len, batch_size, flag='train')

    wrapper.train()
    
    if is_moment:
        print("❄️ MOMENTのBackboneを凍結し、予測Headのみを学習(Linear Probing)します。")
        for name, param in wrapper.named_parameters():
            if "head" in name:
                param.requires_grad = True  # Headだけ学習
            else:
                param.requires_grad = False # Backboneは凍結
    else:
        for p in wrapper.parameters():
            p.requires_grad = True

    # 学習対象のパラメータだけをOptimizerに渡す
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
    parser.add_argument("--attack_method", type=str, default="TSA", choices=["TSA", "GWN"])
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tau", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max_batches", type=int, default=0)
    
    # 自動学習・シード関連
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to pre-trained weights")
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--memo", type=str, default=None)

    # WandB設定
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--project_name", type=str, default="TSFM-Robustness")
    return parser

def main() -> None:
    args = build_arg_parser().parse_args()
    
    set_seed(args.seed)
    print(f"🌱 Random seed fixed to: {args.seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args),
            name=f"{args.model_name}_{args.attack_method}_{time.strftime('%m%d-%H%M')}"
        )

    test_loader, scaler = get_dataloader_and_scaler(args.data_path, args.seq_len, args.pred_len, args.batch_size, flag='test')
    c_in = int(np.asarray(scaler.mean).shape[0]) if np.asarray(scaler.mean).ndim > 0 else 1
    
    # 1. モデルのロード
    model = load_tsfm_wrapper(args.model_name, args.seq_len, args.pred_len, c_in, checkpoint_path=args.checkpoint_path).to(device)
    
    # 2. 自動学習の判定ロジック (MOMENTのLinear Probing対応)
    is_moment = args.model_name.lower().startswith("moment")
    has_weights = args.checkpoint_path is not None and os.path.exists(args.checkpoint_path)

    if has_weights:
        print(f"ℹ️ 指定された重み ({args.checkpoint_path}) を使用して評価します。")
    elif is_moment:
        print(f"⚠️ {args.model_name} は事前学習済みですが、予測Headが未学習です。Headのみ自動学習を開始します。")
        model = train_model(model, args.data_path, args.seq_len, args.pred_len, args.train_batch_size, args.train_epochs, args.learning_rate, device, is_moment=True, use_wandb=args.use_wandb)
        os.makedirs("./weights", exist_ok=True)
        dataset_name = os.path.basename(args.data_path).split('.')[0]
        torch.save(model.model.state_dict(), f"./weights/{args.model_name}_{dataset_name}_head.pth")
    else:
        print(f"⚠️ {args.model_name} の学習済み重みが見つかりません。フルスクラッチ自動学習を開始します。")
        model = train_model(model, args.data_path, args.seq_len, args.pred_len, args.train_batch_size, args.train_epochs, args.learning_rate, device, is_moment=False, use_wandb=args.use_wandb)
        os.makedirs("./weights", exist_ok=True)
        dataset_name = os.path.basename(args.data_path).split('.')[0]
        if args.memo:
            torch.save(model.model.state_dict(), f"./weights/{args.model_name}_{dataset_name}_{args.memo}.pth")
        else:
            torch.save(model.model.state_dict(), f"./weights/{args.model_name}_{dataset_name}.pth")

    # 3. 攻撃の準備
    loss_fn = nn.MSELoss()
    if args.attack_method == "GWN":
        attacker = GWNAttacker(model, loss_fn)
    else:
        attacker = TSAttacker(model, loss_fn, args.tau, args.epsilon, 5, model.is_channel_independent)

    observer = TSFMObserver(model, _auto_select_target_layers(model))
    clean_metrics, adv_metrics = MetricAccumulator(), MetricAccumulator()
    report = {}

    print("\n=== ベンチマーク（攻撃テスト）開始 ===")
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
        
        # ⚠️ 超重要：detach() でメモリ爆発を防ぐ！
        batch_x_adv = torch.cat(adv_list, dim=0).detach()
        
        # ⚠️ 超重要：no_grad() で囲んで計算履歴を保存させない！
        with torch.no_grad():
            report = observer.diagnose_divergence(batch_x, batch_x_adv)
            y_hat_adv = model(batch_x_adv)

        y_real, yc_real, ya_real = map(scaler.inverse_transform, [batch_y, y_hat_clean, y_hat_adv])
        clean_metrics.update(yc_real, y_real)
        adv_metrics.update(ya_real, y_real)

        if args.use_wandb:
            clean_mse_val = (yc_real - y_real).pow(2).mean().item()
            adv_mse_val = (ya_real - y_real).pow(2).mean().item()
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

    # 4. 結果保存と出力
    summary = {
        "final/clean_mse": clean_metrics.mse(), "final/clean_mae": clean_metrics.mae(),
        "final/adv_mse": adv_metrics.mse(), "final/adv_mae": adv_metrics.mae(),
        "final/degradation": (adv_metrics.mse() - clean_metrics.mse()) / max(clean_metrics.mse(), 1e-8)
    }
    
    print("\n" + "="*30 + " SUMMARY " + "="*30)
    for k, v in summary.items(): print(f"{k}: {v:.6f}")

    if args.use_wandb:
        wandb.log(summary)
        # L_inf や Norm_Ratio もすべて出力する完璧なテーブル
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