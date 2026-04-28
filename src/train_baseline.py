import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 作成済みのモジュールをインポート
from data import get_dataloader_and_scaler
from model import load_tsfm_wrapper

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. データセットの準備 (今回は flag='train' で訓練データを取得)
    data_path = "../experiment/dataset/ETTh1.csv"
    seq_len = 96
    pred_len = 48
    batch_size = 32
    
    print("Loading training data...")
    train_loader, scaler = get_dataloader_and_scaler(
        data_path, seq_len, pred_len, batch_size, flag='train'
    )
    
    # チャネル数の推論
    c_in = int(np.asarray(scaler.mean).shape[0]) if np.asarray(scaler.mean).ndim > 0 else 1

    # 2. モデルの呼び出し
    print("Loading PatchTST model...")
    wrapper = load_tsfm_wrapper("PatchTST", seq_len, pred_len, c_in).to(device)
    
    # 【超重要】
    # BaseTSFMWrapper は攻撃用にパラメータを凍結(requires_grad=False)しているので、
    # 学習のために解凍（Unfreeze）して、trainモードにします。
    wrapper.train()
    for p in wrapper.parameters():
        p.requires_grad = True

    # 3. オプティマイザと損失関数
    optimizer = optim.Adam(wrapper.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 4. 学習ループ（10エポックで十分な精度が出ます）
    epochs = 10
    print(f"\n=== Start Training ({epochs} Epochs) ===")
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            # 予測
            pred = wrapper(batch_x)
            # 損失計算 (正規化された空間で計算するのが時系列の基本)
            loss = criterion(pred, batch_y)
            
            # 逆伝播 & パラメータ更新
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss (MSE): {avg_loss:.4f}")

    # 5. 重みの保存
    os.makedirs("/weights", exist_ok=True)
    save_path = "/weights/PatchTST_ETTh1_96_48.pth"
    
    # wrapper.model に生のPatchTSTが格納されているので、その状態を保存
    torch.save(wrapper.model.state_dict(), save_path)
    print(f"\n✅ Training complete! Model saved to: {save_path}")

if __name__ == "__main__":
    main()