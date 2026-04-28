# `src/main.py` 解説（TSFM 敵対的堅牢性ベンチマーク）

この文書は、`/home/ubuntu/TSA/src/main.py` が「データ読み込み→モデル/攻撃/Observerの構築→テスト全体で敵対攻撃＋内部表現診断→逆変換後の誤差を集計→整形して出力」までをどのように行うかを説明します。

## 1. このスクリプトの目的

`main.py` は TSFM（Time Series Foundation Model）に対する敵対的堅牢性評価の“中央指令塔”です。

- `data.py` からテスト用 `DataLoader` と `scaler` を取得
- `model.py` から統一インターフェース（入力 `(B, L, F)`、出力 `(B, pred_len, F)`）を満たすラッパーを取得
- `attacker.py` から攻撃器（`TSAttacker` または `GWNAttacker`）と Observer（`TSFMObserver`）を構築
- テストセット全バッチに対して「攻撃生成（B=1の厳格ループ）」と「Observer診断（バッチ一括）」を実行
- `scaler.inverse_transform` を使って逆正規化した“実スケール”で MSE/MAE を計算し、最後にまとめて表示します

## 2. 実行方法（例）

### 2.1 GWN 攻撃

```bash
conda run -n tsfm-bench python /home/ubuntu/TSA/src/main.py \
  --data_path /home/ubuntu/experiment/dataset/ETTh1.csv \
  --model_name PatchTST \
  --attack_method GWN \
  --seq_len 96 --pred_len 48 --batch_size 32
```

### 2.2 TSA 攻撃（スモークテスト用）

TSA は重いので、まずは短い系列・小さいバッチで完走確認するのがおすすめです。

```bash
conda run -n tsfm-bench python -u /home/ubuntu/TSA/src/main.py \
  --data_path /home/ubuntu/experiment/dataset/ETTh1.csv \
  --model_name PatchTST \
  --attack_method TSA \
  --seq_len 24 --pred_len 12 --batch_size 1 \
  --max_batches 1 --max_iter 1 --tau 3 --epsilon 0.1
```

## 3. CLI 引数

`main.py` は `argparse` で以下の引数を受け取ります。

### 必須

- `--data_path` : データセットのパス（ファイル or フォルダ）

### モデル/攻撃

- `--model_name` : `"PatchTST"` や `"iTransformer"` など（デフォルト `PatchTST`）
- `--attack_method` : `"TSA"` または `"GWN"`（デフォルト `TSA`）

### 形状パラメータ

- `--seq_len` : 入力系列長 `L`（デフォルト `96`）
- `--pred_len` : 予測長 `L_pred`（デフォルト `48`）
- `c_in`（特徴次元）はデータの `scaler` から自動推定されます

### バッチ/評価制御

- `--batch_size` : テストで使用するバッチサイズ（デフォルト `32`）
- `--max_batches` : スモークテスト用。`0` のときは全バッチ処理（デフォルト `0`）

### TSA / GWN ハイパラ（デバッグ・制御用）

- `--gwn_scale` : GWN のノイズスケール（デフォルト `0.05`）
- `--tau` : TSA の攻撃対象時刻数（デフォルト `9`）
- `--epsilon` : TSA の摂動係数（デフォルト `0.1`）
- `--max_iter` : TSA の反復回数（デフォルト `5`）

## 4. 実装構造（主要クラス/関数）

### 4.1 `MetricAccumulator`

クリーン/敵対の予測精度（MSE/MAE）を、テスト全体で合算して最後に平均化します。

- `update(y_pred, y_true)` : 要素ごとの誤差を足し込み
- `mse()` : `sum_sq_err / n_elem`
- `mae()` : `sum_abs_err / n_elem`

重要なのは、「逆変換（inverse_transform）後のテンソル」で `update` している点です（後述）。

### 4.2 `_auto_select_target_layers`

Observer が差分を観測するためのレイヤー名を、内部モデルから自動抽出します。

- `nn.Linear`, `nn.Conv1d`, `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `nn.LayerNorm` 等を候補にする
- 見つからなければパラメータを持つリーフモジュールを候補にフォールバック
- 最後方の候補から最大 `max_layers`（デフォルト `10`）まで採用

### 4.3 `_merge_observer_metrics`

各バッチで得られた Observer の `report` を、層ごとに集計します。

- `report[layer]["mse"]` と `report[layer]["cos_dist"]` はバッチ方向に応じて配列を返す想定（ここでは numpy 平均で集計）
- 最終的に層ごとの平均 MSE と平均 Cosine Distance を表示します

## 5. テスト評価ループ（重要：厳格な Accumulation Protocol）

`main.py` のコアは、`for batch_idx, (batch_x, batch_y) in enumerate(test_loader):` の内部です。

要求される厳格な処理順は以下です。

### 5.1 前処理（デバイス移動）

- `batch_x` は `(B, L, F)`
- `batch_y` は `(B, pred_len, F)`
- 両方を `device`（CPU/GPU）へ移動

### 5.2 Step 1：クリーン予測（バッチ一括）

`with torch.no_grad():`

- `y_hat_clean_full = model(batch_x)` を一括計算して保持

### 5.3 Step 2-4：B=1 攻撃を逐次生成して毒データを cat

攻撃器 `TSAttacker` は実装上、B=1 に強く依存するため、バッチをそのまま渡しません。

このため、

1. `adv_list` を空配列として用意
2. `for i in range(bsz):`
3. `single_x = batch_x[i:i+1]` として `(1, L, F)` を取り出す
4. `single_y_hat_clean = y_hat_clean_full[i:i+1]` で `(1, pred_len, F)` を用意
5. 攻撃器に `single_x` と（必要なら）`single_y_hat_clean` を渡して摂動を計算
6. `single_x_adv = single_x * (1 + w)` を生成して `adv_list` に追加
7. 最後に `batch_x_adv_full = torch.cat(adv_list, dim=0)` で `(B, L, F)` の毒バッチを復元

ここで “攻撃は逐次”、一方で “Observer の診断はバッチ一括” を実現しています。

### 5.4 Step 5：Observer 診断（バッチ一括）

`observer.diagnose_divergence(batch_x, batch_x_adv_full)` を一度だけ呼びます。

これにより、

- クリーン入力の内部表現
- 敵対入力の内部表現

を同じバッチ条件で比較して、層ごとの散度（MSE、Cosine Distance）を収集します。

## 6. 逆変換後の実スケール誤差（必須要件）

`data.py` の `get_dataloader_and_scaler` は `StandardScaler` を返します。

モデル出力とラベルは正規化空間の値なので、

- `y_hat_clean_full`
- `y_hat_adv_full`
- `batch_y`

のいずれも `scaler.inverse_transform(...)` を通してから MSE/MAE を計算します。

具体的には、

- `y_hat_clean_real = scaler.inverse_transform(y_hat_clean_full)`
- `y_hat_adv_real = scaler.inverse_transform(y_hat_adv_full)`
- `batch_y_real = scaler.inverse_transform(batch_y)`

これを `MetricAccumulator.update(...)` に渡します。

## 7. 結果の出力（最後の表示）

ループ終了後、以下をまとめて表示します。

1. 【予測精度の低下】（逆変換後の）
  - Clean MSE/MAE と Adv MSE/MAE
2. 【内部表現の診断】
  - Observer で収集した各レイヤーごとの平均 MSE と Cosine Distance

出力は研究用に見やすいよう、層名でソートして安定した順序で表示します。

## 8. 注意点（運用上のポイント）

- `--max_batches` を小さくして TSA をまずスモークテストするのがおすすめです（計算が重い）。
- `--seq_len/--pred_len` により計算量が大きく変わります。
- Observer はモデル内部から自動で層を選ぶため、非常に大きいモデルでは候補数が増える場合があります（必要なら `max_layers` を調整する拡張も可能です）。

