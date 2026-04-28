import torch
from model import load_tsfm_wrapper

def main():
    print("🚀 MOMENTをロード中...")
    # ※seq_lenやpred_lenは、現在実験しているデータに合わせてください
    # MOMENTは通常 512入力 などをデフォルトにしていることが多いです
    model_name = "MOMENT" # または "MOMENT-1-large" など、あなたの実装に合わせて
    model = load_tsfm_wrapper(model_name=model_name, seq_len=512, pred_len=96, c_in=7)

    print("\n=== MOMENT の内部構造 (named_modules) ===")
    for name, module in model.named_modules():
        # ルート（空文字列）はスキップ
        if name:
            # 階層の深さに合わせてインデントをつける
            indent = "  " * name.count(".")
            print(f"{indent}- {name} : {module.__class__.__name__}")

if __name__ == "__main__":
    main()