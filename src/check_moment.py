import torch
from model import load_tsfm_wrapper

def main():
    print("🚀 Loading MOMENT...")
    model_name = "MOMENT"  # Or use "MOMENT-1-large"
    model = load_tsfm_wrapper(model_name=model_name, seq_len=512, pred_len=96, c_in=7)

    print("\n=== MOMENT architecture (named_modules) ===")
    for name, module in model.named_modules():
        if name:
            indent = "  " * name.count(".")
            print(f"{indent}- {name} : {module.__class__.__name__}")

if __name__ == "__main__":
    main()