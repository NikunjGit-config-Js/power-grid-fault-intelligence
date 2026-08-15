import json
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")

# Load results
with open(ROOT / "results" / "test_3" / "test_3_results.json") as f:
    data = json.load(f)

print("Models:", list(data.keys()))
print()
for name in data:
    s = data[name]
    mean_f1 = s["macro_f1"]["mean"]
    std_f1 = s["macro_f1"]["std"]
    folds = s["macro_f1"]["folds"]
    print(f"{name:<25} Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}  folds: {[f'{v:.4f}' for v in folds]}")
