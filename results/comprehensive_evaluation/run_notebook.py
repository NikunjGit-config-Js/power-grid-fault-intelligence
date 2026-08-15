"""Execute all code cells from the comprehensive evaluation notebook."""
import json, ast, os, sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "numpy"

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")
sys.path.insert(0, str(ROOT))

nb_path = ROOT / "results" / "comprehensive_evaluation" / "comprehensive_evaluation.ipynb"
nb = json.load(open(nb_path))

# Execute all code cells sequentially
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    print(f"\n--- Executing cell {i} ---")
    try:
        exec(compile(src, f'cell_{i}', 'exec'), globals())
        print(f"  Cell {i}: OK")
    except Exception as e:
        print(f"  Cell {i}: ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n\nDone executing all cells.")
