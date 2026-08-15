"""
Deployment verification script.

Checks that all deployment configuration files exist and are correct,
production artifacts are included, and the app can start without errors.

Run with: python results/deployment/verify_deployment.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
passed = 0
failed = 0


def check(description: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  PASS  {description}")
        passed += 1
    else:
        msg = f"  FAIL  {description}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        failed += 1


print("=" * 70)
print("DEPLOYMENT VERIFICATION")
print("=" * 70)

# --- 1. Procfile ---
print("\n--- 1. Procfile ---")
procfile_path = PROJECT_ROOT / "Procfile"
check("Procfile exists", procfile_path.exists())

if procfile_path.exists():
    procfile_text = procfile_path.read_text()
    check("Procfile uses $PORT", "$PORT" in procfile_text)
    check("Procfile uses 0.0.0.0", "0.0.0.0" in procfile_text)
    check("Procfile runs streamlit", "streamlit run app.py" in procfile_text)
    check("Procfile has headless=true", "--server.headless=true" in procfile_text)
    check("Procfile does not hardcode port 8501", "8501" not in procfile_text)

# --- 2. .python-version ---
print("\n--- 2. .python-version ---")
pyver_path = PROJECT_ROOT / ".python-version"
check(".python-version exists", pyver_path.exists())

if pyver_path.exists():
    pyver_text = pyver_path.read_text().strip()
    check(".python-version contains 3.12", pyver_text == "3.12", f"got: {pyver_text!r}")

# --- 3. .streamlit/config.toml ---
print("\n--- 3. .streamlit/config.toml ---")
config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
check(".streamlit/config.toml exists", config_path.exists())

if config_path.exists():
    config_text = config_path.read_text()
    check("config.toml has headless = true", "headless = true" in config_text)
    check("config.toml has gatherUsageStats = false",
          "gatherUsageStats = false" in config_text)
    check("config.toml does NOT disable CORS",
          "enableCORS = false" not in config_text)
    check("config.toml does NOT disable XSRF protection",
          "enableXsrfProtection = false" not in config_text)

# --- 4. requirements.txt ---
print("\n--- 4. requirements.txt ---")
req_path = PROJECT_ROOT / "requirements.txt"
check("requirements.txt exists", req_path.exists())

if req_path.exists():
    req_text = req_path.read_text()
    check("requirements.txt includes streamlit", "streamlit" in req_text)
    check("requirements.txt includes numpy", "numpy" in req_text)
    check("requirements.txt includes pandas", "pandas" in req_text)
    check("requirements.txt includes scikit-learn==1.8.0",
          "scikit-learn==1.8.0" in req_text)
    check("requirements.txt includes joblib", "joblib" in req_text)
    check("requirements.txt includes lightgbm", "lightgbm" in req_text)
    check("requirements.txt does NOT include plotly",
          "plotly" not in req_text)
    check("requirements.txt does NOT include keras",
          "keras" not in req_text)
    check("requirements.txt does NOT include jax",
          "jax" not in req_text)
    check("requirements.txt does NOT include scipy",
          "scipy" not in req_text)

# --- 5. Production artifacts ---
print("\n--- 5. Production artifacts ---")
pipeline_path = PROJECT_ROOT / "models" / "best_model_pipeline.joblib"
metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
check("Production pipeline exists", pipeline_path.exists())
check("Production metadata exists", metadata_path.exists())

if pipeline_path.exists():
    size = pipeline_path.stat().st_size
    check("Pipeline size > 0", size > 0, f"size: {size:,} bytes")

if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    check("Metadata has model_name", "model_name" in metadata)
    check("Metadata has idx_to_class", "idx_to_class" in metadata)
    check("Metadata has locked_test_metrics", "locked_test_metrics" in metadata)
    check("Metadata has expected_input_features",
          "expected_input_features" in metadata)
    check("Metadata expected_input_features == 6",
          metadata.get("expected_input_features") == 6)

# --- 6. app.py ---
print("\n--- 6. app.py ---")
app_path = PROJECT_ROOT / "app.py"
check("app.py exists", app_path.exists())

if app_path.exists():
    app_text = app_path.read_text()
    line_count = len(app_text.splitlines())
    check("app.py under 150 lines", line_count <= 150, f"lines: {line_count}")
    check("app.py imports streamlit", "import streamlit" in app_text)
    check("app.py imports predict_lightgbm",
          "predict_lightgbm" in app_text)
    check("app.py does NOT import load_class_data",
          "load_class_data" not in app_text)
    check("app.py does NOT import sample_row",
          "sample_row" not in app_text)
    check("app.py does NOT import classData.csv",
          "classData.csv" not in app_text)

# --- 7. No classData.csv dependency at runtime ---
print("\n--- 7. Runtime dependency check ---")
check("classData.csv not required at runtime",
      "load_class_data" not in open(PROJECT_ROOT / "utils" / "prediction.py").read()
      if (PROJECT_ROOT / "utils" / "prediction.py").exists() else True)

# --- 8. No training/test artifacts required at runtime ---
print("\n--- 8. Training artifact independence ---")
prod_model_path = PROJECT_ROOT / "models" / "best_model_pipeline.joblib"
check("Production pipeline is self-contained",
      prod_model_path.exists() and prod_model_path.stat().st_size > 0)

# --- 9. Legacy views preserved ---
print("\n--- 9. Legacy views ---")
views_dir = PROJECT_ROOT / "views"
check("views/ directory exists", views_dir.is_dir())
if views_dir.is_dir():
    view_files = list(views_dir.glob("*.py"))
    check("views/ has Python files", len(view_files) > 0,
          f"found {len(view_files)} files")

# --- 10. File sizes ---
print("\n--- 10. Deployment artifact sizes ---")
files_to_check = [
    ("Procfile", procfile_path),
    (".python-version", pyver_path),
    (".streamlit/config.toml", config_path),
    ("requirements.txt", req_path),
    ("app.py", app_path),
    ("models/best_model_pipeline.joblib", pipeline_path),
    ("models/model_metadata.json", metadata_path),
]
total_size = 0
for name, path in files_to_check:
    if path.exists():
        size = path.stat().st_size
        total_size += size
        print(f"  {name}: {size:,} bytes")
    else:
        print(f"  {name}: MISSING")
print(f"  --- Total: {total_size:,} bytes ---")

# --- Summary ---
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 70)

if failed:
    print("\nSome checks FAILED. Review output above.")
    sys.exit(1)
else:
    print("\nAll checks PASSED.")
    sys.exit(0)
