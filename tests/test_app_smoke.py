"""
Streamlit render smoke test.

Drives app.py headlessly with streamlit.testing.v1.AppTest and walks every page with
every loadable model, asserting nothing raises. This catches UI-level breakage that the
inference tests in test_pipeline.py cannot see.

A single AppTest instance is reused across checks. Each instance keeps its own cache, so
spawning one per check would re-deserialize the 16 MB Random Forest every time.

Run from the repository root:

    python tests/test_app_smoke.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("KERAS_BACKEND", "numpy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "Overview",
    "Live Fault Prediction",
    "Batch Prediction",
    "Data Explorer",
    "Model Insights",
    "About Project",
]

MODEL_PAGES = ["Live Fault Prediction", "Batch Prediction", "Model Insights"]

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def record(name: str, fn) -> bool:
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
        return True
    except Exception as exc:
        FAILED.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
        return False


def assert_clean(at: AppTest) -> None:
    if at.exception:
        raise AssertionError("; ".join(str(e.message) for e in at.exception))


def new_app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert_clean(at)
    return at


def main() -> int:
    print("\n=== app boots ===")
    at = new_app()

    record(
        "sidebar exposes all six pages",
        lambda: (_ for _ in ()).throw(AssertionError(at.sidebar.radio[0].options))
        if at.sidebar.radio[0].options != PAGES
        else None,
    )

    model_options = at.sidebar.selectbox[0].options
    print(f"        models offered: {model_options}")
    record("all three models load into the selector", lambda: _assert(len(model_options) == 3))

    print("\n=== every page renders ===")
    for page in PAGES:
        def go(p=page):
            at.sidebar.radio[0].set_value(p).run()
            assert_clean(at)
            _assert(len(at.title) >= 1, "page rendered no title")

        record(f"page: {page}", go)

    print("\n=== every model renders on model-dependent pages ===")
    for model in model_options:
        for page in MODEL_PAGES:
            def go(p=page, m=model):
                at.sidebar.radio[0].set_value(p)
                at.sidebar.selectbox[0].set_value(m).run()
                assert_clean(at)

            record(f"{page} + {model}", go)

    print("\n=== interactions ===")

    def load_sample():
        at.sidebar.radio[0].set_value("Live Fault Prediction")
        at.sidebar.selectbox[0].set_value("Decision Tree").run()
        assert_clean(at)
        button = [b for b in at.button if "Load sample" in b.label]
        _assert(button, "Load sample button not found")
        button[0].click().run()
        assert_clean(at)

    record("live prediction: Load sample populates inputs", load_sample)

    def analyze():
        button = [b for b in at.button if "Analyze Grid Condition" in b.label]
        _assert(button, "Analyze button not found")
        button[0].click().run()
        assert_clean(at)
        blob = " ".join(m.value for m in at.markdown).upper()
        _assert("GRID STATUS" in blob, "status banner missing from output")
        _assert(
            "FAULT DETECTED" in blob or "NORMAL OPERATION" in blob,
            "no grid verdict rendered",
        )

    record("live prediction: Analyze renders a verdict", analyze)

    def analyze_dl():
        at.sidebar.selectbox[0].set_value("Deep Learning (Dense MLP)").run()
        assert_clean(at)
        button = [b for b in at.button if "Analyze Grid Condition" in b.label]
        button[0].click().run()
        assert_clean(at)

    record("live prediction: Analyze works on the Keras model", analyze_dl)

    def explorer():
        at.sidebar.radio[0].set_value("Data Explorer").run()
        assert_clean(at)
        dataset_radio = [r for r in at.radio if "detect_dataset.csv" in (r.options or [])]
        _assert(dataset_radio, "dataset radio not found")
        dataset_radio[0].set_value("detect_dataset.csv").run()
        assert_clean(at)

    record("data explorer: detect_dataset.csv renders", explorer)

    def insights_slider():
        at.sidebar.radio[0].set_value("Model Insights")
        at.sidebar.selectbox[0].set_value("Random Forest").run()
        assert_clean(at)
        _assert(at.slider, "feature-importance slider not found")
        at.slider[0].set_value(27).run()
        assert_clean(at)

    record("model insights: feature-importance slider works", insights_slider)

    def batch_empty_state():
        at.sidebar.radio[0].set_value("Batch Prediction").run()
        assert_clean(at)
        _assert(at.file_uploader, "file uploader not rendered")
        _assert(
            any("template" in b.label.lower() for b in at.download_button),
            "template download button missing",
        )

    record("batch prediction: uploader and template render", batch_empty_state)

    print("\n" + "=" * 66)
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    for name, err in FAILED:
        print(f"    FAILED: {name}\n            {err}")
    print("=" * 66)
    return 1 if FAILED else 0


def _assert(condition, message: str = "assertion failed"):
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    sys.exit(main())
