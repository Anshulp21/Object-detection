#!/usr/bin/env python3
"""
Environment preflight check for Python 3.11 and this YOLOv5 project.
Safely verifies imports and versions needed by the pipeline.
Run: python env_check.py
"""
from __future__ import annotations

import sys
import platform
import shutil
import importlib


def ok(msg: str):
    print(f"[OK] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def fail(msg: str):
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def main():
    print("=== Preflight: Python & OS ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    print("\n=== Core ML Stack ===")
    try:
        import torch
        ok(f"torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")
    except Exception as e:
        fail(f"torch import failed: {e}")

    try:
        import torchvision
        ok(f"torchvision {torchvision.__version__}")
    except Exception as e:
        fail(f"torchvision import failed: {e}")

    print("\n=== YOLOv5 + Ultralytics Compatibility ===")
    try:
        import yolov5  # noqa: F401
        ok("yolov5 package present")
    except Exception as e:
        fail(f"yolov5 import failed: {e}")

    # Check ultralytics and legacy module path
    try:
        import ultralytics
        ok(f"ultralytics {ultralytics.__version__}")
        has_legacy = importlib.util.find_spec("ultralytics.yolo") is not None
        if has_legacy:
            ok("ultralytics.yolo module path available")
        else:
            warn("ultralytics.yolo module path NOT found (may break yolov5 7.0.12)")
    except Exception as e:
        fail(f"ultralytics import failed: {e}")

    # HF hub utils path used by yolov5 7.0.12
    try:
        from huggingface_hub.utils import _errors  # noqa: F401
        ok("huggingface_hub.utils._errors present")
    except Exception as e:
        fail(f"huggingface_hub import failed (needs 0.19.x): {e}")

    print("\n=== CV / Data Stack ===")
    for mod in ("cv2", "numpy", "pandas", "matplotlib", "skimage"):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "(no __version__)")
            ok(f"{mod} {ver}")
        except Exception as e:
            fail(f"{mod} import failed: {e}")

    print("\n=== CLI availability ===")
    yolov5_cli = shutil.which("yolov5")
    if yolov5_cli:
        ok(f"yolov5 CLI at {yolov5_cli}")
    else:
        warn("yolov5 CLI not on PATH (train.py will fallback)")

    print("\nAll required dependencies are present. You are good to train.\n")


if __name__ == "__main__":
    main()
