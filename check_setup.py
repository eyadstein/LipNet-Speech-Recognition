"""Environment check script for the LipNet project.

Run this before using predict.py, app.py, or the training notebook to
catch common setup issues early with clear error messages, instead of
cryptic crashes partway through model loading or inference.
"""

import sys
import os

REQUIRED_PYTHON = (3, 11)
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoint.weights.h5")

REQUIRED_PACKAGES = {
    "tensorflow": "2.21.0",
    "cv2": None,  # opencv-python, version check skipped (import name differs from package name)
    "numpy": None,
    "gradio": None,
}


def check_python_version():
    major, minor = sys.version_info[:2]
    if (major, minor) != REQUIRED_PYTHON:
        print(f"[WARNING] Python {major}.{minor} detected. This project was built and tested "
              f"on Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}. "
              f"TensorFlow does not support Python 3.13+ as of this writing.")
        return False
    print(f"[OK] Python {major}.{minor}")
    return True


def check_package(import_name, expected_version=None):
    try:
        module = __import__(import_name)
    except ImportError:
        print(f"[MISSING] Package for '{import_name}' is not installed. "
              f"Run: pip install -r requirements.txt")
        return False

    version = getattr(module, "__version__", "unknown")
    if expected_version and version != expected_version:
        print(f"[WARNING] {import_name} version {version} found, expected {expected_version}. "
              f"This may cause compatibility issues (see README/EVALUATION.md for known version-sensitive bugs).")
        return False

    print(f"[OK] {import_name} {version}")
    return True


def check_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[MISSING] Model checkpoint not found at: {CHECKPOINT_PATH}\n"
              f"          Train the model using LipNet.ipynb first, or place a trained "
              f"checkpoint.weights.h5 in the models/ folder.")
        return False

    size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    print(f"[OK] Checkpoint found ({size_mb:.1f} MB)")
    return True


def main():
    print("=" * 60)
    print("LipNet Environment Check")
    print("=" * 60)

    checks = [check_python_version()]

    for import_name, expected_version in REQUIRED_PACKAGES.items():
        checks.append(check_package(import_name, expected_version))

    checks.append(check_checkpoint())

    print("=" * 60)
    if all(checks):
        print("All checks passed. Ready to run predict.py or app.py.")
    else:
        print("Some checks failed or gave warnings above. "
              "The project may still work, but see the messages above first.")
    print("=" * 60)


if __name__ == "__main__":
    main()
