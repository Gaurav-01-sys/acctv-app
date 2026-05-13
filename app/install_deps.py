"""Auto-install dependencies at import time. Import this before any other app module."""
import subprocess
import sys
import os
from pathlib import Path

_LOG = Path("/tmp/install.log")

def _log(msg):
    with _LOG.open("a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

def ensure_packages():
    """Install missing packages in the current Python environment."""
    required = {
        "numpy": "numpy==2.2.6",
        "cv2": "opencv-python-headless==4.10.0.84",
        "sklearn": "scikit-learn==1.7.1",
        "onnxruntime": "onnxruntime==1.21.1",
        "huggingface_hub": "huggingface_hub>=0.25.0",
    }
    
    missing = []
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    
    if not missing:
        _log("All packages already installed.")
        return True
    
    _log(f"Missing packages: {missing}")
    _log(f"Python: {sys.executable}")
    _log(f"Installing with pip...")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + missing,
        capture_output=True, text=True, timeout=600
    )
    _log(f"STDOUT: {result.stdout[-3000:]}")
    if result.stderr:
        _log(f"STDERR: {result.stderr[-2000:]}")
    _log(f"Exit code: {result.returncode}")
    
    if result.returncode != 0:
        _log("WARNING: pip install failed!")
        return False
    
    # Verify
    still_missing = []
    for import_name in required:
        try:
            __import__(import_name)
            _log(f"  ✓ {import_name}")
        except ImportError:
            still_missing.append(import_name)
            _log(f"  ✗ {import_name} still missing!")
    
    return len(still_missing) == 0

# Run on import
ensure_packages()
