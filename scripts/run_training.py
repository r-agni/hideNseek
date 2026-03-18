"""Wrapper to run training and capture all output."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

result = subprocess.run(
    [sys.executable, "scripts/train_rl.py", "--headless", "--num-envs", "1", "--steps", "500", "--record"],
    cwd=REPO_ROOT,
    capture_output=False,
)
sys.exit(result.returncode)
