"""Wrapper to run training and capture all output."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "scripts/train_rl.py", "--headless", "--num-envs", "1", "--steps", "500", "--record"],
    cwd=r"C:\Users\agni_\Documents\hideNseek",
    capture_output=False,
)
sys.exit(result.returncode)
