import pathlib
import sys

# Ensure the repo root is importable (src/, tests/) regardless of how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
