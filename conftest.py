import pathlib
import sys

# Ensure the repo root (src/, tests/) and app/ (demo_rank, presets) are importable
# regardless of how pytest is invoked.
_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "app"))
