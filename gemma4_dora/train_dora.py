"""Train a DoRA expert using the shared training implementation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gemma4_lora"))

from train_lora import main  # noqa: E402

if __name__ == "__main__":
    main(use_dora=True, project_root=ROOT)
