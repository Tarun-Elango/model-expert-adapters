"""Test and switch between DoRA adapters with one loaded base model."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gemma4_lora"))

from inference import main  # noqa: E402

if __name__ == "__main__":
    main(project_root=ROOT)
