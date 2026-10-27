"""Use the same shared base model and downloader as the LoRA project."""
import sys
from pathlib import Path

LORA_PROJECT = Path(__file__).resolve().parents[2] / "gemma4_lora"
sys.path.insert(0, str(LORA_PROJECT))

from model.base_model import download_model, load_base_model  # noqa: E402,F401

if __name__ == "__main__":
    download_model()
