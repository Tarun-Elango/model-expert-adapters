"""Download on explicit request and share one frozen base model per process."""
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it"
MODEL_DIR = Path(__file__).resolve().parents[2] / "shared" / "model" / "gemma-4-e4b-it"
_loaded = None


def download_model():
    # local_dir resumes partial downloads and reuses existing files on reruns.
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))
    print(f"Model available at {MODEL_DIR}")


def load_base_model():
    """Return the same processor/model pair on subsequent calls in this process."""
    global _loaded
    if _loaded is None:
        if not (MODEL_DIR / "config.json").is_file():
            raise FileNotFoundError("Model not downloaded. Run python model/base_model.py first.")
        processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, dtype="auto", device_map="auto"
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        _loaded = (processor, model)
    return _loaded


if __name__ == "__main__":
    download_model()
