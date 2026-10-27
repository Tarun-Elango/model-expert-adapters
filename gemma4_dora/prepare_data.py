"""Run the shared data preparation script for both projects."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent.parent / "gemma4_lora" / "prepare_data.py"
    runpy.run_path(str(script), run_name="__main__")
