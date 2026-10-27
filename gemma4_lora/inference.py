"""Interactive adapter switching with a single loaded base model."""
from pathlib import Path

import torch
from peft import PeftModel

from model.base_model import load_base_model

ROOT = Path(__file__).resolve().parent


def main(project_root=ROOT):
    processor, base = load_base_model()
    model = None
    loaded = set()
    selected = None
    print("Commands: /code, /health, /base, /quit. Each prompt is independent.")
    while True:
        try:
            prompt = input(f"[{selected or 'base'}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt == "/base":
            selected = None
            continue
        if prompt in ("/code", "/health"):
            name = prompt[1:]
            path = project_root / "adapters" / name
            if not (path / "adapter_config.json").is_file():
                print(f"Adapter not found at {path}; train it first.")
                continue
            if name not in loaded:
                if model is None:
                    model = PeftModel.from_pretrained(base, path, adapter_name=name, is_trainable=False)
                else:
                    model.load_adapter(path, adapter_name=name, is_trainable=False)
                loaded.add(name)
            model.set_adapter(name)
            model.eval()
            selected = name
            continue
        if prompt.startswith("/"):
            print("Unknown command. Use /code, /health, /base or /quit.")
            continue
        active = model if model is not None else base
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True, enable_thinking=False,
        ).to(base.device)
        with torch.inference_mode():
            if model is not None and selected is None:
                with model.disable_adapter():
                    result = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            else:
                result = active.generate(**inputs, max_new_tokens=512, do_sample=False)
        print(processor.decode(result[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
