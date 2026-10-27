"""Train one text LoRA expert; only adapter files are saved."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from model.base_model import load_base_model

ROOT = Path(__file__).resolve().parent


class Examples(Dataset):
    def __init__(self, path, processor, max_length):
        self.items = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompt = [{"role": "user", "content": row["prompt"]}]
                conversation = prompt + [{"role": "assistant", "content": row["answer"]}]
                prefix = processor.apply_chat_template(
                    prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
                full = processor.apply_chat_template(
                    conversation, tokenize=True, add_generation_prompt=False, enable_thinking=False
                )
                if full[:len(prefix)] != prefix:
                    raise ValueError("Chat template prompt is not a prefix of the completed conversation")
                full = full[:max_length]
                if len(full) <= len(prefix):
                    continue
                self.items.append({"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]})
        if not self.items:
            raise ValueError("No usable examples; increase --max-length or check your data")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collator(pad_id):
    def collate(items):
        length = max(len(item["input_ids"]) for item in items)
        return {
            "input_ids": torch.tensor([item["input_ids"] + [pad_id] * (length - len(item["input_ids"])) for item in items]),
            "attention_mask": torch.tensor([[1] * len(item["input_ids"]) + [0] * (length - len(item["input_ids"])) for item in items]),
            "labels": torch.tensor([item["labels"] + [-100] * (length - len(item["labels"])) for item in items]),
        }
    return collate


def main(use_dora=False, project_root=ROOT):
    parser = argparse.ArgumentParser()
    parser.add_argument("expert", choices=["code", "health"])
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    data_path = ROOT.parent / "shared" / "data" / f"{args.expert}.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Run python prepare_data.py {args.expert} first")
    output = project_root / "adapters" / args.expert
    if output.exists():
        raise FileExistsError(f"{output} already exists; move it aside before training again")
    processor, base = load_base_model()
    data = Examples(data_path, processor, args.max_length)
    # Use fully qualified module names to avoid modifying vision/audio towers.
    targets = [name for name, module in base.named_modules()
               if "language_model" in name and name.endswith((".q_proj", ".v_proj"))
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("No language-model q_proj/v_proj layers found; check your Transformers version")
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=targets,
        use_dora=use_dora,
    ))
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            raise RuntimeError(f"Unexpected trainable base parameter: {name}")
    model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()
    options = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, logging_steps=10, save_strategy="no",
        report_to="none", remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=options, train_dataset=data,
                      data_collator=collator(processor.tokenizer.pad_token_id))
    trainer.train()
    model.save_pretrained(output, safe_serialization=True)
    print(f"Saved {args.expert} adapter to {output}")


if __name__ == "__main__":
    main()
