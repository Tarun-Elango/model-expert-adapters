"""Download and deduplicate training examples only when explicitly run."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "shared" / "data"
SOURCES = {
    "code": "iamtarun/python_code_instructions_18k_alpaca",
    "health": "keivalya/MedQuad-MedicalQnADataset",
}


def normalize(value):
    return " ".join(str(value or "").split()).casefold()


def convert(row, expert):
    if expert == "code":
        prompt = str(row.get("instruction") or "").strip()
        extra = str(row.get("input") or "").strip()
        if extra:
            prompt += "\n\nInput:\n" + extra
        answer = str(row.get("output") or "").strip()
    else:
        prompt = str(row.get("Question") or "").strip()
        answer = str(row.get("Answer") or "").strip()
    return {"prompt": prompt, "answer": answer}


def prepare(expert, limit):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{expert}.jsonl"
    examples = []
    seen = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                key = (normalize(item["prompt"]), normalize(item["answer"]))
                if key not in seen:
                    seen.add(key)
                    examples.append(item)
    if limit and len(examples) >= limit:
        print(f"{path}: already contains {len(examples)} examples")
        return
    dataset = load_dataset(SOURCES[expert], split="train", streaming=True)
    for row in dataset:
        item = convert(row, expert)
        if not item["prompt"] or not item["answer"]:
            continue
        key = (normalize(item["prompt"]), normalize(item["answer"]))
        if key in seen:
            continue
        seen.add(key)
        examples.append(item)
        if limit and len(examples) >= limit:
            break
    # Replace atomically so an interrupted download does not corrupt existing data.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=DATA_DIR, delete=False) as f:
            temporary = f.name
            for item in examples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    print(f"{path}: {len(examples)} distinct prompt/answer pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expert", choices=["code", "health", "all"])
    parser.add_argument("--limit", type=int, default=2000, help="Total examples per expert; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    for name in (SOURCES if args.expert == "all" else [args.expert]):
        prepare(name, args.limit)
