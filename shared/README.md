# Shared original inputs

Both `gemma4_lora` and `gemma4_dora` use `model/gemma-4-e4b-it/` for the downloaded base model and `data/code.jsonl` and `data/health.jsonl` for prepared examples. These folders are created when the download and preparation scripts are run. Training reads these inputs without changing them, freezes the base parameters, and saves only adapters in each project's own `adapters/` directory. Running data preparation again can add examples to the shared dataset.
