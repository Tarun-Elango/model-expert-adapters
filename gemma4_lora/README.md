# Health LoRA training

Run these commands from the repository root on a machine with an NVIDIA GPU,
driver, Python 3.11 or newer, sufficient GPU memory, and internet access. Install a CUDA-enabled
PyTorch build appropriate for the machine before installing the remaining
requirements. Verify `torch.cuda.is_available()` before training.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# Install CUDA-enabled PyTorch using the instructions for your system at pytorch.org.
python -m pip install -r gemma4_lora/requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python gemma4_lora/model/base_model.py
python gemma4_lora/prepare_data.py health --limit 2000
python gemma4_lora/train_lora.py health
```

The full model is loaded on one GPU, without quantization or CPU offloading;
GPU memory requirements have not been validated here. Reduce `--max-length`
if activations do not fit, or use a larger GPU. The model and data are stored
under `shared/`; the adapter is saved under `gemma4_lora/adapters/health`.
Training refuses to overwrite an existing adapter directory unless resuming.
The first 2,000 distinct examples are used by default; answers longer than
`--max-length` are truncated. There is currently no held-out evaluation.
Evaluate medical accuracy and safety before relying on the resulting adapter.

## Stop and resume

Trainer saves a checkpoint every 25 optimizer steps by default, keeping the two
most recent. Change the interval with `--save-steps N`. Checkpoints include the
adapter and Trainer state needed to resume; the base model and prepared data
must still be available. Press Ctrl+C to stop; progress since the last completed
checkpoint is lost. Stopping before the first checkpoint leaves nothing to
resume. To continue from the latest checkpoint, run:

```bash
python -u gemma4_lora/train_lora.py health --resume-from-checkpoint
```

To choose a checkpoint, pass its directory after the flag, for example
`--resume-from-checkpoint gemma4_lora/adapters/health/checkpoint-25`.
Use the same training arguments, model, and data as the original run. A fresh
run requires moving the existing adapter directory aside first. Checkpoints are
saved under the adapter directory, and the final adapter is saved there when
training completes.
