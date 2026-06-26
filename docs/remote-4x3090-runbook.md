# Remote 4x3090 Runbook

This runbook records the current single-node 4-GPU training host setup.

## Access

```bash
ssh aitp-4x3090
```

Host summary:

- User: `ubuntu`
- Project root: `/mnt/data/All-In-Post-Training`
- Python environment: `/mnt/data/aitp-venv`
- Data disk: `/mnt/data`
- GPU inventory: 4 x NVIDIA GeForce RTX 3090, 24GB each

## Environment

Load the project environment:

```bash
source ~/.aitp_env
source /mnt/data/aitp-venv/bin/activate
cd /mnt/data/All-In-Post-Training
```

Important paths:

```bash
export AITP_ROOT=/mnt/data/All-In-Post-Training
export AITP_VENV=/mnt/data/aitp-venv
export AITP_DATA=/mnt/data
export HF_HOME=/mnt/data/cache/huggingface
export HF_HUB_CACHE=/mnt/data/cache/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/data/cache/transformers
export TORCH_HOME=/mnt/data/cache/torch
export MODELSCOPE_CACHE=/mnt/data/cache/modelscope
export PIP_CACHE_DIR=/mnt/data/cache/pip
```

## Validation

Run the control-plane and unit checks:

```bash
PYTHONPATH=src python -m all_in_post_training.cli pipeline validate --config examples/post_training_pipeline.json
PYTHONPATH=src python -m unittest tests.test_pipeline -v
PYTHONPATH=src python -m all_in_post_training.cli pipeline preflight --config examples/post_training_pipeline.json --run-id remote-4x3090-preflight-full --require-cuda --require-training-extras
```

Run the single-node 4-GPU NCCL smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 /mnt/data/nccl_4gpu_smoke.py
```

Run the project DDP fixture SFT smoke:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m all_in_post_training.pipeline.distributed_sft \
  --output-dir /mnt/data/runs/remote-4x3090-ddp-smoke/checkpoints/train_sft \
  --run-id remote-4x3090-ddp-smoke \
  --epochs 4 \
  --batch-size 2 \
  --sequence-length 128 \
  --hidden-size 128 \
  --learning-rate 0.003 \
  --gradient-sync ddp
```

Run the GPU TRL SFT dry-run pipeline:

```bash
PYTHONPATH=src python -m all_in_post_training.cli pipeline run \
  --config examples/post_training_pipeline.json \
  --run-id remote-4x3090-trl-sft-dry-run \
  --backend trl-sft-dry-run \
  --require-cuda
```

Run the real ModelScope LoRA SFT smoke:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m all_in_post_training.pipeline.real_sft \
  --output-dir /mnt/data/runs/remote-4x3090-modelscope-lora-smoke-v3/checkpoints/train_sft \
  --run-id remote-4x3090-modelscope-lora-smoke-v3 \
  --model-source modelscope \
  --model-name Qwen/Qwen3.5-2B-Base \
  --dataset-source modelscope \
  --dataset-name AI-ModelScope/alpaca-gpt4-data-en \
  --dataset-split train \
  --train-samples 16 \
  --eval-samples 8 \
  --max-seq-length 512 \
  --epochs 1 \
  --max-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --lr-scheduler cosine \
  --eval-every 1 \
  --logging-steps 1 \
  --train-eval-samples 4 \
  --tuning-mode lora \
  --backend nccl \
  --gradient-sync deepspeed-zero3 \
  --checkpoint-policy none
```

Expected smoke result:

- `world_size=4`
- `steps=1`
- `initial_eval_loss=1.116132`
- `final_eval_loss=1.115189`
- artifacts in `/mnt/data/runs/remote-4x3090-modelscope-lora-smoke-v3/checkpoints/train_sft/`

## Current Runtime

Verified package versions:

- `torch 2.6.0+cu124`
- `accelerate 1.14.0`
- `transformers 5.12.1`
- `datasets 5.0.0`
- `peft 0.19.1`
- `modelscope 1.37.1`
- `trl 1.7.0`
- `deepspeed 0.19.2`

## Notes

- Keep model weights, datasets, caches, and checkpoints under `/mnt/data`.
- The root disk is only about 40GB; do not use it for training artifacts.
- Single-node NCCL is verified on this host and should be preferred over the earlier two-container Gloo fallback.
- The real SFT runner uses a cached Arrow fallback for ModelScope datasets when the installed `modelscope` and `datasets` versions disagree on `verification_mode`.
- The real ZeRO-3 runner uses external `torch.optim.AdamW` to avoid DeepSpeed `CPUAdam` JIT compilation when system CUDA and PyTorch CUDA versions differ.
