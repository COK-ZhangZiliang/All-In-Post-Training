from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Any


DOLLY_DATASET = "databricks/databricks-dolly-15k"
QWEN_MODELSCOPE_MODEL = "Qwen/Qwen3.5-2B-Base"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run LoRA or full-parameter SFT on a real causal LM and real instruction data"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for rank0 artifacts")
    parser.add_argument("--run-id", default="real-sft", help="Run id for reports")
    parser.add_argument("--model-name", default=QWEN_MODELSCOPE_MODEL, help="Model id or path")
    parser.add_argument(
        "--model-source",
        choices=("modelscope", "huggingface", "local"),
        default="modelscope",
        help="How to resolve --model-name",
    )
    parser.add_argument("--dataset-name", default=DOLLY_DATASET, help="Hugging Face dataset id")
    parser.add_argument(
        "--dataset-file",
        default=None,
        help="Optional local JSONL or JSON array instruction dataset file",
    )
    parser.add_argument("--dataset-split", default="train", help="Dataset split to load")
    parser.add_argument("--train-samples", type=int, default=128, help="Number of training rows")
    parser.add_argument("--eval-samples", type=int, default=32, help="Number of evaluation rows")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum token length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT epochs")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional maximum optimizer steps before stopping the run",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--eval-every", type=int, default=8, help="Optimizer-step eval interval")
    parser.add_argument(
        "--tuning-mode",
        choices=("lora", "full"),
        default="lora",
        help="Whether to train LoRA adapters or all model parameters",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names",
    )
    parser.add_argument("--backend", default="gloo", help="Torch distributed backend")
    parser.add_argument(
        "--gradient-sync",
        choices=("none", "cpu-allreduce", "deepspeed-zero3"),
        default="cpu-allreduce",
        help="Gradient synchronization strategy for distributed runs",
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=("final", "none"),
        default="final",
        help="Whether to save the final model/adapter checkpoint in addition to metrics",
    )
    args = parser.parse_args(argv)

    report = run_real_sft(
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        model_name=args.model_name,
        model_source=args.model_source,
        dataset_name=args.dataset_name,
        dataset_file=Path(args.dataset_file) if args.dataset_file else None,
        dataset_split=args.dataset_split,
        train_samples=max(1, args.train_samples),
        eval_samples=max(1, args.eval_samples),
        max_seq_length=max(64, args.max_seq_length),
        epochs=max(1, args.epochs),
        max_steps=max(1, args.max_steps) if args.max_steps is not None else None,
        batch_size=max(1, args.batch_size),
        learning_rate=args.learning_rate,
        eval_every=max(1, args.eval_every),
        tuning_mode=args.tuning_mode,
        lora_r=max(1, args.lora_r),
        lora_alpha=max(1, args.lora_alpha),
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(
            item.strip() for item in args.lora_target_modules.split(",") if item.strip()
        ),
        backend=args.backend,
        gradient_sync=args.gradient_sync,
        checkpoint_policy=args.checkpoint_policy,
    )
    if report["rank"] == 0:
        first_eval = report["eval_history"][0]["eval_loss"]
        final_eval = report["eval_history"][-1]["eval_loss"]
        print(
            "real_sft_done "
            f"world_size={report['world_size']} "
            f"steps={report['steps']} "
            f"initial_eval_loss={first_eval:.6f} "
            f"final_eval_loss={final_eval:.6f}"
        )
    return 0


def run_real_sft(
    *,
    output_dir: Path,
    run_id: str,
    model_name: str,
    model_source: str,
    dataset_name: str,
    dataset_file: Path | None,
    dataset_split: str,
    train_samples: int,
    eval_samples: int,
    max_seq_length: int,
    epochs: int,
    max_steps: int | None,
    batch_size: int,
    learning_rate: float,
    eval_every: int,
    tuning_mode: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
    backend: str,
    gradient_sync: str,
    checkpoint_policy: str,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    from transformers import AutoModelForCausalLM, AutoTokenizer

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_deepspeed = gradient_sync == "deepspeed-zero3"
    if distributed and not use_deepspeed:
        dist.init_process_group(backend=backend)

    cuda_available = bool(torch.cuda.is_available())
    device = torch.device(f"cuda:{local_rank}" if cuda_available else "cpu")
    if cuda_available:
        torch.cuda.set_device(device)
    seed = 20260624
    torch.manual_seed(seed + rank)

    model_path = resolve_model_path(model_name, model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = select_torch_dtype(torch)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if should_enable_gradient_checkpointing(gradient_sync) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
    if tuning_mode == "lora":
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_target_modules),
        )
        model = get_peft_model(model, lora_config)
    elif tuning_mode != "full":
        raise ValueError(f"unsupported tuning mode: {tuning_mode}")
    model.to(device)
    model.train()

    raw_dataset = load_raw_instruction_dataset(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        dataset_split=dataset_split,
    )
    rows = load_instruction_rows(
        raw_dataset,
        train_samples=train_samples,
        eval_samples=eval_samples,
        seed=seed,
    )
    train_dataset = build_sft_examples(tokenizer, rows["train"], max_seq_length)
    eval_dataset = build_sft_examples(tokenizer, rows["eval"], max_seq_length)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda batch: collate_sft_examples(batch, pad_token_id=int(tokenizer.pad_token_id)),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        collate_fn=lambda batch: collate_sft_examples(batch, pad_token_id=int(tokenizer.pad_token_id)),
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model_engine = None
    if use_deepspeed:
        import deepspeed

        disable_deepspeed_nvtx_if_needed(deepspeed)
        deepspeed.init_distributed(dist_backend=backend)
        trainable_parameter_count = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        deepspeed_config = build_deepspeed_zero3_config(
            batch_size=batch_size,
            world_size=world_size,
            learning_rate=learning_rate,
        )
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=trainable_parameters,
            config=deepspeed_config,
        )
        model = model_engine
        dist = torch.distributed
    else:
        trainable_parameter_count = int(sum(parameter.numel() for parameter in trainable_parameters))
        optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)

    start = time.time()
    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    step = 0

    eval_history.append(
        evaluate_model(
            torch=torch,
            dist=dist,
            model=model,
            loader=eval_loader,
            device=device,
            distributed=distributed,
            world_size=world_size,
            step=step,
            epoch=0,
        )
    )
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if model_engine is not None:
                model_engine.backward(loss)
                model_engine.step()
                grad_norm = 0.0
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if distributed and gradient_sync == "cpu-allreduce":
                    average_trainable_gradients_via_cpu(torch, dist, model, world_size)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()

            reduced_loss = loss.detach().clone()
            if distributed:
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss = reduced_loss / world_size
            train_history.append(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "train_loss": float(reduced_loss.cpu().item()),
                    "local_train_loss": float(loss.detach().cpu().item()),
                    "grad_norm": float(_to_float(grad_norm)),
                }
            )
            if step % eval_every == 0:
                eval_history.append(
                    evaluate_model(
                        torch=torch,
                        dist=dist,
                        model=model,
                        loader=eval_loader,
                        device=device,
                        distributed=distributed,
                        world_size=world_size,
                        step=step,
                        epoch=epoch + 1,
                    )
                )
                model.train()
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    if not eval_history or eval_history[-1]["step"] != step:
        eval_history.append(
            evaluate_model(
                torch=torch,
                dist=dist,
                model=model,
                loader=eval_loader,
                device=device,
                distributed=distributed,
                world_size=world_size,
                step=step,
                epoch=epochs,
            )
        )
        model.train()

    report = {
        "run_id": run_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "distributed": distributed,
        "backend": backend,
        "gradient_sync": gradient_sync,
        "checkpoint_policy": checkpoint_policy,
        "device": str(device),
        "cuda_available": cuda_available,
        "model_name": model_name,
        "model_source": model_source,
        "resolved_model_path": str(model_path),
        "dataset_name": dataset_name,
        "dataset_file": str(dataset_file) if dataset_file else None,
        "dataset_split": dataset_split,
        "train_samples": len(rows["train"]),
        "eval_samples": len(rows["eval"]),
        "max_seq_length": max_seq_length,
        "epochs": epochs,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "tuning_mode": tuning_mode,
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": list(lora_target_modules),
            "trainable_parameters": trainable_parameter_count,
        },
        "steps": step,
        "duration_seconds": round(time.time() - start, 3),
        "train_history": train_history,
        "eval_history": eval_history,
    }

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        write_real_sft_artifacts(output_dir, report)
        (output_dir / "dataset_preview.json").write_text(
            json.dumps(
                {
                    "train": rows["train"][:5],
                    "eval": rows["eval"][:5],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if checkpoint_policy == "final" and model_engine is None:
            checkpoint_dir = output_dir / ("adapter" if tuning_mode == "lora" else "model")
            model.save_pretrained(checkpoint_dir)

    if checkpoint_policy == "final" and model_engine is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "model"
        model_engine.save_checkpoint(str(checkpoint_dir), tag="global_step_final")
    elif checkpoint_policy != "none":
        raise ValueError(f"unsupported checkpoint policy: {checkpoint_policy}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return report


def build_deepspeed_zero3_config(*, batch_size: int, world_size: int, learning_rate: float) -> dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": batch_size,
        "train_batch_size": batch_size * world_size,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": False,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": False,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },
        "steps_per_print": 100000,
        "wall_clock_breakdown": False,
    }


def disable_deepspeed_nvtx_if_needed(deepspeed: Any) -> None:
    nvtx = getattr(getattr(deepspeed, "utils", None), "nvtx", None)
    if nvtx is None:
        return
    nvtx._range_push = lambda *args, **kwargs: None
    nvtx._range_pop = lambda *args, **kwargs: None


def should_enable_gradient_checkpointing(gradient_sync: str) -> bool:
    return gradient_sync != "deepspeed-zero3"


def resolve_model_path(model_name: str, model_source: str) -> str:
    if model_source == "local":
        return model_name
    if model_source == "modelscope":
        from modelscope import snapshot_download

        return snapshot_download(model_name)
    return model_name


def load_raw_instruction_dataset(
    *, dataset_name: str, dataset_file: Path | None, dataset_split: str
) -> Any:
    if dataset_file is not None:
        return load_instruction_dataset_file(dataset_file)

    from datasets import load_dataset

    return load_dataset(dataset_name, split=dataset_split)


def load_instruction_dataset_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"instruction dataset file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON instruction dataset file must contain a top-level array")
        return [require_json_object(item, path, index + 1) for index, item in enumerate(data)]

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        rows.append(require_json_object(item, path, line_number))
    if not rows:
        raise ValueError(f"instruction dataset file is empty: {path}")
    return rows


def require_json_object(item: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{path}:{line_number} must be a JSON object")
    return item


def load_instruction_rows(dataset: Any, train_samples: int, eval_samples: int, seed: int) -> dict[str, list[dict[str, str]]]:
    if hasattr(dataset, "shuffle") and hasattr(dataset, "select"):
        dataset = dataset.shuffle(seed=seed)
        total = min(len(dataset), train_samples + eval_samples)
        selected = dataset.select(range(total))
    else:
        selected = deterministic_shuffle(list(dataset), seed)[: train_samples + eval_samples]
    rows = [normalize_instruction_row(row) for row in selected]
    return {
        "train": rows[: min(train_samples, len(rows))],
        "eval": rows[min(train_samples, len(rows)) :],
    }


def deterministic_shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    import random

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def normalize_instruction_row(row: dict[str, Any]) -> dict[str, str]:
    messages = row.get("messages")
    instruction = str(
        row.get("instruction") or row.get("prompt") or extract_instruction_from_messages(messages) or ""
    ).strip()
    context = str(row.get("context") or row.get("input") or "").strip()
    response = str(
        row.get("response")
        or row.get("output")
        or row.get("completion")
        or extract_response_from_messages(messages)
        or ""
    ).strip()
    category = str(row.get("category") or row.get("source") or "").strip()
    if not instruction or not response:
        raise ValueError("instruction dataset row must contain non-empty instruction and response")
    return {
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": category,
    }


def extract_instruction_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def extract_response_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def format_prompt(row: dict[str, str]) -> str:
    if row["context"]:
        return (
            "### Instruction:\n"
            f"{row['instruction']}\n\n"
            "### Input:\n"
            f"{row['context']}\n\n"
            "### Response:\n"
        )
    return "### Instruction:\n" f"{row['instruction']}\n\n" "### Response:\n"


def build_sft_dataset(tokenizer: Any, rows: list[dict[str, str]], max_seq_length: int) -> tuple[Any, Any, Any]:
    import torch

    examples = build_sft_examples(tokenizer, rows, max_seq_length)
    padded = collate_sft_examples(examples, pad_token_id=int(tokenizer.pad_token_id), pad_to=max_seq_length)
    return padded["input_ids"], padded["attention_mask"], padded["labels"]


def build_sft_examples(
    tokenizer: Any,
    rows: list[dict[str, str]],
    max_seq_length: int,
) -> list[dict[str, list[int]]]:
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    eos = tokenizer.eos_token or ""
    for row in rows:
        prompt_ids = tokenizer(format_prompt(row), add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(row["response"] + eos, add_special_tokens=False)["input_ids"]
        prompt_ids, response_ids = truncate_for_supervised_response(
            prompt_ids,
            response_ids,
            max_seq_length,
        )
        ids = prompt_ids + response_ids
        row_labels = ([-100] * len(prompt_ids)) + response_ids
        input_ids.append(ids)
        labels.append(row_labels)
    return [
        {
            "input_ids": ids,
            "labels": row_labels,
        }
        for ids, row_labels in zip(input_ids, labels)
    ]


def collate_sft_examples(
    examples: list[dict[str, list[int]]],
    *,
    pad_token_id: int,
    pad_to: int | None = None,
) -> dict[str, Any]:
    import torch

    if not examples:
        raise ValueError("cannot collate an empty SFT batch")
    max_length = pad_to or max(len(example["input_ids"]) for example in examples)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        ids = list(example["input_ids"])
        row_labels = list(example["labels"])
        if len(ids) > max_length:
            ids = ids[:max_length]
            row_labels = row_labels[:max_length]
        padding = max_length - len(ids)
        input_ids.append(ids + ([pad_token_id] * padding))
        attention_masks.append(([1] * len(ids)) + ([0] * padding))
        labels.append(row_labels + ([-100] * padding))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def truncate_for_supervised_response(
    prompt_ids: list[int],
    response_ids: list[int],
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    if len(prompt_ids) + len(response_ids) <= max_seq_length:
        return prompt_ids, response_ids
    if not response_ids:
        raise ValueError("SFT row must contain at least one response token")
    response_budget = min(len(response_ids), max(1, max_seq_length // 2))
    prompt_budget = max_seq_length - response_budget
    return prompt_ids[:prompt_budget], response_ids[:response_budget]


def evaluate_model(
    *,
    torch: Any,
    dist: Any,
    model: Any,
    loader: Any,
    device: Any,
    distributed: bool,
    world_size: int,
    step: int,
    epoch: int,
) -> dict[str, Any]:
    del world_size
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            token_count = (labels != -100).sum().to(dtype=torch.float32)
            if int(token_count.cpu().item()) == 0:
                continue
            total_loss += outputs.loss.detach() * token_count
            total_tokens += token_count
    if distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    eval_loss = total_loss / total_tokens.clamp_min(1.0)
    value = float(eval_loss.cpu().item())
    return {
        "step": step,
        "epoch": epoch,
        "eval_loss": value,
        "eval_perplexity": float(math.exp(min(value, 20.0))),
        "eval_tokens": int(total_tokens.cpu().item()),
    }


def average_trainable_gradients_via_cpu(torch: Any, dist: Any, model: Any, world_size: int) -> None:
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        cpu_grad = parameter.grad.detach().cpu()
        dist.all_reduce(cpu_grad, op=dist.ReduceOp.SUM)
        cpu_grad /= world_size
        parameter.grad.copy_(cpu_grad.to(parameter.grad.device))


def write_real_sft_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "trainer_state.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_metric_csv(output_dir / "train_history.csv", report["train_history"])
    write_metric_csv(output_dir / "eval_history.csv", report["eval_history"])
    (output_dir / "sft_eval_curve.svg").write_text(
        render_real_sft_curve_svg(report["train_history"], report["eval_history"], report["run_id"]),
        encoding="utf-8",
    )


def write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_real_sft_curve_svg(
    train_history: list[dict[str, Any]],
    eval_history: list[dict[str, Any]],
    run_id: str,
) -> str:
    width = 1040
    height = 600
    margin_left = 78
    margin_right = 36
    margin_top = 76
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    series = [
        (
            "train_ma8",
            rolling_average_points(
                [(int(item["step"]), float(item["train_loss"])) for item in train_history],
                window=8,
            ),
            "#2563eb",
        ),
        ("eval", [(int(item["step"]), float(item["eval_loss"])) for item in eval_history], "#dc2626"),
    ]
    all_points = [point for _, points, _ in series for point in points if math.isfinite(point[1])]
    if not all_points:
        all_points = [(0, 0.0)]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5

    def x_pos(step: int) -> float:
        return margin_left + ((step - x_min) / (x_max - x_min)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + ((y_max - value) / (y_max - y_min)) * plot_height

    grid_lines = []
    y_labels = []
    for index in range(5):
        ratio = index / 4
        y = margin_top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" class="grid" />'
        )
        y_labels.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'class="axis-label">{value:.3f}</text>'
        )
    x_labels = []
    for index in range(5):
        ratio = index / 4
        x = margin_left + ratio * plot_width
        step = round(x_min + ratio * (x_max - x_min))
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 34}" text-anchor="middle" '
            f'class="axis-label">{step}</text>'
        )

    paths = []
    legend = []
    for index, (name, points, color) in enumerate(series):
        filtered = [(step, value) for step, value in points if math.isfinite(value)]
        if filtered:
            polyline = " ".join(f"{x_pos(step):.2f},{y_pos(value):.2f}" for step, value in filtered)
            paths.append(f'<polyline points="{polyline}" class="curve" stroke="{color}" />')
            last_step, last_value = filtered[-1]
            paths.append(
                f'<circle cx="{x_pos(last_step):.2f}" cy="{y_pos(last_value):.2f}" '
                f'r="5" fill="{color}" />'
            )
        legend_y = 42 + index * 22
        legend.append(f'<line x1="{width - 220}" y1="{legend_y}" x2="{width - 185}" y2="{legend_y}" stroke="{color}" stroke-width="4" />')
        legend.append(f'<text x="{width - 176}" y="{legend_y + 5}" class="legend">{name}</text>')

    first_eval = eval_history[0]["eval_loss"] if eval_history else math.nan
    final_eval = eval_history[-1]["eval_loss"] if eval_history else math.nan
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<style>",
            "  .bg { fill: #f8fafc; }",
            "  .panel { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; }",
            "  .grid { stroke: #e2e8f0; stroke-width: 1; }",
            "  .axis { stroke: #475569; stroke-width: 1.4; }",
            "  .curve { fill: none; stroke-width: 3; stroke-linejoin: round; }",
            "  .title { fill: #0f172a; font: 700 24px system-ui, sans-serif; }",
            "  .subtitle { fill: #475569; font: 14px system-ui, sans-serif; }",
            "  .axis-label { fill: #475569; font: 12px system-ui, sans-serif; }",
            "  .caption { fill: #334155; font: 13px system-ui, sans-serif; }",
            "  .legend { fill: #334155; font: 13px system-ui, sans-serif; }",
            "</style>",
            '<rect class="bg" x="0" y="0" width="100%" height="100%" />',
            f'<rect class="panel" x="20" y="20" width="{width - 40}" height="{height - 40}" rx="8" />',
            f'<text x="{margin_left}" y="42" class="title">{_escape_xml(run_id)} SFT metrics</text>',
            f'<text x="{margin_left}" y="64" class="subtitle">initial_eval_loss={first_eval:.6f} '
            f'final_eval_loss={final_eval:.6f}</text>',
            *legend,
            *grid_lines,
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
            f'y2="{height - margin_bottom}" class="axis" />',
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
            f'x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis" />',
            *y_labels,
            *x_labels,
            *paths,
            f'<text x="{width / 2:.2f}" y="{height - 20}" text-anchor="middle" '
            'class="caption">optimizer step</text>',
            f'<text x="24" y="{height / 2:.2f}" transform="rotate(-90 24 {height / 2:.2f})" '
            'text-anchor="middle" class="caption">loss</text>',
            "</svg>",
        ]
    )


def select_torch_dtype(torch: Any) -> Any:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def rolling_average_points(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1:
        return points
    smoothed: list[tuple[int, float]] = []
    values: list[float] = []
    for step, value in points:
        values.append(value)
        if len(values) > window:
            values.pop(0)
        smoothed.append((step, sum(values) / len(values)))
    return smoothed


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
