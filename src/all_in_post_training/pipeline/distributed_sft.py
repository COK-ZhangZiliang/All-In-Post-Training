from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any


SFT_TEXTS = (
    (
        "Explain supervised fine-tuning in one sentence.",
        "Supervised fine-tuning trains a base model on curated prompt-response pairs.",
    ),
    (
        "What comes after SFT in this project?",
        "The project trains independent domain RL specialists before OPD fusion.",
    ),
    (
        "Name two post-training quality gates.",
        "Dataset lineage and evaluation readiness are required quality gates.",
    ),
    (
        "Why keep generated checkpoints out of Git?",
        "Generated checkpoints are large runtime artifacts and should stay in ignored outputs.",
    ),
    (
        "What does OPD do?",
        "OPD fuses specialist policies into a single deployable student policy.",
    ),
    (
        "Why use distributed training?",
        "Distributed training parallelizes gradient computation across multiple GPUs.",
    ),
    (
        "What does a readiness audit protect?",
        "It prevents real training before model, license, data, and evaluation gates pass.",
    ),
    (
        "What is the current base model target?",
        "The initial target is Qwen/Qwen3.5-2B-Base after revision and license review.",
    ),
)


class TinyCausalLM:
    """Factory wrapper to keep torch imports inside runtime code paths."""

    @staticmethod
    def build(torch: Any, vocab_size: int, hidden_size: int, sequence_length: int) -> Any:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.token_embedding = torch.nn.Embedding(vocab_size, hidden_size)
                self.position_embedding = torch.nn.Embedding(sequence_length, hidden_size)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.0,
                    batch_first=True,
                )
                self.transformer = torch.nn.TransformerEncoder(layer, num_layers=2)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids: Any) -> Any:
                batch_size, seq_len = input_ids.shape
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                    diagonal=1,
                )
                hidden = self.transformer(hidden, mask=causal_mask)
                return self.lm_head(hidden)

        return Model()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny distributed SFT training job")
    parser.add_argument("--output-dir", required=True, help="Output directory for rank0 artifacts")
    parser.add_argument("--run-id", default="distributed-sft", help="Run id for reports")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fixture-data epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-rank batch size")
    parser.add_argument("--sequence-length", type=int, default=96, help="Token sequence length")
    parser.add_argument("--hidden-size", type=int, default=64, help="Tiny model hidden size")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="AdamW learning rate")
    parser.add_argument("--backend", default=None, help="Torch distributed backend override")
    args = parser.parse_args(argv)

    report = run_distributed_sft(
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        sequence_length=max(16, args.sequence_length),
        hidden_size=max(16, args.hidden_size),
        learning_rate=args.learning_rate,
        backend=args.backend,
    )
    if report["rank"] == 0:
        print(
            "distributed_sft_done "
            f"world_size={report['world_size']} "
            f"steps={report['steps']} "
            f"final_loss={report['final_loss']:.6f}"
        )
    return 0


def run_distributed_sft(
    output_dir: Path,
    run_id: str,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    learning_rate: float,
    backend: str | None = None,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cuda_available = bool(torch.cuda.is_available())
    actual_backend = backend or ("nccl" if cuda_available else "gloo")

    if distributed:
        dist.init_process_group(backend=actual_backend)
    device = torch.device(f"cuda:{local_rank}" if cuda_available else "cpu")
    if cuda_available:
        torch.cuda.set_device(device)

    torch.manual_seed(20260624 + rank)
    input_ids, labels = build_sft_tensors(torch, sequence_length)
    dataset = TensorDataset(input_ids, labels)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=20260624,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)

    vocab_size = 258
    model = TinyCausalLM.build(torch, vocab_size, hidden_size, sequence_length).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if cuda_available else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start = time.time()
    losses: list[dict[str, Any]] = []
    step = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch_inputs, batch_labels in loader:
            step += 1
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                batch_labels.reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            reduced_loss = loss.detach().clone()
            if distributed:
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss = reduced_loss / world_size
            losses.append(
                {
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": float(reduced_loss.cpu().item()),
                    "local_loss": float(loss.detach().cpu().item()),
                    "grad_norm": float(_to_float(grad_norm)),
                }
            )

    final_loss = losses[-1]["loss"] if losses else math.nan
    report = {
        "run_id": run_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "distributed": distributed,
        "backend": actual_backend,
        "device": str(device),
        "cuda_available": cuda_available,
        "epochs": epochs,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "steps": step,
        "final_loss": final_loss,
        "duration_seconds": round(time.time() - start, 3),
        "losses": losses,
    }

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if distributed else model
        torch.save(model_to_save.state_dict(), output_dir / "model_state.pt")
        (output_dir / "trainer_state.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "sft_fixture.json").write_text(
            json.dumps(
                [{"prompt": prompt, "response": response} for prompt, response in SFT_TEXTS],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return report


def build_sft_tensors(torch: Any, sequence_length: int) -> tuple[Any, Any]:
    encoded = [_encode_example(prompt, response, sequence_length) for prompt, response in SFT_TEXTS]
    input_ids = torch.tensor(encoded, dtype=torch.long)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = 257
    return input_ids, labels


def _encode_example(prompt: str, response: str, sequence_length: int) -> list[int]:
    text = f"User: {prompt}\nAssistant: {response}"
    tokens = [min(byte, 255) for byte in text.encode("utf-8")]
    tokens.append(256)
    if len(tokens) < sequence_length:
        tokens.extend([257] * (sequence_length - len(tokens)))
    return tokens[:sequence_length]


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
