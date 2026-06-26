from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare real SFT trainer_state.json artifacts")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional output CSV path. Defaults to the JSON path with a .csv suffix.",
    )
    parser.add_argument("runs", nargs="+", help="Paths to trainer_state.json files or run dirs")
    args = parser.parse_args(argv)

    report = build_sft_comparison([Path(item) for item in args.runs])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    write_sft_comparison_csv(csv_path, report["runs"])
    print(f"comparison={output_path}")
    print(f"runs={len(report['runs'])} best_final_eval_run={report['best_final_eval_run']}")
    return 0


def build_sft_comparison(paths: list[Path]) -> dict[str, Any]:
    summaries = [summarize_sft_run(load_trainer_state(path)) for path in paths]
    best = min(
        (item for item in summaries if item["final_eval_loss"] is not None),
        key=lambda item: item["final_eval_loss"],
        default=None,
    )
    return {
        "runs": summaries,
        "best_final_eval_run": best["run_id"] if best else None,
        "notes": [
            "Compare LoRA and full SFT with the same model, dataset, split seed, sequence length, "
            "batching policy, optimizer steps, and evaluation cadence whenever possible.",
            "Use final_eval_delta below zero as the first sanity check for validation improvement.",
        ],
    }


def load_trainer_state(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "trainer_state.json"
    if not path.exists():
        raise FileNotFoundError(f"trainer state does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"trainer state must be a JSON object: {path}")
    return data


def summarize_sft_run(state: dict[str, Any]) -> dict[str, Any]:
    eval_history = state.get("eval_history") or []
    train_history = state.get("train_history") or []
    first_eval = _metric(eval_history[0], "eval_loss") if eval_history else None
    final_eval = _metric(eval_history[-1], "eval_loss") if eval_history else None
    best_eval = min(
        (_metric(item, "eval_loss") for item in eval_history if _metric(item, "eval_loss") is not None),
        default=None,
    )
    final_train = _metric(train_history[-1], "train_loss") if train_history else None
    return {
        "run_id": state.get("run_id"),
        "tuning_mode": state.get("tuning_mode", "lora"),
        "gradient_sync": state.get("gradient_sync"),
        "checkpoint_policy": state.get("checkpoint_policy", "final"),
        "world_size": state.get("world_size"),
        "model_name": state.get("model_name"),
        "dataset_name": state.get("dataset_name"),
        "train_samples": state.get("train_samples"),
        "eval_samples": state.get("eval_samples"),
        "max_seq_length": state.get("max_seq_length"),
        "steps": state.get("steps"),
        "epochs": state.get("epochs"),
        "gradient_accumulation_steps": state.get("gradient_accumulation_steps"),
        "logging_steps": state.get("logging_steps"),
        "train_eval_samples": state.get("train_eval_samples"),
        "learning_rate": state.get("learning_rate"),
        "warmup_ratio": state.get("warmup_ratio"),
        "warmup_steps": state.get("warmup_steps"),
        "lr_scheduler": state.get("lr_scheduler"),
        "planned_steps": state.get("planned_steps"),
        "early_stopped": state.get("early_stopped"),
        "best_eval_step": state.get("best_eval_step"),
        "best_checkpoint_path": state.get("best_checkpoint_path"),
        "trainable_parameters": (state.get("lora") or {}).get("trainable_parameters"),
        "initial_eval_loss": first_eval,
        "final_eval_loss": final_eval,
        "best_eval_loss": best_eval,
        "final_eval_delta": _delta(final_eval, first_eval),
        "final_train_loss": final_train,
        "duration_seconds": state.get("duration_seconds"),
    }


def write_sft_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric(item: Any, key: str) -> float | None:
    if not isinstance(item, dict) or item.get(key) is None:
        return None
    return float(item[key])


def _delta(final: float | None, initial: float | None) -> float | None:
    if final is None or initial is None:
        return None
    return final - initial


if __name__ == "__main__":
    raise SystemExit(main())
