from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from all_in_post_training.pipeline.audit import build_readiness_report
from all_in_post_training.pipeline.backends import (
    ManifestBackend,
    TorchSmokeBackend,
    TrlSftDryRunBackend,
    TrlSftExecuteBackend,
    create_backend,
)
from all_in_post_training.pipeline.dependencies import (
    MissingOptionalDependencyError,
    optional_dependency_status,
    require_optional_dependency,
)
from all_in_post_training.pipeline.lineage import (
    DataInspectionError,
    build_data_lineage_report,
    inspect_pipeline_data,
)
from all_in_post_training.pipeline.preflight import (
    TRAINING_EXTRA_PACKAGES,
    build_training_preflight_report,
    write_training_preflight_report,
)
from all_in_post_training.pipeline.config import (
    PipelineConfigError,
    load_pipeline_config,
    parse_pipeline_config,
)
from all_in_post_training.pipeline.distributed_sft import (
    _encode_example,
    render_loss_curve_svg,
    write_loss_artifacts,
)
from all_in_post_training.pipeline.real_sft import (
    build_deepspeed_zero3_config,
    build_sft_examples,
    collate_sft_examples,
    compute_planned_optimizer_steps,
    compute_scheduled_learning_rate,
    format_prompt,
    load_instruction_dataset_file,
    load_instruction_rows,
    normalize_instruction_row,
    render_real_sft_curve_svg,
    rolling_average_points,
    should_enable_gradient_checkpointing,
    truncate_for_supervised_response,
)
from all_in_post_training.pipeline.sft_compare import build_sft_comparison, write_sft_comparison_csv
from all_in_post_training.pipeline.runner import PipelineRunner


class PipelineConfigTest(unittest.TestCase):
    def test_reference_pipeline_is_valid(self) -> None:
        config = load_pipeline_config("examples/post_training_pipeline.json")
        self.assertEqual(config.name, "qwen3.5-2b-sft-rl-opd-reference")
        self.assertEqual(config.model.base_model, "Qwen/Qwen3.5-2B-Base")
        self.assertEqual(config.model.max_sequence_length, 8192)
        self.assertTrue(config.model.review_checklist)
        self.assertEqual(len(config.stages), 10)
        self.assertEqual(
            [stage.id for stage in config.stages],
            [
                "ingest_data",
                "mix_sft_data",
                "train_sft",
                "train_math_rl",
                "train_code_rl",
                "train_tool_agent_rl",
                "train_safety_instruction_rl",
                "opd_fuse_specialists",
                "evaluate_policy",
                "package_release",
            ],
        )
        self.assertTrue(all(dataset.required_columns for dataset in config.datasets))
        self.assertTrue(all(dataset.license_status for dataset in config.datasets))

    def test_rejects_unknown_dependency(self) -> None:
        with self.assertRaises(PipelineConfigError):
            parse_pipeline_config(
                {
                    "name": "bad",
                    "version": "0",
                    "output_dir": "runs",
                    "model": valid_model(),
                    "datasets": [valid_dataset("sft")],
                    "stages": [
                        {"id": "train", "type": "sft", "depends_on": ["missing"]}
                    ],
                }
            )

    def test_rejects_domain_rl_without_domain(self) -> None:
        with self.assertRaises(PipelineConfigError):
            parse_pipeline_config(
                {
                    "name": "bad-domain",
                    "version": "0",
                    "output_dir": "runs",
                    "model": valid_model(),
                    "datasets": [valid_dataset("rl", role="rl")],
                    "stages": [
                        {
                            "id": "train_rl",
                            "type": "domain_rl",
                            "inputs": {"datasets": ["rl"]},
                        }
                    ],
                }
            )

    def test_rejects_dataset_without_license_metadata(self) -> None:
        dataset = valid_dataset("sft")
        dataset.pop("license_status")
        with self.assertRaises(PipelineConfigError):
            parse_pipeline_config(
                {
                    "name": "bad-license",
                    "version": "0",
                    "output_dir": "runs",
                    "model": valid_model(),
                    "datasets": [dataset],
                    "stages": [{"id": "train", "type": "sft", "inputs": {"datasets": ["sft"]}}],
                }
            )

    def test_runner_materializes_artifacts(self) -> None:
        config = load_pipeline_config("examples/post_training_pipeline.json")
        with tempfile.TemporaryDirectory() as directory:
            patched = parse_pipeline_config(
                {
                    "name": config.name,
                    "version": config.version,
                    "output_dir": directory,
                    "model": config.model.__dict__,
                    "datasets": [dataset.__dict__ for dataset in config.datasets],
                    "stages": [
                        {
                            "id": stage.id,
                            "type": stage.type,
                            "depends_on": list(stage.depends_on),
                            "enabled": stage.enabled,
                            "params": stage.params,
                            "inputs": stage.inputs,
                            "outputs": stage.outputs,
                        }
                        for stage in config.stages
                    ],
                    "metadata": config.metadata,
                }
            )
            result = PipelineRunner().run(patched, run_id="unit-test")
            self.assertEqual(len(result.stages), 10)
            self.assertTrue((Path(directory) / "unit-test" / "run_manifest.json").exists())
            self.assertTrue(result.artifacts)

    def test_data_inspection_uses_fixtures_and_writes_report(self) -> None:
        config = load_pipeline_config("examples/post_training_pipeline.json")
        fixture_root = Path("tests/fixtures/lineage")
        with tempfile.TemporaryDirectory() as directory:
            result = inspect_pipeline_data(
                config,
                run_id="lineage-test",
                fixture_root=fixture_root,
                output_root=directory,
            )
            self.assertTrue(result.report_path.exists())
            self.assertEqual(result.report["summary"]["datasets"], len(config.datasets))
            self.assertEqual(result.report["summary"]["inspected"], len(config.datasets))
            self.assertEqual(result.report["summary"]["errors"], 0)
            self.assertEqual(result.report["summary"]["rejected_records"], 0)
            fingerprints = {
                dataset["id"]: dataset["inspection"]["fingerprint"]
                for dataset in result.report["datasets"]
            }
            self.assertTrue(all(fingerprints.values()))
            code_entry = next(
                dataset for dataset in result.report["datasets"] if dataset["id"] == "code_rl_tasks"
            )
            self.assertEqual(code_entry["inspection"]["source_kind"], "manifest")
            self.assertEqual(code_entry["inspection"]["record_count"], 2)
            self.assertEqual(code_entry["inspection"]["quality"]["status"], "ok")
            self.assertTrue(code_entry["inspection"]["manifest"]["path"].endswith(".manifest.json"))
            second = build_data_lineage_report(config, fixture_root=fixture_root, strict=True)
            self.assertEqual(
                fingerprints["math_rl_prompts"],
                next(
                    dataset["inspection"]["fingerprint"]
                    for dataset in second["datasets"]
                    if dataset["id"] == "math_rl_prompts"
                ),
            )

    def test_data_inspection_fails_on_missing_required_columns(self) -> None:
        fixture_root = Path("tests/fixtures/lineage")
        with tempfile.TemporaryDirectory() as directory:
            bad_root = Path(directory) / "fixtures"
            bad_root.mkdir()
            for fixture in fixture_root.glob("*.jsonl"):
                target = bad_root / fixture.name
                target.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
            bad_root.joinpath("math_rl_prompts.jsonl").write_text(
                '{"prompt":"Compute 1 + 1."}\n',
                encoding="utf-8",
            )
            config = load_pipeline_config("examples/post_training_pipeline.json")
            with self.assertRaises(DataInspectionError):
                inspect_pipeline_data(
                    config,
                    run_id="bad-lineage",
                    fixture_root=bad_root,
                    output_root=directory,
                )

    def test_data_inspection_flags_duplicate_and_missing_quality_fields(self) -> None:
        fixture_root = Path("tests/fixtures/lineage")
        with tempfile.TemporaryDirectory() as directory:
            bad_root = Path(directory) / "fixtures"
            bad_root.mkdir()
            for fixture in fixture_root.glob("*.jsonl"):
                target = bad_root / fixture.name
                target.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
            bad_root.joinpath("code_rl_tasks.jsonl").write_text(
                "\n".join(
                    [
                        '{"prompt":"Write add.","tests":["assert add(1, 1) == 2"]}',
                        '{"prompt":"Write add.","tests":[]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = load_pipeline_config("examples/post_training_pipeline.json")
            report = build_data_lineage_report(config, fixture_root=bad_root, strict=True)
            code_entry = next(
                dataset for dataset in report["datasets"] if dataset["id"] == "code_rl_tasks"
            )
            self.assertEqual(code_entry["inspection"]["status"], "error")
            self.assertEqual(code_entry["inspection"]["quality"]["rejected_count"], 1)
            self.assertGreater(report["summary"]["rejected_records"], 0)
            self.assertTrue(
                any("duplicate prompt" in error for error in code_entry["inspection"]["errors"])
            )
            self.assertTrue(
                any(
                    "required column tests is empty" in error
                    for error in code_entry["inspection"]["errors"]
                )
            )

    def test_readiness_audit_reports_training_blockers(self) -> None:
        config = load_pipeline_config("examples/post_training_pipeline.json")
        report = build_readiness_report(config)
        self.assertEqual(report["status"], "blocked")
        self.assertGreater(report["summary"]["blockers"], 0)
        self.assertTrue(
            any("model revision_status must be pinned" in blocker for blocker in report["blockers"])
        )
        self.assertTrue(
            any(
                "model license_status must be approved" in blocker
                for blocker in report["blockers"]
            )
        )
        self.assertTrue(
            any(
                dataset["id"] == "sft_general_chat" and dataset["status"] == "ready"
                for dataset in report["datasets"]
            )
        )
        self.assertTrue(
            any(
                dataset["id"] == "sft_instruction_scale" and dataset["status"] == "blocked"
                for dataset in report["datasets"]
            )
        )

    def test_backend_factory_selects_manifest(self) -> None:
        self.assertIsInstance(create_backend("manifest"), ManifestBackend)
        self.assertIsInstance(create_backend("trl-sft-dry-run"), TrlSftDryRunBackend)
        training_extras_available = all(
            optional_dependency_status(package)["available"]
            for package in TRAINING_EXTRA_PACKAGES
        )
        if training_extras_available:
            self.assertIsInstance(create_backend("trl-sft-execute"), TrlSftExecuteBackend)
        else:
            with self.assertRaises(MissingOptionalDependencyError):
                create_backend("trl-sft-execute")
        with self.assertRaises(ValueError):
            create_backend("missing")

    def test_optional_dependency_error_names_missing_package(self) -> None:
        with self.assertRaises(MissingOptionalDependencyError) as context:
            require_optional_dependency(
                "all_in_post_training_missing_dependency_for_test",
                "test dependency path",
            )
        self.assertIn("all_in_post_training_missing_dependency_for_test", str(context.exception))

    def test_training_preflight_reports_backend_modes(self) -> None:
        config = load_pipeline_config("examples/post_training_pipeline.json")
        report = build_training_preflight_report(config)
        self.assertEqual(report["pipeline"], config.name)
        self.assertIn("torch", report["packages"])
        self.assertIn("trl_sft_execute", report["modes"])
        self.assertFalse(report["modes"]["trl_sft_execute"]["ready"])
        self.assertTrue(
            any(
                "readiness audit" in blocker
                for blocker in report["modes"]["trl_sft_execute"]["blockers"]
            )
        )

    def test_training_preflight_writes_report(self) -> None:
        config = load_pipeline_config("examples/post_training_pipeline.json")
        with tempfile.TemporaryDirectory() as directory:
            result = write_training_preflight_report(
                config,
                run_id="preflight-test",
                output_root=directory,
                require_training_extras=True,
            )
            self.assertTrue(result.report_path.exists())
            self.assertEqual(result.report["requirements"]["require_training_extras"], True)
            self.assertIn("training_extras", result.report["missing"])

    def test_distributed_sft_fixture_encoding_is_fixed_length(self) -> None:
        tokens = _encode_example("Hello", "World", sequence_length=16)
        self.assertEqual(len(tokens), 16)
        self.assertTrue(all(0 <= token <= 257 for token in tokens))

    def test_distributed_sft_writes_loss_artifacts(self) -> None:
        losses = [
            {"step": 1, "epoch": 1, "loss": 2.0, "local_loss": 2.1, "grad_norm": 0.8},
            {"step": 2, "epoch": 1, "loss": 1.5, "local_loss": 1.6, "grad_norm": 0.7},
        ]
        svg = render_loss_curve_svg(losses, title="unit loss")
        self.assertIn("<svg", svg)
        self.assertIn("unit loss", svg)
        self.assertIn("final_loss=1.500000", svg)
        with tempfile.TemporaryDirectory() as directory:
            write_loss_artifacts(Path(directory), losses, "unit-loss")
            self.assertTrue(Path(directory, "loss_curve.svg").exists())
            self.assertEqual(
                Path(directory, "loss_history.csv").read_text(encoding="utf-8").splitlines()[0],
                "step,epoch,loss,local_loss,grad_norm",
            )

    def test_real_sft_formats_rows_and_metric_curve(self) -> None:
        row = normalize_instruction_row(
            {
                "instruction": "Summarize the text.",
                "context": "A short article.",
                "response": "A brief summary.",
                "category": "summarization",
            }
        )
        self.assertIn("### Instruction:", format_prompt(row))
        self.assertIn("### Input:", format_prompt(row))
        svg = render_real_sft_curve_svg(
            train_history=[
                {"step": 1, "train_loss": 3.0},
                {"step": 2, "train_loss": 2.0},
            ],
            eval_history=[
                {"step": 0, "eval_loss": 3.4},
                {"step": 2, "eval_loss": 2.1},
            ],
            run_id="unit-real-sft",
        )
        self.assertIn("<svg", svg)
        self.assertIn("unit-real-sft", svg)
        self.assertIn("final_eval_loss=2.100000", svg)
        self.assertIn("train_ma8", svg)

    def test_real_sft_normalizes_chat_messages(self) -> None:
        row = normalize_instruction_row(
            {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Explain merge sort."},
                    {"role": "assistant", "content": "Merge sort divides and merges."},
                ],
                "source": "modelscope-chat",
            }
        )
        self.assertEqual(row["instruction"], "Explain merge sort.")
        self.assertEqual(row["response"], "Merge sort divides and merges.")
        self.assertEqual(row["category"], "modelscope-chat")

    def test_real_sft_loads_local_jsonl_instruction_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpaca.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"instruction":"First task","input":"","output":"First answer"}',
                        '{"instruction":"Second task","input":"","output":"Second answer"}',
                        '{"instruction":"Third task","input":"","output":"Third answer"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_instruction_rows(
                load_instruction_dataset_file(path),
                train_samples=2,
                eval_samples=1,
                seed=20260624,
            )
        self.assertEqual(len(rows["train"]), 2)
        self.assertEqual(len(rows["eval"]), 1)
        self.assertTrue(all(row["instruction"] for row in rows["train"] + rows["eval"]))

    def test_real_sft_truncation_keeps_response_labels(self) -> None:
        prompt_ids = list(range(100))
        response_ids = [200, 201, 202, 203]
        prompt, response = truncate_for_supervised_response(
            prompt_ids,
            response_ids,
            max_seq_length=8,
        )
        self.assertEqual(len(prompt) + len(response), 8)
        self.assertEqual(response, response_ids)

    def test_real_sft_collator_uses_dynamic_padding(self) -> None:
        examples = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "labels": [-100, 5]},
        ]
        batch = collate_sft_examples(examples, pad_token_id=0)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        fixed = collate_sft_examples(examples, pad_token_id=0, pad_to=8)
        self.assertEqual(tuple(fixed["input_ids"].shape), (2, 8))

    def test_real_sft_examples_keep_variable_lengths_before_collation(self) -> None:
        class TinyTokenizer:
            eos_token = "<eos>"
            pad_token_id = 0

            def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
                del add_special_tokens
                return {"input_ids": list(range(1, len(text.split()) + 1))}

        rows = [
            {"instruction": "short", "context": "", "response": "brief answer"},
            {"instruction": "a much longer prompt", "context": "", "response": "brief answer"},
        ]
        examples = build_sft_examples(TinyTokenizer(), rows, max_seq_length=32)
        self.assertNotEqual(len(examples[0]["input_ids"]), len(examples[1]["input_ids"]))

    def test_real_sft_rolling_average_points(self) -> None:
        self.assertEqual(
            rolling_average_points([(1, 2.0), (2, 4.0), (3, 6.0)], window=2),
            [(1, 2.0), (2, 3.0), (3, 5.0)],
        )

    def test_real_sft_zero3_config_uses_stage_three_offload(self) -> None:
        config = build_deepspeed_zero3_config(batch_size=1, world_size=2, learning_rate=5e-5)
        self.assertEqual(config["train_batch_size"], 2)
        self.assertEqual(config["zero_optimization"]["stage"], 3)
        self.assertEqual(config["zero_optimization"]["offload_optimizer"]["device"], "cpu")
        self.assertEqual(config["optimizer"]["params"]["lr"], 5e-5)

    def test_real_sft_disables_gradient_checkpointing_for_zero3(self) -> None:
        self.assertFalse(should_enable_gradient_checkpointing("deepspeed-zero3"))
        self.assertTrue(should_enable_gradient_checkpointing("cpu-allreduce"))

    def test_real_sft_cosine_schedule_with_warmup(self) -> None:
        planned = compute_planned_optimizer_steps(steps_per_epoch=50, epochs=3, max_steps=None)
        self.assertEqual(planned, 150)
        warmup_steps = int(planned * 0.1)
        self.assertAlmostEqual(
            compute_scheduled_learning_rate(
                step=1,
                total_steps=planned,
                base_learning_rate=1e-5,
                warmup_steps=warmup_steps,
                schedule="cosine",
            ),
            1e-5 / warmup_steps,
        )
        self.assertAlmostEqual(
            compute_scheduled_learning_rate(
                step=warmup_steps,
                total_steps=planned,
                base_learning_rate=1e-5,
                warmup_steps=warmup_steps,
                schedule="cosine",
            ),
            1e-5,
        )
        self.assertAlmostEqual(
            compute_scheduled_learning_rate(
                step=planned,
                total_steps=planned,
                base_learning_rate=1e-5,
                warmup_steps=warmup_steps,
                schedule="cosine",
            ),
            0.0,
        )

    def test_sft_comparison_summarizes_lora_and_full_runs(self) -> None:
        lora = {
            "run_id": "lora-run",
            "tuning_mode": "lora",
            "gradient_sync": "deepspeed-zero3",
            "checkpoint_policy": "none",
            "world_size": 2,
            "model_name": "Qwen/Qwen3.5-2B-Base",
            "dataset_name": "swift/Qwen3-SFT-Mixin",
            "train_samples": 100,
            "eval_samples": 20,
            "max_seq_length": 2048,
            "steps": 10,
            "epochs": 1,
            "learning_rate": 5e-5,
            "lora": {"trainable_parameters": 1000},
            "train_history": [{"step": 10, "train_loss": 1.8}],
            "eval_history": [
                {"step": 0, "eval_loss": 2.4},
                {"step": 10, "eval_loss": 2.0},
            ],
            "duration_seconds": 12.0,
        }
        full = {
            **lora,
            "run_id": "full-run",
            "tuning_mode": "full",
            "lora": {"trainable_parameters": 2_000_000_000},
            "eval_history": [
                {"step": 0, "eval_loss": 2.4},
                {"step": 10, "eval_loss": 1.9},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            lora_path = Path(directory) / "lora" / "trainer_state.json"
            full_path = Path(directory) / "full" / "trainer_state.json"
            lora_path.parent.mkdir()
            full_path.parent.mkdir()
            lora_path.write_text(json.dumps(lora), encoding="utf-8")
            full_path.write_text(json.dumps(full), encoding="utf-8")
            comparison = build_sft_comparison([lora_path, full_path.parent])
            self.assertEqual(comparison["best_final_eval_run"], "full-run")
            self.assertEqual(comparison["runs"][0]["checkpoint_policy"], "none")
            self.assertEqual(comparison["runs"][0]["final_eval_delta"], -0.3999999999999999)
            csv_path = Path(directory) / "comparison.csv"
            write_sft_comparison_csv(csv_path, comparison["runs"])
            self.assertIn("tuning_mode", csv_path.read_text(encoding="utf-8").splitlines()[0])

    def test_torch_smoke_backend_materializes_when_torch_is_available(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        config = load_pipeline_config("examples/post_training_pipeline.json")
        with tempfile.TemporaryDirectory() as directory:
            patched = parse_pipeline_config(
                {
                    "name": config.name,
                    "version": config.version,
                    "output_dir": directory,
                    "model": config.model.__dict__,
                    "datasets": [dataset.__dict__ for dataset in config.datasets],
                    "stages": [
                        {
                            "id": "train_sft",
                            "type": "sft",
                            "params": {"backend": "torch-smoke"},
                            "inputs": {"datasets": ["sft_general_chat"]},
                        }
                    ],
                    "metadata": config.metadata,
                }
            )
            result = PipelineRunner(backend=TorchSmokeBackend()).run(patched, run_id="torch-smoke")
            self.assertEqual(len(result.artifacts), 1)
            artifact_path = Path(result.artifacts[0].path)
            self.assertTrue(artifact_path.exists())
            self.assertIn('"backend": "torch_smoke"', artifact_path.read_text(encoding="utf-8"))

    def test_trl_sft_dry_run_backend_materializes_sft_checkpoint(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        config = load_pipeline_config("examples/post_training_pipeline.json")
        with tempfile.TemporaryDirectory() as directory:
            patched = parse_pipeline_config(
                {
                    "name": config.name,
                    "version": config.version,
                    "output_dir": directory,
                    "model": config.model.__dict__,
                    "datasets": [dataset.__dict__ for dataset in config.datasets],
                    "stages": [
                        {
                            "id": "train_sft",
                            "type": "sft",
                            "params": {
                                "backend": "trl",
                                "dry_run_steps": 1,
                                "dry_run_seq_length": 8,
                            },
                            "inputs": {"datasets": ["sft_general_chat"]},
                            "outputs": {"checkpoint": "checkpoints/qwen3.5-2b-sft"},
                        }
                    ],
                    "metadata": config.metadata,
                }
            )
            result = PipelineRunner(backend=TrlSftDryRunBackend()).run(
                patched,
                run_id="trl-sft-dry-run",
            )
            self.assertEqual(len(result.artifacts), 1)
            artifact_path = Path(result.artifacts[0].path)
            checkpoint_dir = Path(directory) / "trl-sft-dry-run" / "checkpoints" / "train_sft"
            self.assertTrue(artifact_path.exists())
            self.assertTrue((checkpoint_dir / "trainer_state.json").exists())
            self.assertTrue((checkpoint_dir / "adapter_config.json").exists())
            artifact_text = artifact_path.read_text(encoding="utf-8")
            self.assertIn('"backend": "trl_sft_dry_run"', artifact_text)
            self.assertIn('"target": "trl.SFTTrainer"', artifact_text)


def valid_model() -> dict[str, object]:
    return {
        "name": "m",
        "base_model": "base",
        "source_url": "https://example.com/model",
        "revision": "main",
        "revision_status": "test",
        "tokenizer": "base",
        "tokenizer_revision": "main",
        "license": "test",
        "license_status": "needs_review",
        "precision": "bf16",
        "max_sequence_length": 128,
        "chat_template": "test_template",
        "review_checklist": ["pin revision", "confirm license"],
    }


def valid_dataset(dataset_id: str, role: str = "sft") -> dict[str, object]:
    return {
        "id": dataset_id,
        "path": f"{dataset_id}.jsonl",
        "role": role,
        "domain": "general",
        "task_role": "test",
        "format": "jsonl",
        "schema": "sft_chat",
        "required_columns": ["messages"],
        "split": "train",
        "split_policy": "test_holdout",
        "license": "test",
        "license_status": "needs_review",
        "contamination_status": "not_checked",
        "quality_filters": ["message_non_empty"],
    }


if __name__ == "__main__":
    unittest.main()
