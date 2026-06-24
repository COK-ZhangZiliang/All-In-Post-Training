from __future__ import annotations

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
    format_prompt,
    load_instruction_dataset_file,
    load_instruction_rows,
    normalize_instruction_row,
    render_real_sft_curve_svg,
    truncate_for_supervised_response,
)
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
