from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from all_in_post_training.pipeline.config import (
    PipelineConfigError,
    load_pipeline_config,
    parse_pipeline_config,
)
from all_in_post_training.pipeline.runner import PipelineRunner


class PipelineConfigTest(unittest.TestCase):
    def test_reference_pipeline_is_valid(self) -> None:
        config = load_pipeline_config("examples/post_training_pipeline.json")
        self.assertEqual(config.name, "qwen3.5-2b-sft-rl-opd-reference")
        self.assertEqual(config.model.base_model, "Qwen/Qwen3.5-2B-Base")
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

    def test_rejects_unknown_dependency(self) -> None:
        with self.assertRaises(PipelineConfigError):
            parse_pipeline_config(
                {
                    "name": "bad",
                    "version": "0",
                    "output_dir": "runs",
                    "model": {"name": "m", "base_model": "base"},
                    "datasets": [
                        {"id": "sft", "path": "data.jsonl", "role": "sft", "format": "jsonl"}
                    ],
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
                    "model": {"name": "m", "base_model": "base"},
                    "datasets": [
                        {"id": "rl", "path": "data.jsonl", "role": "rl", "format": "jsonl"}
                    ],
                    "stages": [
                        {
                            "id": "train_rl",
                            "type": "domain_rl",
                            "inputs": {"datasets": ["rl"]},
                        }
                    ],
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


if __name__ == "__main__":
    unittest.main()
