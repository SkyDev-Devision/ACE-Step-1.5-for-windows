"""Regression tests for training V2 official model alias resolution."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path


class TestTrainingVariantAliases(unittest.TestCase):
    """Training V2 should resolve both legacy and XL aliases to official XL dirs."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = Path(self._tmpdir.name)
        xl_sft_dir = self.checkpoint_dir / "acestep-v15-xl-sft"
        xl_sft_dir.mkdir(parents=True)
        (xl_sft_dir / "config.json").write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_config_builder_maps_legacy_sft_alias_to_xl_sft(self) -> None:
        from acestep.training_v2.cli.config_builder import _resolve_model_config_path

        resolved = _resolve_model_config_path(self.checkpoint_dir, "sft")

        self.assertEqual(
            self.checkpoint_dir / "acestep-v15-xl-sft" / "config.json",
            resolved,
        )

    def test_config_builder_accepts_explicit_xl_sft_variant(self) -> None:
        from acestep.training_v2.cli.config_builder import _resolve_model_config_path

        resolved = _resolve_model_config_path(self.checkpoint_dir, "xl_sft")

        self.assertEqual(
            self.checkpoint_dir / "acestep-v15-xl-sft" / "config.json",
            resolved,
        )

    def test_validation_accepts_explicit_xl_sft_variant(self) -> None:
        from acestep.training_v2.cli.validation import validate_paths

        args = argparse.Namespace(
            checkpoint_dir=str(self.checkpoint_dir),
            model_variant="xl_sft",
            dataset_dir=None,
            resume_from=None,
        )

        self.assertTrue(validate_paths(args))
        self.assertEqual(self.checkpoint_dir / "acestep-v15-xl-sft", args.model_dir)

    def test_model_loader_maps_legacy_sft_alias_to_xl_sft(self) -> None:
        from acestep.training_v2.model_loader import _resolve_model_dir

        resolved = _resolve_model_dir(self.checkpoint_dir, "sft")

        self.assertEqual(self.checkpoint_dir / "acestep-v15-xl-sft", resolved)


if __name__ == "__main__":
    unittest.main()
