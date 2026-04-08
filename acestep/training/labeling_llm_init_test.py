"""Tests for on-demand LLM initialization used by training auto-label flows."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from acestep.training.labeling_llm_init import ensure_training_labeling_llm_ready


class EnsureTrainingLabelingLlmReadyTests(unittest.TestCase):
    """Verify the helper keeps training auto-label compatible with lazy load."""

    @patch.dict("os.environ", {}, clear=False)
    def test_reuses_previous_init_params_when_available(self) -> None:
        handler = MagicMock()
        handler.llm_initialized = False
        handler.last_init_params = {
            "checkpoint_dir": "checkpoints",
            "lm_model_path": "acestep-5Hz-lm-0.6B",
            "backend": "pt",
            "device": "cpu",
            "offload_to_cpu": False,
            "dtype": None,
        }
        handler.initialize.return_value = ("✅ ready", True)

        status, ok = ensure_training_labeling_llm_ready(handler)

        self.assertTrue(ok)
        self.assertEqual("✅ ready", status)
        handler.initialize.assert_called_once_with(**handler.last_init_params)

    @patch.dict("os.environ", {}, clear=False)
    @patch("acestep.training.labeling_llm_init.resolve_lm_backend", return_value="pt")
    @patch("acestep.training.labeling_llm_init.get_checkpoints_dir", return_value=Path("/repo/checkpoints"))
    @patch("acestep.training.labeling_llm_init.find_best_lm_model_on_disk", return_value="acestep-5Hz-lm-0.6B-v4")
    @patch("acestep.training.labeling_llm_init.get_recommended_lm_model", return_value="acestep-5Hz-lm-0.6B")
    @patch(
        "acestep.training.labeling_llm_init.get_global_gpu_config",
        return_value=SimpleNamespace(available_lm_models=[], offload_to_cpu_default=True),
    )
    def test_selects_disk_model_and_cpu_fallback_for_first_init(
        self,
        _mock_gpu_config,
        _mock_recommended_model,
        _mock_find_best_model,
        _mock_checkpoints_dir,
        _mock_resolve_backend,
    ) -> None:
        handler = MagicMock()
        handler.llm_initialized = False
        handler.last_init_params = None
        handler.get_available_5hz_lm_models.return_value = ["acestep-5Hz-lm-0.6B-v4"]
        handler.initialize.return_value = ("✅ initialized", True)

        status, ok = ensure_training_labeling_llm_ready(handler)

        self.assertTrue(ok)
        self.assertEqual("✅ initialized", status)
        handler.initialize.assert_called_once_with(
            checkpoint_dir=str(Path("/repo/checkpoints")),
            lm_model_path="acestep-5Hz-lm-0.6B-v4",
            backend="pt",
            device="cpu",
            offload_to_cpu=True,
            dtype=None,
        )

    @patch.dict("os.environ", {}, clear=False)
    @patch(
        "acestep.training.labeling_llm_init.get_global_gpu_config",
        return_value=SimpleNamespace(available_lm_models=["acestep-5Hz-lm-0.6B"], offload_to_cpu_default=False),
    )
    def test_returns_clear_error_when_no_lm_model_is_available(self, _mock_gpu_config) -> None:
        handler = MagicMock()
        handler.llm_initialized = False
        handler.last_init_params = None
        handler.get_available_5hz_lm_models.return_value = []

        status, ok = ensure_training_labeling_llm_ready(handler)

        self.assertFalse(ok)
        self.assertIn("No 5Hz LM model found", status)
        handler.initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
