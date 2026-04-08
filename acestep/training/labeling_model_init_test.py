"""Tests for on-demand DiT initialization used by training auto-label flows."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from acestep.training.labeling_model_init import ensure_training_labeling_model_ready


class EnsureTrainingLabelingModelReadyTests(unittest.TestCase):
    """Verify the helper keeps training auto-label compatible with lazy model load."""

    def test_reuses_previous_init_params_when_available(self) -> None:
        handler = MagicMock()
        handler.model = None
        handler.last_init_params = {
            "project_root": "repo",
            "config_path": "acestep-v15-turbo",
            "device": "cpu",
            "use_flash_attention": False,
            "compile_model": False,
            "offload_to_cpu": False,
            "offload_dit_to_cpu": False,
            "quantization": None,
            "use_mlx_dit": True,
            "prefer_source": None,
        }
        handler.initialize_service.return_value = ("✅ ready", True)

        status, ok = ensure_training_labeling_model_ready(handler)

        self.assertTrue(ok)
        self.assertEqual("✅ ready", status)
        handler.initialize_service.assert_called_once_with(**handler.last_init_params)

    @patch.dict("os.environ", {}, clear=False)
    @patch("acestep.training.labeling_model_init.get_project_root", return_value="E:/repo")
    @patch(
        "acestep.training.labeling_model_init.get_gpu_config",
        return_value=MagicMock(gpu_memory_gb=12.0),
    )
    def test_uses_environment_defaults_for_first_init(
        self,
        _mock_gpu_config,
        _mock_project_root,
    ) -> None:
        handler = MagicMock()
        handler.model = None
        handler.last_init_params = None
        handler.initialize_service.return_value = ("✅ initialized", True)

        status, ok = ensure_training_labeling_model_ready(handler)

        self.assertTrue(ok)
        self.assertEqual("✅ initialized", status)
        handler.initialize_service.assert_called_once_with(
            project_root="E:/repo",
            config_path="acestep-v15-turbo",
            device="auto",
            use_flash_attention=True,
            compile_model=False,
            offload_to_cpu=True,
            offload_dit_to_cpu=False,
        )


if __name__ == "__main__":
    unittest.main()
