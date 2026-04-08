"""Unit tests for trainer-side auxiliary module movement helpers."""

import unittest

import torch
import torch.nn as nn

from acestep.training.trainer import _move_auxiliary_trainable_module


class _MetaModule(nn.Module):
    """Minimal module carrying a meta parameter."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2, device="meta"))


class TestMoveAuxiliaryTrainableModule(unittest.TestCase):
    """Test cases for external LyCORIS module movement."""

    def test_moves_materialized_module_to_requested_dtype(self):
        """Materialized auxiliary modules should be moved in-place."""
        module = nn.Linear(2, 2)

        _move_auxiliary_trainable_module(
            module,
            device=torch.device("cpu"),
            dtype=torch.float64,
            label="lycoris_net",
        )

        self.assertEqual(module.weight.dtype, torch.float64)

    def test_raises_for_meta_parameters_before_move(self):
        """Meta parameters should fail early with a targeted error."""
        module = _MetaModule()

        with self.assertRaisesRegex(RuntimeError, "contains meta parameters"):
            _move_auxiliary_trainable_module(
                module,
                device=torch.device("cpu"),
                dtype=torch.float32,
                label="lycoris_net",
            )


if __name__ == "__main__":
    unittest.main()
