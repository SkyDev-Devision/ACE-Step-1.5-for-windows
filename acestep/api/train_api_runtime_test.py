"""Unit tests for training API runtime component management."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import torch
import torch.nn as nn

from acestep.api.train_api_runtime import RuntimeComponentManager


class _TrackableModule(nn.Module):
    """Small module test double that records ``to()`` and ``eval()`` calls."""

    def __init__(self) -> None:
        """Initialize a minimal parameterized module for device inspection."""

        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.to_calls: list[tuple[tuple, dict]] = []
        self.eval_called = False

    def to(self, *args, **kwargs):
        """Record ``to()`` invocations and return ``self`` for call chaining."""

        self.to_calls.append((args, kwargs))
        return self

    def eval(self):
        """Record ``eval()`` usage during restore flows."""

        self.eval_called = True
        return self


class RuntimeComponentManagerTests(unittest.TestCase):
    """Behavior tests for training runtime offload and restore helpers."""

    def test_move_decoder_prefers_handler_recursive_transfer(self) -> None:
        """Training runtime should use handler recursive transfer when available."""

        decoder = _TrackableModule()
        recursive_move = MagicMock()
        handler = SimpleNamespace(
            model=SimpleNamespace(decoder=decoder),
            dtype=torch.bfloat16,
            _recursive_to_device=recursive_move,
        )
        manager = RuntimeComponentManager(
            handler=handler,
            llm=None,
            app_state=SimpleNamespace(),
        )

        manager.move_decoder_to("cuda:0")

        recursive_move.assert_called_once_with(decoder, "cuda:0", torch.bfloat16)
        self.assertEqual([], decoder.to_calls)

    def test_move_decoder_falls_back_to_direct_module_to(self) -> None:
        """Training runtime should keep direct ``module.to`` fallback semantics."""

        decoder = _TrackableModule()
        handler = SimpleNamespace(
            model=SimpleNamespace(decoder=decoder),
            dtype=torch.float16,
        )
        manager = RuntimeComponentManager(
            handler=handler,
            llm=None,
            app_state=SimpleNamespace(),
        )

        manager.move_decoder_to("cuda:0")

        self.assertEqual(
            [
                (("cuda:0",), {}),
                ((torch.float16,), {}),
            ],
            decoder.to_calls,
        )

    def test_restore_uses_same_recursive_transfer_preference(self) -> None:
        """Restore should use the same preferred recursive move path as offload."""

        decoder = _TrackableModule()
        vae = _TrackableModule()
        recursive_move = MagicMock()
        handler = SimpleNamespace(
            model=SimpleNamespace(
                decoder=decoder,
                encoder=None,
                tokenizer=None,
                detokenizer=None,
            ),
            vae=vae,
            text_encoder=None,
            _recursive_to_device=recursive_move,
        )
        manager = RuntimeComponentManager(
            handler=handler,
            llm=None,
            app_state=SimpleNamespace(),
        )
        manager._decoder_prev_device = "cuda:0"
        manager._decoder_prev_dtype = torch.bfloat16
        manager._vae_prev_device = "cpu"

        manager.restore()

        recursive_move.assert_has_calls(
            [
                call(decoder, "cuda:0", torch.bfloat16),
                call(vae, "cpu", None),
            ]
        )
        self.assertTrue(decoder.eval_called)


if __name__ == "__main__":
    unittest.main()
