"""On-demand LLM initialization helpers for training auto-label flows."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

from loguru import logger

from acestep.gpu_config import (
    find_best_lm_model_on_disk,
    get_global_gpu_config,
    get_recommended_lm_model,
    resolve_lm_backend,
)
from acestep.model_downloader import get_checkpoints_dir


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment variable value with a fallback default."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _update_app_state(app_state: Any, status: str, ok: bool) -> None:
    """Mirror LLM readiness back onto the optional app state container."""

    if app_state is None:
        return
    app_state._llm_initialized = bool(ok)
    app_state._llm_init_error = None if ok else status
    if ok and hasattr(app_state, "_llm_lazy_load_disabled"):
        app_state._llm_lazy_load_disabled = False


def _select_lm_model_path(llm_handler: Any, gpu_config: Any) -> str | None:
    """Pick the LM model path to use for on-demand training initialization."""

    requested_model = (os.getenv("ACESTEP_LM_MODEL_PATH") or "").strip()
    if requested_model:
        return requested_model

    disk_models_getter = getattr(llm_handler, "get_available_5hz_lm_models", None)
    disk_models = list(disk_models_getter() or []) if callable(disk_models_getter) else []
    recommended_model = get_recommended_lm_model(gpu_config) or ""
    return find_best_lm_model_on_disk(recommended_model, disk_models)


def ensure_training_labeling_llm_ready(llm_handler: Any, app_state: Any = None) -> tuple[str, bool]:
    """Initialize the LLM once on demand for training auto-label entry points."""

    if llm_handler is None:
        return "❌ LLM handler is unavailable.", False

    if getattr(llm_handler, "llm_initialized", False):
        _update_app_state(app_state, "", True)
        return "✅ LLM already initialized.", True

    init_lock = getattr(app_state, "_llm_init_lock", None)
    lock_context = init_lock if init_lock is not None else nullcontext()
    with lock_context:
        if getattr(llm_handler, "llm_initialized", False):
            _update_app_state(app_state, "", True)
            return "✅ LLM already initialized.", True

        last_init_params = getattr(llm_handler, "last_init_params", None)
        if isinstance(last_init_params, dict) and last_init_params:
            logger.info("Training auto-label: reinitializing LLM with previous parameters")
            status, ok = llm_handler.initialize(**last_init_params)
            _update_app_state(app_state, status, ok)
            return status, bool(ok)

        gpu_config = get_global_gpu_config()
        lm_model_path = _select_lm_model_path(llm_handler, gpu_config)
        if not lm_model_path:
            status = "❌ No 5Hz LM model found under checkpoints. Please install an LM model first."
            _update_app_state(app_state, status, False)
            return status, False

        device = (os.getenv("ACESTEP_LM_DEVICE") or os.getenv("ACESTEP_DEVICE") or "auto").strip() or "auto"
        if not getattr(gpu_config, "available_lm_models", None):
            device = "cpu"

        backend = resolve_lm_backend(os.getenv("ACESTEP_LM_BACKEND"), gpu_config)
        offload_to_cpu = _env_bool(
            "ACESTEP_LM_OFFLOAD_TO_CPU",
            bool(getattr(gpu_config, "offload_to_cpu_default", False)),
        )
        checkpoint_dir = str(get_checkpoints_dir())

        logger.info(
            "Training auto-label: initializing LLM on demand "
            f"(model={lm_model_path}, backend={backend}, device={device})"
        )
        status, ok = llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_model_path,
            backend=backend,
            device=device,
            offload_to_cpu=offload_to_cpu,
            dtype=None,
        )
        _update_app_state(app_state, status, ok)
        return status, bool(ok)
