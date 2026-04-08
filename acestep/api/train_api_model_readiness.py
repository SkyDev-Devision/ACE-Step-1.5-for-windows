"""Shared model-readiness helpers for training API routes."""

from __future__ import annotations

import asyncio
from fastapi import FastAPI

from acestep.api.http.model_init_service import initialize_models_for_request
from acestep.api.model_download import ensure_model_downloaded
from acestep.api.server_utils import env_bool, get_model_name
from acestep.model_downloader import get_project_root


async def ensure_primary_training_model_ready(app: FastAPI) -> None:
    """Initialize the primary DiT model on demand for training routes."""

    handler = getattr(app.state, "handler", None)
    if handler is None:
        raise RuntimeError("Model handler is unavailable")

    if getattr(handler, "model", None) is not None:
        return

    async with app.state._init_lock:
        if getattr(handler, "model", None) is not None:
            return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            app.state.executor,
            lambda: initialize_models_for_request(
                app_state=app.state,
                model_name=get_model_name(getattr(app.state, "_config_path", "")),
                slot=1,
                init_llm=False,
                requested_lm_model_path=None,
                get_project_root=lambda: str(get_project_root()),
                get_model_name=get_model_name,
                ensure_model_downloaded=ensure_model_downloaded,
                env_bool=env_bool,
            ),
        )
