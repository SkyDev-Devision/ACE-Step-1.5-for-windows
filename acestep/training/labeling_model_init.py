"""On-demand DiT initialization helpers for training auto-label flows."""

from __future__ import annotations

import os
from typing import Any

from acestep.api.server_utils import env_bool
from acestep.gpu_config import VRAM_AUTO_OFFLOAD_THRESHOLD_GB, get_gpu_config
from acestep.model_downloader import get_project_root


def ensure_training_labeling_model_ready(dit_handler: Any) -> tuple[str, bool]:
    """Initialize the DiT model once on demand for training auto-label entry points."""

    if dit_handler is None:
        return "❌ Model handler is unavailable.", False

    if getattr(dit_handler, "model", None) is not None:
        return "✅ Model already initialized.", True

    last_init_params = getattr(dit_handler, "last_init_params", None)
    if isinstance(last_init_params, dict) and last_init_params:
        return dit_handler.initialize_service(**last_init_params)

    gpu_config = get_gpu_config()
    auto_offload = (
        gpu_config.gpu_memory_gb > 0
        and gpu_config.gpu_memory_gb < VRAM_AUTO_OFFLOAD_THRESHOLD_GB
    )
    offload_to_cpu_env = os.getenv("ACESTEP_OFFLOAD_TO_CPU")
    offload_to_cpu = (
        env_bool("ACESTEP_OFFLOAD_TO_CPU", False)
        if offload_to_cpu_env is not None
        else auto_offload
    )

    project_root = str(get_project_root())
    config_path = (os.getenv("ACESTEP_CONFIG_PATH") or "acestep-v15-turbo").strip() or "acestep-v15-turbo"
    device = (os.getenv("ACESTEP_DEVICE") or "auto").strip() or "auto"
    use_flash_attention = env_bool("ACESTEP_USE_FLASH_ATTENTION", True)
    offload_dit_to_cpu = env_bool("ACESTEP_OFFLOAD_DIT_TO_CPU", False)
    compile_model = env_bool("ACESTEP_COMPILE_MODEL", False)

    return dit_handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=compile_model,
        offload_to_cpu=offload_to_cpu,
        offload_dit_to_cpu=offload_dit_to_cpu,
    )
