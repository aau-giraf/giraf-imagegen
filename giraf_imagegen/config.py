"""Configuration via environment variables."""

import os


def get_config() -> dict:
    return {
        "checkpoint": os.environ.get(
            "IMAGEGEN_CHECKPOINT", "black-forest-labs/FLUX.1-schnell"
        ),
        "gpu_mem_fraction": float(os.environ.get("IMAGEGEN_GPU_MEM", "0.9")),
        "default_steps": int(os.environ.get("IMAGEGEN_DEFAULT_STEPS", "4")),
        "default_width": int(os.environ.get("IMAGEGEN_DEFAULT_WIDTH", "512")),
        "default_height": int(os.environ.get("IMAGEGEN_DEFAULT_HEIGHT", "512")),
        "max_width": int(os.environ.get("IMAGEGEN_MAX_WIDTH", "1024")),
        "max_height": int(os.environ.get("IMAGEGEN_MAX_HEIGHT", "1024")),
        "dtype": os.environ.get("IMAGEGEN_DTYPE", "bfloat16"),
    }
