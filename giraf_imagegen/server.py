"""
FastAPI server for local GPU image generation.

Start with:
    imagegen-serve --port 8300

Or:
    uvicorn giraf_imagegen.server:app
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from giraf_imagegen.config import get_config
from giraf_imagegen.pipeline import ImagePipeline

log = logging.getLogger(__name__)

_pipeline: ImagePipeline | None = None
_generate_lock: asyncio.Lock | None = None
_config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _generate_lock, _config

    _config = get_config()
    log.info(
        "Starting giraf-imagegen (checkpoint=%s, steps=%d) …",
        _config["checkpoint"],
        _config["default_steps"],
    )
    _pipeline = ImagePipeline(
        checkpoint=_config["checkpoint"],
        dtype=_config["dtype"],
        gpu_mem_fraction=_config["gpu_mem_fraction"],
    )
    _generate_lock = asyncio.Lock()
    log.info("giraf-imagegen ready.")
    yield
    _pipeline = None


app = FastAPI(title="GIRAF ImageGen", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=30.0)
    seed: int | None = None
    format: str = Field(default="png", pattern="^(png|webp|jpeg)$")


@app.post("/v1/image/generate")
async def generate(req: GenerateRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    width = min(req.width, _config["max_width"])
    height = min(req.height, _config["max_height"])
    steps = req.steps or _config["default_steps"]

    async with _generate_lock:
        try:
            image_bytes = await asyncio.to_thread(
                _pipeline.generate_bytes,
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                output_format=req.format,
            )
        except Exception as e:
            log.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    media_type = {
        "png": "image/png",
        "webp": "image/webp",
        "jpeg": "image/jpeg",
    }[req.format]

    return Response(content=image_bytes, media_type=media_type)


@app.get("/v1/models")
async def models():
    checkpoint = _config.get("checkpoint", "unknown")
    device = _pipeline.device_name if _pipeline else "not loaded"
    return {
        "models": [
            {
                "id": checkpoint,
                "device": device,
            }
        ]
    }


@app.get("/health")
async def health():
    if _pipeline is None:
        return Response(
            content='{"status":"not_ready"}',
            media_type="application/json",
            status_code=503,
        )
    return {
        "status": "ok",
        "model": _config.get("checkpoint"),
        "device": _pipeline.device_name,
    }


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="GIRAF ImageGen server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8300, help="Bind port (default: 8300)")
    args = parser.parse_args()

    uvicorn.run(
        "giraf_imagegen.server:app",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
