"""Load and run a diffusers image generation pipeline."""

import io
import logging

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ImagePipeline:
    """Wraps a diffusers text-to-image pipeline with device auto-detection."""

    def __init__(
        self,
        checkpoint: str,
        dtype: str = "bfloat16",
        gpu_mem_fraction: float = 0.9,
    ) -> None:
        self.checkpoint = checkpoint
        self._torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)
        self._device = self._detect_device()

        log.info(
            "Loading pipeline %s on %s (dtype=%s) …",
            checkpoint,
            self._device,
            dtype,
        )

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            checkpoint,
            torch_dtype=self._torch_dtype,
        )

        if self._device.type == "cuda" and gpu_mem_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
        self._pipe.to(self._device)

        log.info("Pipeline ready on %s.", self._device)

    @staticmethod
    def _detect_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        log.warning("No GPU detected — falling back to CPU (will be very slow).")
        return torch.device("cpu")

    @property
    def device_name(self) -> str:
        if self._device.type == "cuda":
            return torch.cuda.get_device_name(0)
        return str(self._device)

    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        result = self._pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]

    def generate_bytes(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        output_format: str = "png",
    ) -> bytes:
        image = self.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        buf = io.BytesIO()
        image.save(buf, format=output_format.upper())
        return buf.getvalue()
