# giraf-imagegen

Local GPU image generation service for the GIRAF platform. Runs diffusers-based models (Flux, SDXL, etc.) behind a simple REST API.

## Quick start

```bash
# Install
uv sync --extra serve

# Run (downloads model on first start)
imagegen-serve --port 8300

# Or with environment overrides
IMAGEGEN_CHECKPOINT=stabilityai/stable-diffusion-xl-base-1.0 \
IMAGEGEN_DEFAULT_STEPS=20 \
imagegen-serve
```

## API

### `POST /v1/image/generate`

Generate an image from a text prompt.

```json
{
  "prompt": "A simple pictogram of a child eating lunch",
  "width": 512,
  "height": 512,
  "steps": 4,
  "guidance_scale": 0.0,
  "seed": 42,
  "format": "png"
}
```

Returns the image binary with appropriate `Content-Type`.

### `GET /v1/models`

List loaded models and device info.

### `GET /health`

Health check with model and device status.

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGEGEN_CHECKPOINT` | `black-forest-labs/FLUX.1-schnell` | HuggingFace model ID |
| `IMAGEGEN_GPU_MEM` | `0.9` | GPU memory fraction to use |
| `IMAGEGEN_DEFAULT_STEPS` | `4` | Default inference steps |
| `IMAGEGEN_DEFAULT_WIDTH` | `512` | Default image width |
| `IMAGEGEN_DEFAULT_HEIGHT` | `512` | Default image height |
| `IMAGEGEN_MAX_WIDTH` | `1024` | Maximum allowed width |
| `IMAGEGEN_MAX_HEIGHT` | `1024` | Maximum allowed height |
| `IMAGEGEN_DTYPE` | `bfloat16` | Torch dtype (`float32`, `float16`, `bfloat16`) |

## Docker

```bash
docker build -t giraf-imagegen .
docker run --gpus all -p 8300:8300 giraf-imagegen
```

Swap the `FROM` line in the Dockerfile for your GPU vendor (NVIDIA/AMD/Intel).

## Integration with giraf-ai

Set in giraf-ai's environment:

```bash
IMAGE_PROVIDER=imagegen
IMAGEGEN_BASE_URL=http://localhost:8300
```
