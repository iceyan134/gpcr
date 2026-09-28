# Build Instructions

## Prerequisites
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with at least 24 GB VRAM
- 100 GB disk space for models and data

## Quick Start
```bash
make build       # Build the Docker image
make cli         # Run interactive CLI
make up          # Start the API service
make test        # Run unit tests
```

## Manual Build
```bash
docker build -t gpcr-mdscreen -f docker/Dockerfile .
docker run --gpus all -v $(pwd):/app -w /app gpcr-mdscreen python -m app.screen_cli --config configs/screen_example.json
```

## GPU Architecture Support
The base image supports compute capability >= 7.0. For Blackwell (SM 12.0) GPUs, see `docker/Dockerfile` for the custom CUDA compilation flags.
