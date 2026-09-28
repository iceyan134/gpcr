# Operations Runbook

## Overview
This runbook documents the operational parameters discovered during development. It ensures reproducible execution across different environments.

## Environment Variables

The following environment variables control the pipeline:

| Variable | Purpose | Default |
|----------|---------|---------|
| `BIOHUB_TOKEN` | API token for ESMFold2 model access | Required |
| `HF_HOME` | HuggingFace model cache directory | `~/.cache/huggingface` |
| `NESSO_CACHE` | Nesso-1 embedding cache | `/tmp/nesso_cache` |
| `NESSO_CCD_PATH` | Path to CCD pickle file | Auto-download |
| `NESSO_CKPT_DIR` | Nesso-1 model weights directory | Auto-download |
| `NESSO_SCORE_TIMEOUT` | Timeout for single scoring call (seconds) | 120 |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch memory allocator config | `expandable_segments:True` |

## Docker Execution

### Recommended flags
```bash
docker run --gpus all --shm-size=16g \
  -v <repo>:/app -v <hf_cache>:/cache/huggingface \
  -w /app -e HF_HOME=/cache/huggingface \
  --entrypoint python gpcr-mdscreen \
  -m app.screen_cli --config output/config.json --out-root /app/output
```

### Critical flags
- `--shm-size=16g`: Required for Nesso-1 (default 64 MB causes deadlocks)
- `--out-root /app/output`: Mount output to host (default `/workspace/output` is container-internal)

## GPU Considerations

- OpenMM must be compiled for your GPU's compute capability
- Multiple concurrent GPU jobs require careful memory management
- See `docs/GPU_MD_SETUP.md` in the main repository for compilation instructions

## Known Issues and Solutions

1. **Nesso-1 returns 0.0 scores**: Check /dev/shm size and CCD file availability
2. **CUDA PTX version error**: Compile OpenMM from source for your architecture
3. **HF model download hangs**: Set HF_HOME to a persistent cache, use HF_HUB_OFFLINE=1 after initial download
