#!/bin/bash
set -e
export LD_LIBRARY_PATH=/opt/conda/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-12/lib64:${LD_LIBRARY_PATH:-}

case "${1:-api}" in
    api)
        echo "Starting ESMFold2 API server..."
        exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
        ;;
    cli)
        shift
        exec python -m app.cli "$@"
        ;;
    batch)
        shift
        exec python -m app.batch "$@"
        ;;
    *)
        echo "Usage: $0 {api|cli|batch} [args...]"
        exit 1
        ;;
esac
