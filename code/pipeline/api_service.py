"""
ESMFold2 API-based FastAPI application (Biohub cloud inference).
Separate entrypoint for docker-compose API-only profile.
"""
from __future__ import annotations

import os

# Force API mode
os.environ["ESMFOLD2_MODE"] = "api"

from app.main import app  # noqa: E402, F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
