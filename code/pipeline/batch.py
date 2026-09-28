"""
Batch processing worker for ESMFold2.
Pulls jobs from Redis queue, runs predictions on GPU, stores results.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import redis

from app.models import PredictionRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch")


class BatchWorker:
    def __init__(self, redis_url: str, output_dir: str = "/output"):
        self.redis = redis.from_url(redis_url)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.running = True

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutting down...")
        self.running = False

    def run(self):
        from app.local_inference import LocalInferenceEngine

        logger.info("Loading model...")
        engine = LocalInferenceEngine()

        logger.info("Worker ready, waiting for jobs...")
        while self.running:
            try:
                job_id = self.redis.brpop("queue:pending", timeout=5)
                if job_id is None:
                    continue

                _, job_id = job_id
                job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                self._process_job(job_id, engine)

            except redis.ConnectionError:
                logger.error("Redis connection lost, retrying...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

    def _process_job(self, job_id: str, engine):
        data = self.redis.get(f"job:{job_id}")
        if data is None:
            logger.warning(f"Job {job_id} not found in store")
            return

        job = json.loads(data)
        job["status"] = "running"
        self.redis.set(f"job:{job_id}", json.dumps(job))

        logger.info(f"Processing job {job_id}: {job['request']['name']}")

        try:
            request = PredictionRequest(**job["request"])
            result = engine.predict(request)

            out_path = self.output_dir / f"{result.name}_{result.job_id}.cif"
            out_path.write_text(result.mmcif)

            job["status"] = "completed"
            job["result"] = result.model_dump()
            del job["result"]["mmcif"]  # don't store mmCIF in redis
            job["result"]["mmcif"] = f"Saved to {out_path}"

            logger.info(f"Job {job_id} complete — pLDDT: {result.plddt_mean}, pTM: {result.ptm}")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job["status"] = "failed"
            job["error"] = str(e)

        self.redis.set(f"job:{job_id}", json.dumps(job))


def main():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    output_dir = os.environ.get("OUTPUT_DIR", "/output")
    worker = BatchWorker(redis_url=redis_url, output_dir=output_dir)
    worker.run()


if __name__ == "__main__":
    main()
