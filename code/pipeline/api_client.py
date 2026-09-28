"""
Biohub API client for cloud-based ESMFold2 inference.
Used as alternative when local GPU is not available or for fast mode.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Optional

from app.models import (
    ChainInput,
    FoldingConfig,
    MoleculeType,
    PredictionRequest,
    PredictionResult,
)


class BiohubAPIClient:
    def __init__(
        self,
        token: Optional[str] = None,
        url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.token = token or os.environ.get("BIOHUB_TOKEN")
        self.url = url or os.environ.get("BIOHUB_API_URL", "https://biohub.ai")
        self.model_id = model or os.environ.get("ESMFOLD2_FAST_MODEL_ID", "esmfold2-fast-2026-05")

    def _get_client(self):
        from esm.sdk.forge import SequenceStructureForgeInferenceClient
        return SequenceStructureForgeInferenceClient(
            model=self.model_id,
            url=self.url,
            token=self.token,
        )

    def _build_input(self, chains: list[ChainInput]):
        from esm.utils.structure.input_builder import (
            DNAInput,
            LigandInput,
            ProteinInput,
            RNAInput,
            StructurePredictionInput,
        )

        sequences = []
        for chain in chains:
            if chain.type == MoleculeType.protein:
                sequences.append(ProteinInput(id=chain.id, sequence=chain.sequence))
            elif chain.type == MoleculeType.dna:
                sequences.append(DNAInput(id=chain.id, sequence=chain.sequence))
            elif chain.type == MoleculeType.rna:
                sequences.append(RNAInput(id=chain.id, sequence=chain.sequence))
            elif chain.type == MoleculeType.ligand:
                sequences.append(LigandInput(id=chain.id, ccd=chain.ccd))

        return StructurePredictionInput(sequences=sequences)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        if not self.token:
            raise ValueError(
                "BIOHUB_TOKEN not set. Get your token at https://biohub.ai/developer-console"
            )

        t0 = time.time()
        job_id = uuid.uuid4().hex[:12]

        client = self._get_client()
        spi = self._build_input(request.chains)
        cfg = request.config

        from esm.sdk.api import FoldingConfig as APIFoldingConfig

        api_config = APIFoldingConfig(
            num_loops=cfg.num_loops,
            num_sampling_steps=cfg.num_sampling_steps,
        )

        result = client.fold_all_atom(spi, config=api_config)

        mmcif_str = result.mmcif
        plddt_mean = float(result.plddt.mean()) if hasattr(result, 'plddt') else 0.0
        ptm = float(result.ptm) if hasattr(result, 'ptm') else 0.0
        iptm = float(result.iptm) if hasattr(result, 'iptm') else None

        elapsed = time.time() - t0

        return PredictionResult(
            job_id=job_id,
            name=request.name,
            mmcif=mmcif_str,
            plddt_mean=round(plddt_mean, 4),
            ptm=round(ptm, 4),
            iptm=round(iptm, 4) if iptm else None,
            num_chains=len(request.chains),
            num_residues=0,
            wall_time_s=round(elapsed, 1),
        )


_api_client: Optional[BiohubAPIClient] = None


def get_api_client() -> BiohubAPIClient:
    global _api_client
    if _api_client is None:
        _api_client = BiohubAPIClient()
    return _api_client
