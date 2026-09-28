"""
SMILES to PDB CCD code lookup database.
Downloads the PDB Chemical Component Dictionary and builds a searchable local index.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Cache locations
CCD_CACHE_DIR = Path("/cache/ccd")
CCD_INDEX_FILE = CCD_CACHE_DIR / "smiles_to_ccd.json"
CCD_SOURCES = [
    "https://ftp.wwpdb.org/pub/pdb/data/monomers/ccd-to-smiles.json",
    "https://files.rcsb.org/pub/pdb/data/monomers/ccd-to-smiles.json",
]
# PDBe API for online SMILES→CCD lookup
PDBE_SMILES_API = "https://www.ebi.ac.uk/pdbe/graph-api/compound/smiles/{smiles}"
RCSB_CHEMCOMP_API = "https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_id}"


class CCDDatabase:
    def __init__(self):
        self.smiles_to_ccd: dict[str, str] = {}  # canonical SMILES → CCD code
        self.ccd_to_smiles: dict[str, str] = {}  # CCD code → SMILES
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        CCD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if CCD_INDEX_FILE.exists():
            self._load_from_cache()
        else:
            self._download_and_build()
        self._loaded = True
        logger.info(f"CCD database ready: {len(self.smiles_to_ccd)} compounds indexed")

    def _load_from_cache(self):
        with open(CCD_INDEX_FILE) as f:
            data = json.load(f)
        self.smiles_to_ccd = data.get("smiles_to_ccd", {})
        self.ccd_to_smiles = data.get("ccd_to_smiles", {})
        logger.info(f"Loaded {len(self.smiles_to_ccd)} CCD entries from cache")

    def _download_and_build(self):
        """Download the CCD SMILES mapping from multiple mirrors."""
        logger.info("Downloading CCD-to-SMILES mapping...")
        raw = None
        for url in CCD_SOURCES:
            try:
                logger.info(f"Trying: {url}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                raw = resp.json()
                logger.info(f"Successfully downloaded from {url}")
                break
            except Exception as e:
                logger.warning(f"Failed: {url} — {e}")
                continue

        if raw is None:
            logger.warning("All CCD download sources failed, using built-in database")
            raw = _BUILTIN_CCD_SMILES

        # Build both indexes
        for ccd_code, smiles in raw.items():
            if not smiles or not ccd_code or len(ccd_code) != 3:
                continue
            canonical = self._canonicalize(smiles)
            if canonical:
                self.ccd_to_smiles[ccd_code.upper()] = canonical
                # First-come-first-served for SMILES → CCD
                if canonical not in self.smiles_to_ccd:
                    self.smiles_to_ccd[canonical] = ccd_code.upper()

        # Save to cache
        with open(CCD_INDEX_FILE, "w") as f:
            json.dump({
                "smiles_to_ccd": self.smiles_to_ccd,
                "ccd_to_smiles": self.ccd_to_smiles,
            }, f, indent=2)
        logger.info(f"Built CCD index: {len(self.smiles_to_ccd)} unique SMILES")

    def _canonicalize(self, smiles: str) -> Optional[str]:
        """Canonicalize a SMILES string using RDKit."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return smiles.strip()

    def lookup_exact(self, smiles: str) -> Optional[str]:
        """Exact SMILES match → CCD code."""
        self._ensure_loaded()
        canonical = self._canonicalize(smiles)
        if canonical and canonical in self.smiles_to_ccd:
            return self.smiles_to_ccd[canonical]
        return None

    def lookup_similar(self, smiles: str, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Similarity search → list of (CCD code, SMILES, Tanimoto score)."""
        self._ensure_loaded()
        try:
            from rdkit import Chem, DataStructs
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            fp = Chem.RDKFingerprint(mol)
            results = []
            for ccd, ccd_smiles in self.ccd_to_smiles.items():
                ccd_mol = Chem.MolFromSmiles(ccd_smiles)
                if ccd_mol is None:
                    continue
                ccd_fp = Chem.RDKFingerprint(ccd_mol)
                score = DataStructs.TanimotoSimilarity(fp, ccd_fp)
                if score >= threshold:
                    results.append((ccd, ccd_smiles, round(score, 4)))
            results.sort(key=lambda x: -x[2])
            return results[:20]
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def _online_lookup(self, smiles: str) -> Optional[dict]:
        """Try online SMILES→CCD lookup via PubChem + PDBe APIs."""
        import urllib.parse
        canonical = self._canonicalize(smiles)
        if not canonical:
            return None

        # Step 1: PubChem SMILES → CID
        try:
            encoded = urllib.parse.quote(canonical, safe="")
            pc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/cids/JSON"
            resp = requests.get(pc_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                cids = data.get("IdentifierList", {}).get("CID", [])
                if cids:
                    cid = cids[0]
                    # Step 2: PubChem CID → synonyms → look for 3-letter CCD code
                    syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
                    syn_resp = requests.get(syn_url, timeout=10)
                    if syn_resp.status_code == 200:
                        syn_data = syn_resp.json()
                        synonyms = syn_data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
                        for syn in synonyms:
                            syn_upper = syn.strip().upper()
                            if len(syn_upper) == 3 and syn_upper.isalpha():
                                if self.is_known_ccd(syn_upper):
                                    return {
                                        "match_type": "exact",
                                        "ccd_code": syn_upper,
                                        "smiles": canonical,
                                        "source": "pubchem_pdbe",
                                    }
        except Exception:
            pass
        return None

    def search(self, smiles: str) -> dict:
        """
        Search for CCD code matching a SMILES string.
        Checks local database first, then tries online API.
        """
        exact = self.lookup_exact(smiles)
        if exact:
            return {
                "match_type": "exact",
                "ccd_code": exact,
                "smiles": self.ccd_to_smiles.get(exact, smiles),
            }

        # Try online lookup before falling back to similarity
        online = self._online_lookup(smiles)
        if online:
            return online

        similar = self.lookup_similar(smiles)
        return {
            "match_type": "similar" if similar else "none",
            "ccd_code": similar[0][0] if similar else None,
            "candidates": [{"ccd": c, "smiles": s, "score": sc} for c, s, sc in similar[:5]],
        }

    def is_known_ccd(self, ccd_code: str) -> bool:
        """Check if a CCD code exists in the database."""
        self._ensure_loaded()
        return ccd_code.upper() in self.ccd_to_smiles

    def get_smiles(self, ccd_code: str) -> Optional[str]:
        """Get SMILES for a CCD code."""
        self._ensure_loaded()
        return self.ccd_to_smiles.get(ccd_code.upper())


# Import comprehensive CCD database
from app.ccd_data import CCD_SMILES as _BUILTIN_CCD_SMILES  # noqa: E402


_db: Optional[CCDDatabase] = None


def get_ccd_db() -> CCDDatabase:
    global _db
    if _db is None:
        _db = CCDDatabase()
    return _db
