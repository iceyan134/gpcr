"""
Online SMILES → CCD code lookup engine.
Uses PubChem + PDBe/RCSB APIs with local database validation.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_cache: dict[str, Optional[str]] = {}


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return smiles


def _smiles_similarity(smi1: str, smi2: str) -> float:
    try:
        from rdkit import Chem, DataStructs
        m1 = Chem.MolFromSmiles(smi1)
        m2 = Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = Chem.RDKFingerprint(m1)
        fp2 = Chem.RDKFingerprint(m2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def _validate_ccd(ccd_code: str) -> bool:
    ccd_code = ccd_code.upper().strip()
    if len(ccd_code) != 3:
        return False
    try:
        url = f"https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/{ccd_code}"
        r = requests.get(url, timeout=10)
        return r.status_code == 200 and len(r.text) > 100
    except Exception:
        return False


def _get_ccd_smiles_from_pdbe(ccd_code: str) -> Optional[str]:
    try:
        url = f"https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/{ccd_code}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            comp_data = data.get(ccd_code.upper(), data.get(ccd_code.lower(), {}))
            if isinstance(comp_data, list) and comp_data:
                comp_data = comp_data[0]
            smiles = comp_data.get("smiles", None)
            if smiles:
                if isinstance(smiles, dict):
                    return smiles.get("name")
                if isinstance(smiles, list):
                    item = smiles[0]
                    if isinstance(item, dict):
                        return item.get("name")
                    return item
                return smiles
    except Exception:
        pass
    return None


def _pubchem_smiles_to_cid(smiles: str) -> Optional[int]:
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/cids/JSON"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return cids[0]
    except Exception as e:
        logger.debug(f"PubChem CID lookup failed: {e}")
    return None


def _pubchem_cid_to_synonyms(cid: int) -> list[str]:
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
    except Exception:
        return []


def _chembl_lookup(smiles: str) -> Optional[dict]:
    """ChEMBL API SMILES → PDB cross-reference."""
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={encoded}&format=json&limit=3"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        for mol in data.get("molecules", [])[:3]:
            mol_id = mol.get("molecule_chembl_id")
            if not mol_id:
                continue
            xref_r = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json", timeout=10)
            if xref_r.status_code != 200:
                continue
            for xref in xref_r.json().get("cross_references", []):
                src = str(xref.get("xref_src", "")).lower()
                xref_id = str(xref.get("xref_id", "")).strip()
                if ("pdb" in src or "ccd" in src) and len(xref_id) == 3 and xref_id.isalpha():
                    ccd = xref_id.upper()
                    if _validate_ccd(ccd):
                        ccd_smiles = _get_ccd_smiles_from_pdbe(ccd)
                        return {"ccd_code": ccd, "exact_smiles": ccd_smiles,
                                "match_method": f"chembl_{mol_id}"}
        return None
    except Exception:
        return None


def lookup_smiles_online(smiles: str) -> Optional[dict]:
    """
    Full online SMILES → CCD lookup.
    Collects candidates from PubChem + local DB, ranks by SMILES similarity.
    """
    canonical = _canonicalize_smiles(smiles)
    if not canonical:
        return None

    cache_key = canonical
    if cache_key in _cache:
        cached = _cache[cache_key]
        if cached:
            return {"ccd_code": cached, "exact_smiles": None,
                    "match_method": "cache", "similarity": 1.0}
        return None

    all_matches: list[tuple[str, str, float, str]] = []  # (ccd, smiles, score, source)

    # Source 1: PubChem CID → synonyms → 3-letter candidates
    cid = _pubchem_smiles_to_cid(canonical)
    if cid:
        synonyms = _pubchem_cid_to_synonyms(cid)
        ccd_candidates = [s.strip().upper() for s in synonyms
                          if len(s.strip()) == 3 and s.strip().isalpha() and s.strip().isupper()]
        for ccd in ccd_candidates:
            if _validate_ccd(ccd):
                ccd_smiles = _get_ccd_smiles_from_pdbe(ccd)
                if ccd_smiles:
                    score = _smiles_similarity(canonical, ccd_smiles)
                    all_matches.append((ccd, ccd_smiles, score, f"pubchem_{cid}"))

    # Source 2: Local database (catches well-known codes PubChem might miss)
    from app.ccd_database import get_ccd_db
    db = get_ccd_db()
    for local_ccd, local_smi in db.ccd_to_smiles.items():
        score = _smiles_similarity(canonical, local_smi)
        if score > 0.8:
            all_matches.append((local_ccd, local_smi, score, "local_db"))

    # Source 3: Common CCD codes via PDBe
    common = ["ATP", "ADP", "GTP", "NAD", "FAD", "HEM", "SAM", "COA", "PLP", "FMN",
              "BNZ", "TOL", "GOL", "ACT", "DMS", "CFF", "SAH", "PO4", "SO4"]
    for ccd in common:
        if ccd not in db.ccd_to_smiles:
            ccd_smiles = _get_ccd_smiles_from_pdbe(ccd)
            if ccd_smiles:
                score = _smiles_similarity(canonical, ccd_smiles)
                if score > 0.8:
                    all_matches.append((ccd, ccd_smiles, score, "pdbe_common"))

    # Rank: highest score first; prefer well-known codes for ties
    preferred = {"ATP", "ADP", "GTP", "NAD", "FAD", "HEM", "SAM", "COA", "PLP",
                 "BNZ", "TOL", "GOL", "CFF", "SAH", "PO4"}
    all_matches.sort(key=lambda x: (-x[2], x[0] not in preferred))

    # Return best high-confidence match
    if all_matches and all_matches[0][2] > 0.9:
        ccd, ccd_smiles, score, source = all_matches[0]
        _cache[cache_key] = ccd
        logger.info(f"Online match: {smiles[:40]}... → {ccd} (score={score:.3f}, {source})")
        return {"ccd_code": ccd, "exact_smiles": ccd_smiles,
                "match_method": source, "similarity": round(score, 4)}

    # Accept partial match
    if all_matches and all_matches[0][2] > 0.5:
        ccd, ccd_smiles, score, source = all_matches[0]
        _cache[cache_key] = ccd
        return {"ccd_code": ccd, "exact_smiles": ccd_smiles,
                "match_method": f"{source}_partial", "similarity": round(score, 4)}

    # Source 4: ChEMBL as last resort
    chembl_result = _chembl_lookup(canonical)
    if chembl_result:
        _cache[cache_key] = chembl_result["ccd_code"]
        return chembl_result

    _cache[cache_key] = None
    return None


def search_compound(smiles: str) -> dict:
    """
    Comprehensive SMILES → CCD search.
    Online-first (PubChem/PDBe/ChEMBL) → local similarity fallback.
    """
    canonical = _canonicalize_smiles(smiles)

    online_result = lookup_smiles_online(smiles)
    if online_result:
        return {
            "match_type": "exact" if online_result.get("similarity", 0) > 0.9 else "similar",
            "ccd_code": online_result["ccd_code"],
            "canonical_smiles": canonical,
            "match_method": online_result.get("match_method", "online"),
            "similarity": online_result.get("similarity"),
        }

    # Fallback: local similarity search
    from app.ccd_database import get_ccd_db
    db = get_ccd_db()
    similar = db.lookup_similar(smiles, threshold=0.4)
    candidates = [{"ccd": c, "smiles": s, "score": sc} for c, s, sc in similar]

    return {
        "match_type": "similar" if candidates else "none",
        "ccd_code": candidates[0]["ccd"] if candidates else None,
        "canonical_smiles": canonical,
        "candidates": candidates[:10],
    }
