"""L1.5 drug-likeness ANNOTATION layer (not filter).

Philosophy: help the user decide, never decide for the user. All molecules
pass through to the expensive screening stage (recall-first, §2.1); this
layer attaches multi-dimensional drug-likeness evidence to every candidate.

Dimensions (each with measured discriminative power on L1000-vs-L6000):
  - QED score            (AUC 0.601 - only positive discriminative score)
  - Lipinski violations  (count + per-rule detail)
  - Veber violations     (count + per-rule detail)
  - PAINS warning        (Baell 2010: high FP rate; 5.5% of approved drugs hit)
  - DBPP-Predictor prob  (AUC 0.338 REVERSED - kept as evidence, not decision)
  - Raw properties       (MW/logP/HBD/HBA/RotB/TPSA - user can judge)

No molecule is removed. Output: per-mol annotation dict + CSV table.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_pains_catalog = None
_dbpp_model = None
_dbpp_gen = None


def _get_pains():
    global _pains_catalog
    if _pains_catalog is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_catalog = FilterCatalog.FilterCatalog(params)
    return _pains_catalog


def _get_dbpp():
    """Lazy-load DBPP-Predictor (GBM_Descriptor_Norm). Returns None if
    dependencies/model unavailable - annotation continues without it."""
    global _dbpp_model, _dbpp_gen
    if _dbpp_model is not None or _dbpp_gen is not None:
        return _dbpp_model, _dbpp_gen
    try:
        import pickle
        import numpy as np
        from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
        _dbpp_gen = MakeGenerator(("rdkit2dnormalized",))
        model_path = (
            Path(__file__).resolve().parent.parent
            / "third_party" / "DBPP-Predictor"
            / "Models" / "models" / "Druglike_GBM_Descriptor_Norm.model"
        )
        if model_path.exists():
            with open(model_path, "rb") as f:
                _dbpp_model = pickle.load(f)
    except Exception:
        _dbpp_model, _dbpp_gen = None, None
    return _dbpp_model, _dbpp_gen


@dataclass
class L15Annotation:
    smiles: str
    parse_ok: bool
    qed: Optional[float] = None
    lipinski_violations: int = 0
    lipinski_detail: Optional[list] = None
    veber_violations: int = 0
    veber_detail: Optional[list] = None
    pains: Optional[bool] = None
    dbpp_prob: Optional[float] = None
    dbpp_available: bool = False
    mw: Optional[float] = None
    logp: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    rotb: Optional[int] = None
    tpsa: Optional[float] = None


def annotate(smiles: str) -> L15Annotation:
    """Attach drug-likeness annotations to one SMILES. Never removes."""
    ann = L15Annotation(smiles=smiles, parse_ok=False)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ann
    ann.parse_ok = True
    ann.mw = Descriptors.MolWt(mol)
    ann.logp = Descriptors.MolLogP(mol)
    ann.hbd = Lipinski.NumHDonors(mol)
    ann.hba = Lipinski.NumHAcceptors(mol)
    ann.rotb = Descriptors.NumRotatableBonds(mol)
    ann.tpsa = rdMolDescriptors.CalcTPSA(mol)
    ann.qed = QED.qed(mol)

    # Lipinski (rule-of-5)
    lip = []
    if ann.mw > 500: lip.append("MW>500")
    if ann.logp > 5: lip.append("logP>5")
    if ann.hbd > 5: lip.append("HBD>5")
    if ann.hba > 10: lip.append("HBA>10")
    ann.lipinski_detail = lip
    ann.lipinski_violations = len(lip)

    # Veber
    veb = []
    if ann.rotb > 10: veb.append("RotB>10")
    if ann.tpsa > 140: veb.append("TPSA>140")
    ann.veber_detail = veb
    ann.veber_violations = len(veb)

    # PAINS (warning only)
    ann.pains = bool(_get_pains().HasMatch(mol))

    # DBPP-Predictor (kept as weak evidence; AUC 0.338 reversed - not decision)
    model, gen = _get_dbpp()
    if model is not None and gen is not None:
        try:
            res = gen.process(smiles)
            if res[0]:
                prob = model.predict_proba([res[1:]])[0, 1]
                ann.dbpp_prob = float(prob)
                ann.dbpp_available = True
        except Exception:
            pass
    return ann


def annotate_batch(smiles_list: list[str]) -> list[L15Annotation]:
    return [annotate(s) for s in smiles_list]


def to_records(anns: list[L15Annotation]) -> list[dict]:
    """Flatten for CSV/JSON export. dbpp col = score or NA."""
    out = []
    for a in anns:
        d = asdict(a)
        d["dbpp_prob"] = d["dbpp_prob"] if d["dbpp_available"] else None
        out.append(d)
    return out


def export_table(anns: list[L15Annotation], path: str) -> None:
    import csv
    recs = to_records(anns)
    fields = ["smiles", "qed", "lipinski_violations", "lipinski_detail",
              "veber_violations", "veber_detail", "pains", "dbpp_prob",
              "mw", "logp", "hbd", "hba", "rotb", "tpsa"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            row = {k: r.get(k) for k in fields}
            row["lipinski_detail"] = ";".join(row["lipinski_detail"] or [])
            row["veber_detail"] = ";".join(row["veber_detail"] or [])
            w.writerow(row)
    print(f"L1.5 annotation table -> {path} ({len(recs)} mols)")


if __name__ == "__main__":
    import sys
    smis = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    anns = annotate_batch(smis)
    out_path = sys.argv[2] if len(sys.argv) > 2 else "l15_annotation.csv"
    export_table(anns, out_path)
