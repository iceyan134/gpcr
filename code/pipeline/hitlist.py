"""Hit-list post-processing: diversity, reactivity alerts, OOD annotation.

Philosophy (project rule): annotate, never silently filter — the L1.5 layer
helps the user decide instead of deciding for them. Every function returns
labels/picks; the input records are never dropped without an explicit reason
attached.

Evidence base:
  - GPR146 top-20 contained same-scaffold near-duplicates (#1/#2, #12/#13;
    identical QED/MW/logP to 3 decimals) — 20 slots carried <18 chemotypes.
  - GCGR top-20 contained haloethyl-nitrogen mustards, boronate esters,
    multiple nitro groups — PAINS alone does not cover reactive/covalent
    chemotypes that would waste wet-lab budget.
  - >550 Da ligands score low systematically (OOD blind spot, dual-target
    validated) — surface as a run-time flag instead of a paper footnote.
"""
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

OOD_MW_THRESHOLD = 550.0  # dual-target validated blind spot (MK-0893 573, C073 578)
DIVERSITY_SIMILARITY = 0.6  # ECFP4 Tanimoto above this = same chemotype cluster

# Heuristic reactive/covalent alerts (annotation, not filter). Named patterns
# cover the chemotypes actually observed in our hit lists.
REACTIVE_SMARTS: dict[str, str] = {
    "nitrogen_mustard": "[NX3;H0,H1,H2]-[CH2]-[CH2]-[Cl,Br]",
    "nitro": "[$([#6]-[NX3](=O)=O),$([#6]-[NX3+](=O)[O-])]",
    "nitroso": "[#6]-[NX2]=O",
    "michael_acceptor": "[#6]=[#6]-[#6]=[#8]",
    "boronate": "[#5]-[OX2]",
    "acyl_halide": "[CX3](=[OX1])-[Cl,Br,F]",
    "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])-[Cl,Br,F]",
    "epoxide": "[CH2]1[CH2][OX2]1",
    "aziridine": "[CH2]1[CH2][NX3;H0,H1,H2]1",
    "alkyl_sulfonate_leaving": "[SX4](=[OX1])(=[OX1])-[CH2]-[CH2]-[OX2]-[CH3]",
}
_COMPILED = {name: Chem.MolFromSmarts(sma) for name, sma in REACTIVE_SMARTS.items()}


def reactive_alerts(smiles: str) -> list[str]:
    """Names of reactive patterns matched (possibly empty)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["unparseable_smiles"]
    return sorted(name for name, patt in _COMPILED.items()
                  if patt is not None and mol.HasSubstructMatch(patt))


def ood_flag(smiles: str, threshold: float = OOD_MW_THRESHOLD) -> bool:
    """MW above threshold → outside Nesso/ESMFold2 training comfort zone."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return Descriptors.MolWt(mol) > threshold


def _fp(smiles: str, radius: int = 2, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def diversity_pick(
    smiles_scored: list[tuple[str, float]],
    k: int,
    similarity: float = DIVERSITY_SIMILARITY,
) -> tuple[list[tuple[str, float]], list[tuple[str, str]]]:
    """Greedy diversity pick over a score-ranked list.

    Returns (picks, skipped) where skipped carries (smiles, reason) for every
    candidate excluded by near-duplicate clustering — nothing is silently
    dropped.
    """
    ranked = sorted(smiles_scored, key=lambda kv: -kv[1])
    picks: list[tuple[str, float]] = []
    picked_fps = []
    skipped: list[tuple[str, str]] = []
    for smi, score in ranked:
        if len(picks) >= k:
            skipped.append((smi, "beyond_k"))
            continue
        fp = _fp(smi)
        if fp is None:
            skipped.append((smi, "unparseable_smiles"))
            continue
        dup_of = next(
            (p[0] for p, pf in zip(picks, picked_fps)
             if DataStructs.TanimotoSimilarity(fp, pf) >= similarity),
            None,
        )
        if dup_of is not None:
            skipped.append((smi, f"near_duplicate(tanimoto>={similarity})_of:{dup_of[:50]}"))
            continue
        picks.append((smi, score))
        picked_fps.append(fp)
    return picks, skipped


def z_prime(positive_scores: list[float], null_scores: list[float]) -> float | None:
    """HTS assay-window quality (Z' factor) for one screening run.

    Z' = 1 - 3(sp + sn) / |mp - mn|;  >0.5 excellent, 0-0.5 usable, <0 no
    window. positive_scores: replicate measurements of the positive control
    (across runs or across samples); null_scores: assumed non-binders
    (library sample). Returns None if either side is degenerate.
    """
    p = [s for s in positive_scores if s is not None]
    n = [s for s in null_scores if s is not None]
    if len(p) < 2 or len(n) < 2:
        return None
    import numpy as np
    mp, sp = float(np.mean(p)), float(np.std(p))
    mn, sn = float(np.mean(n)), float(np.std(n))
    if abs(mp - mn) < 1e-12:
        return float("-inf")
    return 1.0 - 3.0 * (sp + sn) / abs(mp - mn)


def annotate_hits(records: list[dict]) -> list[dict]:
    """Annotate hit records ({'smiles':..., 'binary':...}) in place.

    Adds: reactive_alerts (list), mw, ood_risk (bool).
    """
    for r in records:
        smi = r.get("smiles", "")
        mol = Chem.MolFromSmiles(smi)
        r["reactive_alerts"] = reactive_alerts(smi)
        r["mw"] = round(Descriptors.MolWt(mol), 1) if mol is not None else None
        r["ood_risk"] = ood_flag(smi)
    return records
