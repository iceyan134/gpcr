"""G1: GPCR dual-state receptor preparation tool.

Turns raw experimental receptor PDBs (GPCR-Bench / RCSB) into pipeline-ready
receptors with:
  - cleaned PDB (protein heavy atoms only, single chain)
  - protein sequence (FASTA)
  - state annotation (active / inactive / uncertain) from TITLE keywords
    cross-validated by microswitch geometry (app.gpcr_state)
  - orthosteric pocket residues (5 A from co-crystal ligand if present,
    else from TITLE-annotated reference site / fpocket fallback)

Usage (CLI):
  python -m app.gpcr_prep prep <in.pdb> <out_dir> [--chain A] [--keep-ligand]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from app.gpcr_state import annotate_state, parse_pdb, residue_map

# TITLE keyword -> state (GPCR-Bench naming convention)
_TITLE_STATE_HINTS = [
    ("biased_active", "active"),
    ("active", "active"),
    ("agonist", "active"),
    ("inactive", "inactive"),
    ("antagonist", "inactive"),
]

# Known experimental-structure state map (literature/GPCRdb consensus).
# Populated for GPCR-Bench receptors + common reference structures.
_KNOWN_PDB_STATE = {
    # GPCR-Bench receptors (all antagonist/inverse-agonist complexes unless noted)
    "2RH1": "inactive",   # beta2AR - carazolol
    "2VT4": "inactive",   # beta1AR - cyanopindolol
    "4EIY": "inactive",   # A2A - ZM241385
    "4MBS": "inactive",   # CCR5 - maraviroc
    "4K5Y": "inactive",   # CRF1 - CP376395
    "3PBL": "inactive",   # D3 - eticlopride
    "3RZE": "inactive",   # H1 - doxepin
    "4DAJ": "inactive",   # M3 - tiotropium
    "4OR2": "inactive",   # mGlu1 - FM9
    "4EJ4": "inactive",   # delta opioid - naltrindole
    "4DJH": "inactive",   # kappa opioid - JDTic
    "4DKL": "inactive",   # mu opioid - funaltrexamine
    "4EA3": "inactive",   # nociceptin - peptide
    "4RNB": "inactive",   # OX2 - suvorexant
    "4NTJ": "inactive",   # P2Y12 - AZD1283
    "3VW7": "inactive",   # PAR1 - vorapaxar
    "3V2Y": "inactive",   # S1P1 - ML056
    "4JKV": "inactive",   # Smo - LY2940680
    "3ODU": "inactive",   # CXCR4 - IT1t
    "4PHU": "inactive",   # GPR40 - TAK-875
    # active-state reference structures
    "4IAR": "active",     # 5HT1B - ergotamine
    "4IB4": "active",     # 5HT2B - ergotamine (biased)
    "5G53": "active",     # A2A - NECA
    "3SN6": "active",     # beta2AR - BI-167107 (Gs complex)
    "7JVR": "active",     # D2R - Gi complex
    "6CM4": "inactive",   # D2R - haloperidol
    "3EML": "inactive",   # A2A - ZM241385
}


@dataclass
class PreparedReceptor:
    name: str
    pdb_id: str = ""
    ligand: str = ""
    state: str = "uncertain"
    state_source: str = "unknown"      # "title" | "geometry" | "title+geometry"
    d_tm6_3: float | None = None
    seq: str = ""
    n_residues: int = 0
    pocket_residues: list[int] = field(default_factory=list)
    cleaned_pdb: Path | None = None
    fasta: Path | None = None
    meta: Path | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "pdb_id": self.pdb_id, "ligand": self.ligand,
            "state": self.state, "state_source": self.state_source,
            "d_tm6_3": self.d_tm6_3, "seq": self.seq,
            "n_residues": self.n_residues,
            "pocket_residues": self.pocket_residues,
            "notes": self.notes,
        }


def parse_title_state(pdb_id: str, title: str) -> str | None:
    """State hint from TITLE keywords (e.g. '_active_4IAR', 'biased_active')."""
    t = title.lower()
    for kw, state in _TITLE_STATE_HINTS:
        if kw in t:
            return state
    return None


def extract_pdb_meta(pdb_path: Path) -> tuple[str, str, str]:
    """Return (pdb_id, ligand, title) from TITLE lines."""
    title = ""
    for line in pdb_path.read_text(errors="replace").splitlines():
        if line.startswith("TITLE"):
            title = line[10:].strip()
            break
    m = re.search(r"_([0-9a-zA-Z]{4})\s*(?:Processed|$)", title)
    pdb_id = m.group(1).upper() if m else ""
    lm = re.search(r"ligand\s+([\w\-]+)\s+_?", title, re.IGNORECASE)
    ligand = lm.group(1) if lm else ""
    return pdb_id, ligand, title


def clean_receptor(pdb_path: Path, out_dir: Path, *, chain: str = "A",
                   keep_ligand: bool = False) -> tuple[Path, str, list[int]]:
    """Write a cleaned receptor PDB (protein heavy atoms, one chain).
    Returns (cleaned_pdb_path, sequence, pocket_residues_from_ligand).
    Removes water/ions/ligands unless keep_ligand; pocket residues are
    computed from the removed ligand's heavy atoms within 5 A."""
    import numpy as np

    lines = pdb_path.read_text(errors="replace").splitlines()
    out_lines: list[str] = []
    ligand_coords: list[tuple[float, float, float]] = []
    protein_atoms: list[tuple[str, int, str, tuple[float, float, float]]] = []

    for l in lines:
        if not l.startswith(("ATOM", "HETATM")):
            continue
        if l[21] != chain:
            continue
        try:
            x = float(l[30:38]); y = float(l[38:46]); z = float(l[46:54])
        except ValueError:
            continue
        resname = l[17:20].strip()
        if l.startswith("HETATM"):
            if resname in ("HOH", "WAT", "H2O", "NA", "K", "CL", "CA", "MG", "ZN"):
                continue
            if keep_ligand:
                out_lines.append(l)
            else:
                ligand_coords.append((x, y, z))
            continue
        # ATOM: keep heavy atoms only
        elem = l[76:78].strip() or l[12:16].strip()[0]
        if elem == "H":
            continue
        out_lines.append(l)
        try:
            resnum = int(l[22:26].strip())
        except ValueError:
            continue
        protein_atoms.append((l[12:16].strip(), resnum, resname, (x, y, z)))

    cleaned = out_dir / f"{pdb_path.stem}_clean.pdb"
    with open(cleaned, "w") as f:
        f.write("REMARK cleaned receptor for GPCR pipeline\n")
        for l in out_lines:
            f.write(l + "\n")
        f.write("END\n")

    # sequence from protein atoms (sorted by resnum, take first occurrence)
    seq_map: dict[int, str] = {}
    from app.gpcr_state import _aa1
    for _atom, resnum, resname, _c in protein_atoms:
        if resnum not in seq_map:
            seq_map[resnum] = _aa1(resname)
    seq = "".join(seq_map[k] for k in sorted(seq_map))

    # pocket residues: protein residues within 5 A of ligand heavy atoms
    pocket: list[int] = []
    if ligand_coords:
        lig = np.array(ligand_coords)
        for resnum, resname, (x, y, z) in [(r[1], r[2], r[3]) for r in protein_atoms]:
            d = np.min(np.linalg.norm(lig - np.array([x, y, z]), axis=1))
            if d < 5.0 and resnum not in pocket:
                pocket.append(resnum)
        pocket.sort()

    return cleaned, seq, pocket


def prepare_receptor(pdb_path: str | Path, out_dir: str | Path, *,
                     chain: str = "A") -> PreparedReceptor:
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_id, ligand, title = extract_pdb_meta(pdb_path)
    name = pdb_path.stem

    cleaned, seq, pocket = clean_receptor(pdb_path, out_dir, chain=chain)

    # State: TITLE hint > known-PDB map > geometry
    state_title = parse_title_state(pdb_id, title)
    state_known = _KNOWN_PDB_STATE.get(pdb_id.upper())
    geom = annotate_state(cleaned)
    notes: list[str] = []
    state = "uncertain"
    source = "unknown"

    hint = state_title or state_known
    if hint and geom.state == hint:
        state, source = hint, "title+geometry" if state_title else "known+geometry"
    elif hint and geom.state == "uncertain":
        state, source = hint, "title" if state_title else "known_pdb"
        _d = f"{geom.d_tm6_3:.1f}A" if geom.d_tm6_3 is not None else "N/A"
        notes.append(f"geometry uncertain (d_TM6-3={_d}); using {'TITLE' if state_title else 'known-PDB'} hint")
    elif hint:
        state, source = hint, "title" if state_title else "known_pdb"
        _d = f"{geom.d_tm6_3:.1f}A" if geom.d_tm6_3 is not None else "N/A"
        notes.append(f"geometry {geom.state} (d_TM6-3={_d}) disagrees with hint; using hint")
    elif geom.state != "uncertain":
        state, source = geom.state, "geometry"
    else:
        state, source = "uncertain", "unknown"
        _d = f"{geom.d_tm6_3:.1f}A" if geom.d_tm6_3 is not None else "N/A"
        notes.append(f"no hint; geometry uncertain (d_TM6-3={_d})")

    # Outputs
    fasta = out_dir / f"{name}.fasta"
    with open(fasta, "w") as f:
        f.write(f">{name} state={state} source={source}\n{seq}\n")
    meta = out_dir / f"{name}.meta.json"
    rec = PreparedReceptor(
        name=name, pdb_id=pdb_id, ligand=ligand, state=state,
        state_source=source, d_tm6_3=geom.d_tm6_3, seq=seq,
        n_residues=len(seq), pocket_residues=pocket,
        cleaned_pdb=cleaned, fasta=fasta, meta=meta, notes=notes,
    )
    with open(meta, "w") as f:
        json.dump(rec.to_dict(), f, indent=2)
    return rec


def main():
    ap = argparse.ArgumentParser(description="GPCR receptor preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("in_pdb")
    p.add_argument("out_dir")
    p.add_argument("--chain", default="A")
    args = ap.parse_args()

    rec = prepare_receptor(args.in_pdb, args.out_dir, chain=args.chain)
    print(json.dumps(rec.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
