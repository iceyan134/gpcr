"""
Receptor alignment for pose-validation export (corrected plan revision #3).

Predicted co-fold poses live in the predicted receptor's coordinate frame.
Before computing ligand RMSD to a crystal reference, the predicted receptor is
superimposed onto the reference receptor (via Ca atoms) and the SAME rotation +
translation is applied to the predicted ligand. This is receptor-to-receptor
rigid-body superposition (NOT ligand self-alignment, which would cheat).

Ca CORRESPONDENCE: ESMFold2 numbers residues 1..N; a crystal PDB uses author
numbering (e.g. 696..) and is frequently MISSING residues (disordered loops).
So neither residue-number matching nor positional "first-n" matching is safe --
one missing residue shifts every later pair. We therefore establish the
correspondence by GLOBAL SEQUENCE ALIGNMENT and keep only aligned (non-gap)
columns. This is robust to terminal and internal gaps and to length differences.

Deps: numpy (core/Kabsch); Biopython (PDB parsing + Bio.Align); RDKit (write SDF).
Logic check: python structure_align.py --self-test
Run on real files (pocket-local superposition is recommended for pose RMSD):
  python structure_align.py --pred-receptor pred.cif --pred-ligand lig.sdf \
      --ref-receptor xtal.pdb --ref-ligand xtal_lig.sdf --pocket-radius 10 \
      --out aligned_lig.sdf
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ===========================================================================
# Ca extraction (per chain, sorted by residue number)
# ===========================================================================
def _extract_ca_from_mmcif(mmcif_path: str):
    """Parse an mmCIF (e.g. ESMFold2 output) for protein Ca atoms.

    Returns dict[chain_id -> (residue_names, ca_coords)], each sorted by
    label_seq_id. Manual parser (tolerates missing _atom_site sub-fields).
    """
    with open(mmcif_path) as fh:
        lines = fh.readlines()

    field_idx: dict[str, int] = {}
    for line in lines:
        if line.startswith("_atom_site."):
            field_idx[line.strip().split(".", 1)[1]] = len(field_idx)
        elif line.startswith(("ATOM", "HETATM")):
            break

    grp = field_idx.get("group_PDB")
    chain_f = field_idx.get("label_asym_id")
    resname_f = field_idx.get("label_comp_id")
    resseq_f = field_idx.get("label_seq_id")
    atom_f = field_idx.get("label_atom_id")
    x_f, y_f, z_f = field_idx.get("Cartn_x"), field_idx.get("Cartn_y"), field_idx.get("Cartn_z")
    elem_f = field_idx.get("type_symbol")

    required = [grp, chain_f, resname_f, resseq_f, atom_f, x_f, y_f, z_f]
    if any(v is None for v in required):
        raise KeyError(f"mmCIF missing required _atom_site fields. Found: {list(field_idx)}")
    min_cols = max(required + ([elem_f] if elem_f is not None else [])) + 1

    chain_data: dict[str, list[tuple[int, str, np.ndarray]]] = {}
    for line in lines:
        s = line.strip()
        if not s or s[0] == "#" or s.startswith(("data_", "loop_", "_")):
            continue
        parts = s.split()
        if len(parts) < min_cols or parts[grp] != "ATOM":
            continue
        resname = parts[resname_f]
        if resname not in _STANDARD_AA or parts[atom_f] != "CA":
            continue
        if elem_f is not None and parts[elem_f] == "H":
            continue
        try:
            resseq = int(parts[resseq_f])
            coord = np.array([float(parts[x_f]), float(parts[y_f]), float(parts[z_f])], dtype=float)
        except ValueError:
            continue
        chain_data.setdefault(parts[chain_f], []).append((resseq, resname, coord))

    return _finalize_chains(chain_data)


def _extract_ca_from_pdb(pdb_path: str):
    """Extract protein Ca atoms from a standard PDB via Biopython."""
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    chain_data: dict[str, list[tuple[int, str, np.ndarray]]] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":      # skip hetero / water
                    continue
                resname = residue.get_resname().strip()
                if resname not in _STANDARD_AA or "CA" not in residue:
                    continue
                chain_data.setdefault(chain.get_id(), []).append(
                    (residue.get_id()[1], resname, np.array(residue["CA"].get_coord(), dtype=float)))
        break  # first model only
    return _finalize_chains(chain_data)


def _finalize_chains(chain_data):
    result: dict[str, tuple[list[str], list[np.ndarray]]] = {}
    for cid, entries in chain_data.items():
        entries.sort(key=lambda e: e[0])
        result[cid] = ([e[1] for e in entries], [e[2] for e in entries])
    return result


def _extract_ca(path: str | Path):
    """Dispatch by extension: mmCIF (manual) vs PDB (Biopython)."""
    p = str(path)
    return _extract_ca_from_mmcif(p) if p.endswith((".cif", ".mmcif")) else _extract_ca_from_pdb(p)


# ===========================================================================
# Sequence-alignment-based Ca correspondence (robust to gaps)
# ===========================================================================
def _aligned_index_pairs(seq_pred: str, seq_ref: str):
    """(pred_idx, ref_idx) pairs from a global sequence alignment.

    Robust to missing residues / terminal & internal gaps, unlike positional
    "first-n" matching (which shifts every pair after the first gap).
    """
    from Bio.Align import PairwiseAligner

    a = PairwiseAligner()
    a.mode = "global"
    a.match_score, a.mismatch_score = 2.0, -1.0
    a.open_gap_score, a.extend_gap_score = -5.0, -0.5
    aln = a.align(seq_pred, seq_ref)[0]
    pairs: list[tuple[int, int]] = []
    for (p0, p1), (r0, r1) in zip(aln.aligned[0], aln.aligned[1]):
        pairs += [(p0 + k, r0 + k) for k in range(p1 - p0)]
    return pairs


def _ligand_heavy_coords(sdf_path: str | Path) -> np.ndarray:
    """Heavy-atom coordinates of a ligand SDF (used to define pocket residues)."""
    from rdkit import Chem

    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=True)[0]
    if mol is None:
        raise ValueError(f"Failed to read ligand SDF: {sdf_path}")
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)


def _pocket_mask(ref_ca: np.ndarray, lig: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask over ref_ca: True where a Ca is within `radius` A of any
    ligand heavy atom (Ca-based pocket definition)."""
    d = np.sqrt(((ref_ca[:, None, :] - lig[None, :, :]) ** 2).sum(-1))   # (N, L)
    return d.min(axis=1) <= radius


# ===========================================================================
# Kabsch superposition (pure numpy; verified)
# ===========================================================================
def _kabsch(ref: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find rotation R and translation t minimising || ref - (mov @ R.T + t) ||."""
    ref_mean, mov_mean = ref.mean(axis=0), mov.mean(axis=0)
    H = (mov - mov_mean).T @ (ref - ref_mean)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:                 # reflection correction
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = ref_mean - mov_mean @ R.T
    return R, t


# ===========================================================================
# Public entry point
# ===========================================================================
def align_ligand_to_reference(
    pred_receptor_pdb: str | Path,
    pred_ligand_sdf: str | Path,
    ref_receptor_pdb: str | Path,
    out_ligand_sdf: str | Path,
    min_identity: float = 0.9,
    pocket_radius: float | None = None,
    ref_ligand_sdf: str | Path | None = None,
) -> Path:
    """Superpose predicted receptor onto reference receptor (Ca, by sequence
    alignment) and apply the transform to the predicted ligand.

    If ``pocket_radius`` is given, superpose using ONLY the reference Ca atoms
    within ``pocket_radius`` A of the reference ligand (``ref_ligand_sdf``
    required). Pocket-local superposition gives a meaningful ligand pose RMSD:
    it isolates "given the pocket is aligned, how far off is the ligand", and is
    not dragged by distal-loop / domain-orientation misprediction. Recommended
    for pose evaluation (e.g. pocket_radius=10.0). pocket_radius=None uses the
    whole receptor (backward-compatible).

    Returns the path to the aligned ligand SDF (predicted ligand now in the
    reference-receptor coordinate frame, ready for symmetry-corrected RMSD).
    """
    from rdkit import Chem

    if pocket_radius is not None and ref_ligand_sdf is None:
        raise ValueError("pocket_radius set but ref_ligand_sdf not provided "
                         "(needed to define pocket residues)")

    pred_chains = _extract_ca(pred_receptor_pdb)
    ref_chains = _extract_ca(ref_receptor_pdb)

    # pair chains: shared ids; or single-vs-single with different ids; else error
    shared = sorted(set(pred_chains) & set(ref_chains))
    if shared:
        chain_pairs = [(c, c) for c in shared]
    elif len(pred_chains) == len(ref_chains) == 1:
        chain_pairs = [(next(iter(pred_chains)), next(iter(ref_chains)))]
    else:
        raise ValueError(
            f"No common chains: pred {sorted(pred_chains)} vs ref {sorted(ref_chains)}")

    pred_pts: list[np.ndarray] = []
    ref_pts: list[np.ndarray] = []
    for pred_cid, ref_cid in chain_pairs:
        pred_names, pred_coords = pred_chains[pred_cid]
        ref_names, ref_coords = ref_chains[ref_cid]
        if len(pred_names) < 5 or len(ref_names) < 5:
            continue
        seq_pred = "".join(_AA3TO1.get(x, "X") for x in pred_names)
        seq_ref = "".join(_AA3TO1.get(x, "X") for x in ref_names)
        pairs = _aligned_index_pairs(seq_pred, seq_ref)
        if len(pairs) < 5:
            continue
        ident = sum(pred_names[ip] == ref_names[ir] for ip, ir in pairs) / len(pairs)
        if ident < min_identity:
            raise ValueError(
                f"Chain {pred_cid}->{ref_cid}: aligned identity only {ident:.0%} "
                f"(<{min_identity:.0%}); sequences likely differ")
        for ip, ir in pairs:
            pred_pts.append(pred_coords[ip])
            ref_pts.append(ref_coords[ir])

    if len(ref_pts) < 5:
        raise ValueError(f"Too few aligned Ca atoms ({len(ref_pts)}); need >=5")

    pred_arr, ref_arr = np.array(pred_pts), np.array(ref_pts)

    scope = "all-chain"
    if pocket_radius is not None:
        lig = _ligand_heavy_coords(ref_ligand_sdf)
        mask = _pocket_mask(ref_arr, lig, pocket_radius)
        if int(mask.sum()) < 8:
            raise ValueError(
                f"Only {int(mask.sum())} aligned Ca within {pocket_radius:g} A of the "
                f"reference ligand; increase pocket_radius or check ref_ligand_sdf")
        pred_arr, ref_arr = pred_arr[mask], ref_arr[mask]
        scope = f"pocket(<{pocket_radius:g}A)"

    rot, tran = _kabsch(ref_arr, pred_arr)
    aligned = pred_arr @ rot.T + tran
    rmsd = float(np.sqrt(np.mean(np.sum((ref_arr - aligned) ** 2, axis=1))))
    logger.info("Receptor alignment [%s]: %d Ca, RMSD=%.2f A", scope, len(ref_arr), rmsd)

    mol = Chem.SDMolSupplier(str(pred_ligand_sdf), removeHs=False)[0]
    if mol is None:
        raise ValueError(f"Failed to read ligand SDF: {pred_ligand_sdf}")
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i), dtype=float)
        conf.SetAtomPosition(i, (pos @ rot.T + tran).tolist())

    Path(out_ligand_sdf).parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(out_ligand_sdf)) as w:
        w.write(mol)
    logger.info("Aligned ligand written to %s (%s receptor Ca RMSD %.2f A)",
                out_ligand_sdf, scope, rmsd)
    return Path(out_ligand_sdf)


# ===========================================================================
# Self-test
# ===========================================================================
def _self_test() -> int:
    rng = np.random.default_rng(0)
    # --- Kabsch: recover a known rigid transform exactly ---
    ref = rng.normal(size=(60, 3)) * 12.0
    A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t0 = rng.normal(size=3) * 4.0
    mov = (ref - t0) @ Q                       # so that mov @ Q.T + t0 == ref
    R, t = _kabsch(ref, mov)
    aligned = mov @ R.T + t
    assert np.allclose(R, Q, atol=1e-6) and np.allclose(t, t0, atol=1e-6), "Kabsch transform wrong"
    assert np.allclose(aligned, ref, atol=1e-6), "Kabsch did not align"
    # applying the SAME transform to an independent 'ligand' must also recover it
    lig_ref = rng.normal(size=(15, 3)) * 6.0
    lig_mov = (lig_ref - t0) @ Q
    assert np.allclose(lig_mov @ R.T + t, lig_ref, atol=1e-6), "ligand transform wrong"
    print("Kabsch OK: exact transform recovery, ligand transform consistent")

    # --- alignment correspondence: robust to one missing (disordered) residue ---
    try:
        from Bio.Align import PairwiseAligner  # noqa: F401
        seq_pred = "ACDEFGHIKLMNPQRST"          # full predicted 1..17
        seq_ref = seq_pred[:4] + seq_pred[5:]   # crystal missing residue at index 4
        pairs = _aligned_index_pairs(seq_pred, seq_ref)
        assert len(pairs) == len(seq_ref), f"expected {len(seq_ref)} pairs, got {len(pairs)}"
        assert all(seq_pred[ip] == seq_ref[ir] for ip, ir in pairs), "misaligned pairs"
        assert (4, 4) not in pairs, "should not pair the deleted residue"
        print(f"alignment OK: 1 missing residue -> {len(pairs)} correct pairs (positional would fail)")
    except ImportError:
        print("Biopython absent: alignment path skipped (install biopython to test it).")

    # --- pocket mask + pocket-restricted Kabsch (pure numpy) ---
    core = rng.normal(size=(12, 3)) * 2.0                  # tight cluster (pocket)
    far = rng.normal(size=(28, 3)) * 30.0 + 120.0          # distal residues
    ref_ca = np.vstack([core, far])
    lig = core.mean(axis=0) + rng.normal(size=(8, 3)) * 0.4
    mask = _pocket_mask(ref_ca, lig, radius=6.0)
    assert mask[:12].sum() >= 10 and not mask[12:].any(), "pocket mask selected wrong residues"
    sub = ref_ca[mask]
    mov_sub = (sub - t0) @ Q
    Rp, tp = _kabsch(sub, mov_sub)
    assert np.allclose(mov_sub @ Rp.T + tp, sub, atol=1e-6), "pocket-restricted Kabsch wrong"
    print(f"pocket OK: {int(mask.sum())} Ca within radius selected, restricted Kabsch exact")
    print("PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Align predicted ligand into the reference-receptor frame.")
    ap.add_argument("--pred-receptor", help="predicted receptor (mmCIF or PDB)")
    ap.add_argument("--pred-ligand", help="predicted ligand SDF (predicted frame)")
    ap.add_argument("--ref-receptor", help="reference (crystal) receptor PDB/mmCIF")
    ap.add_argument("--out", help="output aligned ligand SDF")
    ap.add_argument("--min-identity", type=float, default=0.9)
    ap.add_argument("--pocket-radius", type=float, default=None,
                    help="superpose only on ref Ca within this many A of the ref ligand "
                         "(recommended ~10 for pose RMSD); requires --ref-ligand")
    ap.add_argument("--ref-ligand", help="reference (crystal) ligand SDF; required with --pocket-radius")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not all([args.pred_receptor, args.pred_ligand, args.ref_receptor, args.out]):
        ap.error("need --pred-receptor --pred-ligand --ref-receptor --out (or --self-test)")
    if args.pocket_radius is not None and not args.ref_ligand:
        ap.error("--pocket-radius requires --ref-ligand")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    align_ligand_to_reference(args.pred_receptor, args.pred_ligand,
                              args.ref_receptor, args.out, args.min_identity,
                              args.pocket_radius, args.ref_ligand)
    return 0


if __name__ == "__main__":
    sys.exit(main())
