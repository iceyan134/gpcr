"""G1: GPCR state annotator based on microswitch geometry.

State discrimination for GPCR structures via conservative-sequence anchor
residues, without requiring external Ballesteros-Weinstein numbering:

  Anchor 1: TM3 DRY motif   - D/E(3.49)-R(3.50)-Y(3.51); R(3.50) is >90%
             conserved in Class A and forms a salt bridge with E6.30 in the
             inactive state.
  Anchor 2: TM6 CWxP motif  - C(6.47)-W(6.48)-x-P(6.50); W(6.48) is ~96%
             conserved. 6.37 (activation microswitch) sits 11 residues
             N-terminal of W(6.48) in the same helix.

Metrics (literature-established, e.g. DRY/Rose 2010, Van Eps 2018, GPCRdb):
  d_TM6_3 = |Cα(6.37) - Cα(3.50)| : TM6 outward movement upon activation.
            inactive ~5-9 Å, active ~11-16 Å (Class A canonical).
  d_DRY   = |Cα(3.50) - Cα(6.30)| : ionic lock distance; inactive < 5 Å
            (salt bridge), active > 7 Å (broken).

State calls are heuristic (threshold-based) with an explicit 'uncertain'
tier; they are calibration data for the pipeline, not definitive labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


# ── Structure parsing (minimal PDB reader, no external deps) ──────────────

@dataclass
class Residue:
    chain: str
    resnum: int
    resname: str
    coords_ca: tuple[float, float, float] | None = None


def parse_pdb(path: str | Path) -> list[Residue]:
    """Parse a PDB file, extracting CA coordinates per residue."""
    residues: list[Residue] = []
    cur: Residue | None = None
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("ATOM"):
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            chain = line[21]
            resnum_raw = line[22:26].strip()
            try:
                resnum = int(resnum_raw)
            except ValueError:
                continue
            resname = line[17:20].strip()
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            residues.append(Residue(chain=chain, resnum=resnum,
                                    resname=resname, coords_ca=(x, y, z)))
    return residues


# ── Sequence helpers ──────────────────────────────────────────────────────

def chain_sequence(residues: list[Residue]) -> str:
    """Reconstruct one-letter sequence from parsed residues (sorted by resnum)."""
    by_chain: dict[str, list[Residue]] = {}
    for r in residues:
        by_chain.setdefault(r.chain, []).append(r)
    out = {}
    for ch, rs in by_chain.items():
        rs_sorted = sorted(rs, key=lambda r: r.resnum)
        out[ch] = "".join(_aa1(r.resname) for r in rs_sorted)
    return out.get("A", out.get(list(out.keys())[0], "") if out else "")


_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _aa1(resname: str) -> str:
    return _AA1.get(resname.upper(), "X")


def find_motif(seq: str, pattern: str) -> int:
    """Return 0-based index of first match of pattern (e.g. 'DRY'), else -1."""
    return seq.find(pattern)


def find_cwxp(seq: str, *, r350_seq_idx: int = -1) -> int:
    """Find TM6 CWxP motif with Class-A variant tolerance.

    Canonical: C(6.47)-W(6.48)-x-P(6.50). Variants observed across GPCRs:
      CWLP (CXCR4), CWAP (S1PR1), CWAG (CRFR1, no P), WxP (CCR5, no C).
    Strategy: (1) strict CWxP; (2) C-W-x-x-P (P at W+3); (3) W-x-P without
    the preceding C, ONLY if the W lies downstream of R(3.50) by at least
    50 residues (TM3->TM6 spacing), which rejects N-terminal false anchors.
    Returns index of the motif start (C if present else W), or -1.
    """
    n = len(seq)
    # 1) strict CWxP
    for i in range(n - 3):
        if seq[i] == "C" and seq[i + 1] == "W" and seq[i + 3] == "P":
            return i
    # 2) CWxxP
    for i in range(n - 4):
        if seq[i] == "C" and seq[i + 1] == "W" and seq[i + 4] == "P":
            return i
    # 3) WxP / WxxP without preceding C, with downstream-of-R3.50 constraint
    if r350_seq_idx >= 0:
        min_w = r350_seq_idx + 50
        for i in range(max(min_w, 0), n - 2):
            if seq[i] == "W" and (seq[i + 2] == "P" or
                                  (i + 3 < n and seq[i + 3] == "P")):
                return i
    return -1


# ── Geometry ──────────────────────────────────────────────────────────────

def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def residue_map(residues: list[Residue]) -> dict[int, Residue]:
    """Map resnum -> Residue for chain A (first chain)."""
    chain_a = [r for r in residues if r.chain == "A"] or residues
    return {r.resnum: r for r in chain_a if r.coords_ca}


# ── State annotation ──────────────────────────────────────────────────────

@dataclass
class StateAnnotation:
    state: str                      # "active" | "inactive" | "uncertain"
    d_tm6_3: float | None = None    # distance 6.37(Ca) - 3.50(Ca) in Å
    d_ionic_lock: float | None = None  # distance 3.50 - 6.30(Ca) in Å
    score: float = 0.0              # signed confidence: + active, - inactive
    anchors: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# Thresholds (Class A canonical; from literature ranges, calibrated in G1 gate)
TH_TM6_3_ACTIVE = 11.0    # > this → active-like
TH_TM6_3_INACTIVE = 9.0   # < this → inactive-like; between → uncertain
TH_IONIC_LOCK = 6.0       # < this → salt bridge intact (inactive-like)


def annotate_state(path: str | Path, *, chain: str = "A") -> StateAnnotation:
    """Annotate GPCR structure state from PDB geometry."""
    residues = parse_pdb(path)
    rmap = residue_map(residues)
    seq = chain_sequence(residues)
    if not rmap or not seq:
        return StateAnnotation(state="uncertain", notes=["parse failed"])

    # Sorted residue numbers in sequence order (seq[i] <-> resnums[i])
    resnums = sorted(rmap.keys())
    if len(resnums) != len(seq):
        # guard: sequence length must match residue count (may mismatch on
        # non-standard residues encoded as 'X'); fall back to shorter
        n = min(len(resnums), len(seq))
        resnums, seq = resnums[:n], seq[:n]

    # Locate anchor residues by SEQUENCE index, then map to resnum.
    notes: list[str] = []
    anchors: dict = {}

    # TM3 DRY motif → R(3.50) is the 2nd residue of DRY
    dry_idx = seq.find("DRY")
    r350_num = None
    r350 = None
    if dry_idx >= 0:
        r350_num = resnums[dry_idx + 1]
        r350 = rmap.get(r350_num)
        anchors["3.50_R"] = r350_num
    else:
        notes.append("DRY motif not found")

    # TM6 CWxP motif → W(6.48); W position depends on match variant
    cwxp_idx = find_cwxp(seq, r350_seq_idx=dry_idx if dry_idx >= 0 else -1)
    w648_num = None
    r637 = None
    r630 = None
    if cwxp_idx >= 0:
        # W is at idx+1 if the match started with C, else at idx (WxP fallback)
        w_off = 1 if cwxp_idx + 1 < len(seq) and seq[cwxp_idx] == "C" else 0
        if cwxp_idx + w_off < len(resnums):
            w648_num = resnums[cwxp_idx + w_off]
            # 6.37 is 11 residues N-terminal of 6.48; 6.30 is 18 N-terminal
            r637 = rmap.get(w648_num - 11)
            r630 = rmap.get(w648_num - 18)
            anchors["6.48_W"] = w648_num
            anchors["6.37"] = w648_num - 11
            anchors["6.30"] = w648_num - 18
    else:
        notes.append("CWxP motif not found")

    # Compute distances
    d_tm6_3 = dist(r637.coords_ca, r350.coords_ca) if r637 and r350 and r637.coords_ca and r350.coords_ca else None
    d_lock = dist(r350.coords_ca, r630.coords_ca) if r350 and r630 and r350.coords_ca and r630.coords_ca else None

    # Score: d_TM6_3 is the PRIMARY discriminator (TM6 outward movement is
    # the canonical activation metric across Class A). The ionic-lock distance
    # (3.50-6.30) is receptor-dependent (D2R etc. lack a canonical DRY-E6.30
    # salt bridge), so it is only used to arbitrate the intermediate zone.
    score = 0.0
    if d_tm6_3 is not None:
        if d_tm6_3 >= TH_TM6_3_ACTIVE:
            score += 2.0
        elif d_tm6_3 <= TH_TM6_3_INACTIVE:
            score -= 2.0
        else:
            # intermediate: let ionic lock arbitrate if available
            if d_lock is not None:
                if d_lock >= TH_IONIC_LOCK:
                    score += 0.5
                else:
                    score -= 0.5

    # Decide
    if score >= 2.0:
        state = "active"
    elif score <= -2.0:
        state = "inactive"
    else:
        state = "uncertain"

    return StateAnnotation(
        state=state, d_tm6_3=d_tm6_3, d_ionic_lock=d_lock,
        score=score, anchors=anchors, notes=notes)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        a = annotate_state(p)
        print(f"{Path(p).name}: state={a.state} d_TM6-3={a.d_tm6_3:.1f}Å "
              f"lock={a.d_ionic_lock:.1f}Å score={a.score} anchors={a.anchors}")
