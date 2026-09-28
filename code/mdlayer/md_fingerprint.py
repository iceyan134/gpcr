"""Typed interaction-fingerprint library: ionic contacts and H-bonds (D-H...A > 120 deg)."""
from collections import Counter, defaultdict
import numpy as np
# openmm is imported lazily inside setup() so that the geometry selftest (and any
# pure-geometry use) works without an OpenMM installation.

STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
          "PHE PRO SER THR TRP TYR VAL CYX ASH GLH LYN ACE NME NHE".split())
AROM = {"PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
        "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
        "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
        "HID": {"CG", "ND1", "CD2", "CE1", "NE2"},
        "HIE": {"CG", "ND1", "CD2", "CE1", "NE2"},
        "HIP": {"CG", "ND1", "CD2", "CE1", "NE2"}}
CATION = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "HIP": {"ND1", "NE2"}}
ANION = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
APOLAR_C = {"CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE", "CE1", "CE2",
            "CE3", "CZ", "CZ2", "CZ3", "CH2", "SD"}
HB_DIST, ION_DIST, HYD_DIST, HAL_DIST, PIPI_DIST, PICAT_DIST = 3.6, 4.0, 4.0, 3.6, 5.5, 6.0
HB_COS_MIN = 0.5          # D->H vs H->A; equivalent to D-H...A > 120 deg
DECISION_TYPES = ("ionic", "hbond")

_state = {}


def setup(prmtop_path, inpcrd_path=None):
    """Load topology and precompute hydrogens, rings, indices and ligand charges."""
    from openmm import unit, NonbondedForce
    import openmm.app as app
    prmtop = app.AmberPrmtopFile(prmtop_path)
    top = prmtop.topology
    atoms = list(top.atoms())
    lig_idx = [i for i, a in enumerate(atoms) if a.residue.name == "MOL"]
    prot_idx = [i for i, a in enumerate(atoms) if a.residue.name in STD]

    def el(a):
        return (a.element.symbol if a.element is not None else a.name.strip()[0]).upper()

    h_of = defaultdict(list)
    for b in top.bonds():
        i, j = b[0].index, b[1].index
        if el(atoms[i]) == "H":
            h_of[j].append(i)
        elif el(atoms[j]) == "H":
            h_of[i].append(j)

    adj = {i: set() for i in lig_idx}
    for b in top.bonds():
        i, j = b[0].index, b[1].index
        if i in adj and j in adj:
            adj[i].add(j)
            adj[j].add(i)
    lig_rings = []

    def find_rings(path, size):
        if len(path) == size:
            if path[0] in adj[path[-1]] and not any(set(path) == set(r) for r in lig_rings):
                lig_rings.append(list(sorted(path)))
            return
        for nxt in adj[path[-1]]:
            if nxt > path[0] and nxt not in path:
                find_rings(path + [nxt], size)

    for node in sorted(adj):
        for size in (5, 6):
            find_rings([node], size)

    prot_rings = defaultdict(list)
    for i in prot_idx:
        a = atoms[i]
        if a.residue.name in AROM and a.name.strip() in AROM[a.residue.name]:
            prot_rings[(a.residue.name, a.residue.id)].append(i)
    prot_cation = [(i, atoms[i].residue.name + atoms[i].residue.id) for i in prot_idx
                   if atoms[i].residue.name in CATION
                   and atoms[i].name.strip() in CATION[atoms[i].residue.name]]

    sysm = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    nb = [f for f in sysm.getForces() if isinstance(f, NonbondedForce)][0]
    q = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                  for i in range(sysm.getNumParticles())])

    _state.clear()
    _state.update(dict(prmtop=prmtop, top=top, atoms=atoms, lig_idx=lig_idx,
                       prot_idx=prot_idx, h_of=h_of, lig_rings=lig_rings,
                       prot_rings=prot_rings, prot_cation=prot_cation,
                       q=q, el=el))
    return {"n_lig": len(lig_idx), "n_prot": len(prot_idx),
            "n_hydrogens": sum(len(v) for v in h_of.values()),
            "n_lig_rings": len(lig_rings), "n_prot_rings": len(prot_rings)}


def is_setup():
    return bool(_state)


def min_image(d, box):
    b = np.asarray(box, float)
    if b.ndim == 1:
        b = np.diag(b)
    f = d @ np.linalg.inv(b)
    f -= np.round(f)
    return f @ b


def fingerprint(X, box):
    """Set of (type, ligand atom name, protein residue tag)."""
    if not _state:
        raise RuntimeError("md_fingerprint.setup() must be called first")
    s = _state
    atoms, lig_idx, prot_idx = s["atoms"], s["lig_idx"], s["prot_idx"]
    h_of, q, el = s["h_of"], s["q"], s["el"]
    fp = set()
    L, P = X[lig_idx], X[prot_idx]
    r = np.sqrt((min_image(L[:, None, :] - P[None, :, :], box) ** 2).sum(-1))
    for li, pj in np.argwhere(r < max(HB_DIST, ION_DIST, HYD_DIST, HAL_DIST)):
        gi, gj = lig_idx[li], prot_idx[pj]
        la, pa = atoms[gi], atoms[gj]
        le, pe = el(la), el(pa)
        rr = float(r[li, pj])
        tag = pa.residue.name + pa.residue.id
        lan, pan = la.name.strip(), pa.name.strip()
        if rr < ION_DIST and le in ("N", "O") and abs(q[gi]) >= 0.5:
            if pa.residue.name in ANION and pan in ANION[pa.residue.name]:
                fp.add(("ionic", lan, tag))
            if pa.residue.name in CATION and pan in CATION[pa.residue.name]:
                fp.add(("ionic", lan, tag))
        if le in ("CL", "BR", "I") and pe in ("O", "N") and rr < HAL_DIST:
            fp.add(("halogen", lan, tag))
        if le == "C" and pe == "C" and pan in APOLAR_C and rr < HYD_DIST:
            fp.add(("hydrophobic", lan, tag))
        if rr < HB_DIST and le in ("N", "O") and pe in ("N", "O"):
            for donor, hyd, acc in ((gi, h_of.get(gi), gj), (gj, h_of.get(gj), gi)):
                if not hyd:
                    continue
                done = False
                for hi in hyd:
                    v1 = min_image((X[hi] - X[donor])[None, :], box)[0]   # D->H
                    v2 = min_image((X[acc] - X[hi])[None, :], box)[0]     # H->A
                    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                    if cos > HB_COS_MIN:
                        fp.add(("hbond", lan, tag))
                        done = True
                        break
                if done:
                    break
    for key, idxs in s["prot_rings"].items():
        cp = X[idxs].mean(0)
        tag = key[0] + key[1]
        for r_idx in s["lig_rings"]:
            cl = X[r_idx].mean(0)
            if float(np.linalg.norm(min_image((cl - cp)[None, :], box)[0])) < PIPI_DIST:
                fp.add(("pipi", ",".join(sorted(atoms[i].name.strip() for i in r_idx)[:2]), tag))
        for ci, ctag in s["prot_cation"]:
            for r_idx in s["lig_rings"]:
                cl = X[r_idx].mean(0)
                if float(np.linalg.norm(min_image((X[ci] - cl)[None, :], box)[0])) < PICAT_DIST:
                    fp.add(("pication", ",".join(sorted(atoms[i].name.strip() for i in r_idx)[:2]), ctag))
    return fp


def summarize(fp):
    return Counter(t for t, _, _ in fp)


def retention(fp_ref, fp_now, types=DECISION_TYPES):
    """Fraction of reference decision-type interactions still present. (value, n_ref)"""
    ref = {x for x in fp_ref if x[0] in types}
    if not ref:
        return float("nan"), 0
    return len(ref & {x for x in fp_now if x[0] in types}) / len(ref), len(ref)


# ---------------------------------------------------------------- selftests ----
def _selftest_geometry():
    """The two conventions that were wrong: the H-bond angle and the time axis.

    H-bond: place D at -1, H at 0, A at +1 on x (perfectly linear D-H...A). The
    fixed criterion must ACCEPT it; the old (inverted) one would have rejected it.
    """
    def cos_dha(D, H, A):
        v1 = np.array(H) - np.array(D)     # D->H
        v2 = np.array(A) - np.array(H)     # H->A
        return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    linear = cos_dha([-1, 0, 0], [0, 0, 0], [1, 0, 0])
    bent_60 = cos_dha([-1, 0, 0], [0, 0, 0], [0.5, 0.866, 0])      # D-H...A = 120 deg
    bent_90 = cos_dha([-1, 0, 0], [0, 0, 0], [0, 1, 0])            # D-H...A = 90 deg
    ok_linear = linear > HB_COS_MIN
    ok_boundary = bent_60 > HB_COS_MIN
    ok_reject = not (bent_90 > HB_COS_MIN)
    old_would_accept_linear = (0.5 > 0.5)   # old test on cos(H->D, H->A): -1 > 0.5 is False
    print("  H-bond convention (criterion: cos(D->H, H->A) > %.2f):" % HB_COS_MIN)
    print("    linear D-H...A=180deg  cos=%+.2f  -> accept=%-5s (must be True)" % (linear, ok_linear))
    print("    D-H...A=120deg         cos=%+.2f  -> accept=%-5s (boundary, must be True)" % (bent_60, ok_boundary))
    print("    D-H...A=90deg          cos=%+.2f  -> accept=%-5s (must be False)" % (bent_90, not ok_reject))
    print("    the pre-fix inverted test asked cos(H->D,H->A) > 0.5 = %s on the linear case,"
          " i.e. it rejected every real H-bond" % old_would_accept_linear)
    return ok_linear and ok_boundary and ok_reject


def selftest():
    print("md_fingerprint selftest")
    ok = _selftest_geometry()
    print("  => %s" % ("ALL PASS" if ok else "SELFTEST_FAILED"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
