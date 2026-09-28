"""Place negative-control decoy poses at wrong-site pockets."""
import sys, os, json
import numpy as np

STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
          "PHE PRO SER THR TRP TYR VAL CYX ASH GLH LYN ACE NME NHE".split())
MIN_SEP, MAX_TOUCH, JAC_MAX = 2.2, 4.5, 0.30
MODE = os.environ.get("DECOY_MODE", "wrong_site")

rec_pdb, ref_pdb, mol2_in, mol2_out = sys.argv[1:5]
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0


def read_pdb(path):
    prot, lig = [], []
    for ln in open(path, errors="ignore"):
        if ln[:6] not in ("ATOM  ", "HETATM"):
            continue
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ln[17:20].strip() == "MOL":
            if el != "H":
                lig.append((ln[12:16].strip(), xyz))
        elif ln[17:20].strip() in STD and el != "H":
            prot.append((ln[17:20].strip() + ln[22:26].strip(), xyz))
    return prot, lig


def read_mol2(path):
    lines = open(path).read().splitlines(True)
    idx, in_atoms = [], False
    for i, ln in enumerate(lines):
        if ln.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if in_atoms:
            if ln.startswith("@<TRIPOS>"):
                break
            if ln.strip():
                idx.append(i)
    el = np.array([[float(lines[i].split()[2]), float(lines[i].split()[3]),
                    float(lines[i].split()[4])] for i in idx])
    return lines, idx, el


prot, ref_lig = read_pdb(ref_pdb)
P = np.array([a[1] for a in prot])
prot_res = np.array([a[0] for a in prot])
Lref = np.array([a[1] for a in ref_lig])
anchor = Lref.mean(0)
cryst_contacts = {prot_res[i] for i in np.where(
    np.sqrt(((Lref[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(0) < 4.5)[0]}

target_anchor = anchor
if MODE == "wrong_site":
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    lo, hi, step = P.min(0) - 2.0, P.max(0) + 2.0, 2.0
    cand = []
    for x in np.arange(lo[0], hi[0] + step, step):
        for y in np.arange(lo[1], hi[1] + step, step):
            for z in np.arange(lo[2], hi[2] + step, step):
                pt = np.array([x, y, z])
                dmin_pt, _ = tree.query(pt)
                if dmin_pt < 3.0 or np.linalg.norm(pt - anchor) < 10.0:
                    continue
                n_near = len(tree.query_ball_point(pt, 8.0))
                if n_near >= 15:
                    cand.append((n_near, float(np.linalg.norm(pt - anchor)), pt))
    cand.sort(key=lambda t: (-t[0], t[1]))
    if not cand:
        print(json.dumps({"pass": False, "reason": "no alternative cavity found"}))
        sys.exit(1)
    # deep internal cavities (100+ heavy atoms around) admit no random orientation
    # (400/400 clashes); scan several cavities, most-enclosed first but skipping the
    # hopeless ones, and try a limited number of rotations at each
    cavities = [c for c in cand[:40]]
    print("wrong-site: %d candidate cavities (enclosure %d..%d heavy atoms within 8 A)"
          % (len(cavities), cavities[-1][0], cavities[0][0]))
else:
    print("pocket mode needs a docked pose (random orientations clash 400/400 in a "
          "buried pocket); refusing", flush=True)

lines, idx, D = read_mol2(mol2_in)
Dc = D - D.mean(0)
rng = np.random.default_rng(seed)
log, ok = [], None
attempt = 0
anchors = [c[2] for c in cavities] if MODE == "wrong_site" else [target_anchor]
for cav_i, anc in enumerate(anchors):
    for _ in range(60):
        A = rng.normal(size=(3, 3))
        Q, Rm = np.linalg.qr(A)
        Q = Q @ np.diag(np.sign(np.diag(Rm)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        X = Dc @ Q.T + anc
        d = np.sqrt(((X[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        dmin = float(d.min())
        contacts = {prot_res[i] for i in np.where(d.min(0) < 4.5)[0]}
        jac = len(contacts & cryst_contacts) / max(1, len(contacts | cryst_contacts))
        rec = {"attempt": attempt, "cavity": cav_i, "dmin": round(dmin, 2),
               "n_contacts": len(contacts), "jaccard_vs_crystal": round(jac, 3)}
        log.append(rec)
        attempt += 1
        if dmin >= MIN_SEP and len(contacts) > 0 and jac <= JAC_MAX:
            ok = (X, rec, cav_i)
            break
    if ok:
        break

if ok is None:
    print(json.dumps({"pass": False, "mode": MODE, "attempts": len(log),
                      "last": log[-1] if log else None}, indent=1))
    sys.exit(1)

X, rec, _cav = ok
target_anchor = anchors[_cav]
for k, i in enumerate(idx):
    p = lines[i].split()
    p[2], p[3], p[4] = "%.4f" % X[k][0], "%.4f" % X[k][1], "%.4f" % X[k][2]
    lines[i] = " ".join(p) + "\n"
open(mol2_out, "w").writelines(lines)
rep = {"pass": True, "mode": MODE, "placed": mol2_out, "accepted": rec,
       "n_attempts": len(log), "seed": seed,
       "anchor": [round(float(v), 1) for v in target_anchor],
       "crystal_contacts": sorted(cryst_contacts),
       "decoy_contacts": sorted({prot_res[i] for i in np.where(
           np.sqrt(((X[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(0) < 4.5)[0]})}
print(json.dumps(rep, indent=1))
print("DECOY_PLACED %s (%s, attempt %d, dmin %.2f, jaccard %.2f)"
      % (mol2_out, MODE, rec["attempt"], rec["dmin"], rec["jaccard_vs_crystal"]))
