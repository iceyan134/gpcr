"""Transfer ligand coordinates into the packed frame with template atom naming."""
import sys
import numpy as np

PACKED, MOL2_IN, MOL2_OUT = sys.argv[1:4]
LIG_RES = sys.argv[4] if len(sys.argv) > 4 else "MOL"

STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
          "PHE PRO SER THR TRP TYR VAL CYX ASH GLH LYN ACE NME NHE".split())

packed_lig, packed_prot = {}, []
for ln in open(PACKED, errors="ignore"):
    if ln[:6] not in ("ATOM  ", "HETATM"):
        continue
    nm = ln[12:16].strip()
    el = (ln[76:78].strip() or nm[:1]).upper()
    rid = ln[17:20].strip()
    # both sides of the name match must be heavy-only: bilayer_amber.pdb carries the
    # packed ligand without hydrogens, but com_final.pdb (post-tleap) carries it with
    # them, so filtering one side only makes the names disagree
    if rid == LIG_RES:
        if el != "H":
            packed_lig[nm] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    elif el != "H" and rid in STD:
        packed_prot.append(np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
if not packed_lig:
    sys.exit("LIG_FRAME_FAIL: no %s residue in %s" % (LIG_RES, PACKED))
if not packed_prot:
    sys.exit("LIG_FRAME_FAIL: no protein heavy atoms in %s" % PACKED)

lines = open(MOL2_IN).read().splitlines(True)
atom_idx, in_atoms = [], False
for i, ln in enumerate(lines):
    if ln.startswith("@<TRIPOS>ATOM"):
        in_atoms = True
        continue
    if in_atoms:
        if ln.startswith("@<TRIPOS>"):
            break
        if ln.strip():
            atom_idx.append(i)
hvy_idx = [i for i in atom_idx if not lines[i].split()[5].upper().startswith("H")]
if not hvy_idx:
    sys.exit("LIG_FRAME_FAIL: no heavy atoms in the mol2 template")

mol2_hvy = {lines[i].split()[1]: np.array([float(lines[i].split()[2]),
                                           float(lines[i].split()[3]),
                                           float(lines[i].split()[4])]) for i in hvy_idx}
missing = [n for n in mol2_hvy if n not in packed_lig]
extra = [n for n in packed_lig if n not in mol2_hvy]
if missing or extra:
    sys.exit("LIG_FRAME_FAIL: atom names differ between the packed %s and the mol2 "
             "template. packed-only=%s mol2-only=%s"
             % (LIG_RES, extra[:12], missing[:12]))
names = list(mol2_hvy)
P = np.array([mol2_hvy[n] for n in names])
Q = np.array([packed_lig[n] for n in names])

pc, qc = P.mean(0), Q.mean(0)
U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
d = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
t = qc - R @ pc
rms = float(np.sqrt((((R @ P.T).T + t - Q) ** 2).sum(1).mean()))
if rms > 0.5:
    sys.exit("LIG_FRAME_FAIL: template and packed ligand conformations differ by "
             "%.2f A RMSD on heavy atoms; hydrogens cannot be placed safely" % rms)
print("heavy atoms: %d matched by name, template -> packed RMSD %.3f A" % (len(names), rms))

A = np.array(packed_prot)
dm = np.sqrt(((Q[:, None, :] - A[None, :, :]) ** 2).sum(-1))
dmin = float(dm.min())
n_close = int((dm.min(0) < 4.0).sum())
print("packed-frame pose: dmin=%.2f A  protein heavy atoms within 4 A=%d" % (dmin, n_close))
if not (2.0 <= dmin <= 3.6):
    sys.exit("LIG_FRAME_FAIL: closest ligand-protein contact %.2f A outside "
             "[2.0, 3.6] A -- refusing to build a system with an invalid pose" % dmin)

for i in atom_idx:
    p = lines[i].split()
    xyz = mol2_hvy.get(p[1])
    if xyz is None:
        xyz = np.array([float(p[2]), float(p[3]), float(p[4])])
    q = (R @ xyz) + t
    p[2], p[3], p[4] = "%.4f" % q[0], "%.4f" % q[1], "%.4f" % q[2]
    lines[i] = " ".join(p) + "\n"
open(MOL2_OUT, "w").writelines(lines)
print("LIG_FRAME_OK -> %s (%d atoms, %d heavy)" % (MOL2_OUT, len(atom_idx), len(hvy_idx)))
