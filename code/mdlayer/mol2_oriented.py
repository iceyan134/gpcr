"""Apply the OPM orientation transform to a ligand mol2 (writes mol2 + pdb)."""
import sys, json
import numpy as np

src, prov_path, prefix = sys.argv[1:4]
prov = json.load(open(prov_path))
R = np.array(prov["rotation_matrix"])
c = np.array(prov["tm_centroid"])

lines = open(src).read().splitlines()
out_mol2, out_pdb = [], []
in_atoms = False
n = 0
for ln in lines:
    if ln.startswith("@<TRIPOS>ATOM"):
        in_atoms = True
        out_mol2.append(ln)
        continue
    if ln.startswith("@<TRIPOS>") and in_atoms:
        in_atoms = False
        out_mol2.append(ln)
        continue
    if in_atoms and ln.strip():
        p = ln.split()
        name, x, y, z = p[1], float(p[2]), float(p[3]), float(p[4])
        q = R @ (np.array([x, y, z]) - c)
        out_mol2.append("%7d %-8s %10.4f %10.4f %10.4f %-6s %3d %-8s %10.4f" %
                        (int(p[0]), name, q[0], q[1], q[2], p[5], int(p[6]), p[7], float(p[8])))
        # p[5] is the mol2 atom TYPE (c3, ca, os, cl, hc ...), not an element;
        # the PDB element column (77-78) must be a real element symbol.
        raw = p[5].split(".")[0].lower()
        if raw.startswith("cl"):
            el = "Cl"
        elif raw.startswith("br"):
            el = "Br"
        else:
            el = {"c": "C", "n": "N", "o": "O", "h": "H", "s": "S", "f": "F",
                  "p": "P", "i": "I"}.get(raw[0], "C")
        if not el.upper().startswith("H"):
            n += 1
            out_pdb.append("HETATM%5d %-4s MOL A 381    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                           % (n, name[:4], q[0], q[1], q[2], el[:2].upper()))
        continue
    out_mol2.append(ln)

open(prefix + ".mol2", "w").write("\n".join(out_mol2) + "\n")
with open(prefix + ".pdb", "w") as f:
    f.writelines(out_pdb)
    f.write("END\n")
print("wrote %s.mol2 (%d atoms) and %s.pdb (%d heavy, names from mol2)"
      % (prefix, sum(1 for l in out_mol2 if l[:1].strip() and l.split()[0].isdigit()), prefix, n))
