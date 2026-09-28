"""Detect disulfides by SG-SG geometry and rename CYS -> CYX."""
import sys
import numpy as np

inp, out = sys.argv[1], sys.argv[2]
sg = {}
lines = list(open(inp, errors="ignore"))
for k, ln in enumerate(lines):
    if ln[:6] in ("ATOM  ", "HETATM") and ln[17:20].strip() == "CYS" and ln[12:16].strip() == "SG":
        sg[k] = (ln[21], ln[22:26].strip(),
                 np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
pairs = []
ks = sorted(sg)
for a in range(len(ks)):
    for b in range(a + 1, len(ks)):
        d = float(np.linalg.norm(sg[ks[a]][2] - sg[ks[b]][2]))
        if d < 2.3:
            pairs.append((ks[a], ks[b], d, sg[ks[a]], sg[ks[b]]))
# residue keys to rename
renamed = set()
bond_lines = []
for i, j, d, si, sj in pairs:
    renamed.add((si[0], si[1]))
    renamed.add((sj[0], sj[1]))
    bond_lines.append("bond rec.%d.SG rec.%d.SG" % (int(si[1]), int(sj[1])))
    print("disulfide: chain %s %s - chain %s %s  SG-SG %.2f A -> CYX + bond"
          % (si[0], si[1], sj[0], sj[1], d))
with open(out, "w") as f:
    for ln in lines:
        if ln[:6] in ("ATOM  ", "HETATM") and ln[17:20].strip() == "CYS" \
                and (ln[21], ln[22:26].strip()) in renamed:
            f.write(ln[:17] + "CYX" + ln[20:])
        else:
            f.write(ln)
open(out + ".disulfide.leap", "w").write("\n".join(bond_lines) + "\n")
print("renamed %d residues to CYX; %d bond commands -> %s.disulfide.leap"
      % (len(renamed), len(bond_lines), out))
