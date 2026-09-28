"""Remove all residues with a given name from a PDB."""
import sys

src, dst = sys.argv[1], sys.argv[2]
drop = set(sys.argv[3:])
n, kept = 0, []
for ln in open(src):
    if ln.startswith(("ATOM", "HETATM")) and ln[17:20].strip() in drop:
        n += 1
        continue
    kept.append(ln)
with open(dst, "w") as f:
    f.writelines(kept)
    if not any(l.startswith("END") for l in kept[-3:]):
        f.write("END\n")
print("removed %d atoms of %s -> %s" % (n, ",".join(sorted(drop)), dst))
