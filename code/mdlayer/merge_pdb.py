"""Merge non-lipid and converted-lipid PDBs preserving order."""
import sys

out_path, parts = sys.argv[1], sys.argv[2:]
lines = []
for p in parts:
    for ln in open(p):
        if ln.startswith("END"):
            continue                      # never let an intermediate END through
        if ln.startswith("CRYST1"):
            continue                      # box comes from tleap
        if ln.startswith(("ATOM", "HETATM", "TER", "ANISOU", "CONECT")):
            lines.append(ln)
with open(out_path, "w") as f:
    f.writelines(lines)
    f.write("TER\nEND\n")
n_atom = sum(1 for l in lines if l.startswith(("ATOM", "HETATM")))
print("merged %s: %d parts, %d atom records" % (out_path, len(parts), n_atom))
