"""Insert TER records at chain breaks, mapped from the oriented template (fusion junctions stay bonded)."""
import sys

STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
          "PHE PRO SER THR TRP TYR VAL CYX ASH GLH LYN ACE NME NHE".split())

tpl, target, out = sys.argv[1:4]


def protein_residues(path):
    """(chain, resid, resname) in file order, protein only, H excluded from counting."""
    seen, order = set(), []
    for ln in open(path, errors="ignore"):
        if ln[:6] not in ("ATOM  ", "HETATM"):
            continue
        rn = ln[17:20].strip()
        if rn not in STD:
            continue
        key = (ln[21], ln[22:26].strip(), rn)
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


T = protein_residues(tpl)
breaks = set()
for i in range(len(T) - 1):
    if T[i][0] != T[i + 1][0]:
        breaks.add(i)
        continue
    try:
        if int(T[i + 1][1]) != int(T[i][1]) + 1:
            breaks.add(i)
    except ValueError:
        breaks.add(i)
print("template %s: %d protein residues, %d chain breaks at indices %s"
      % (tpl.split("/")[-1], len(T), len(breaks), sorted(breaks)[:20]))

n_prot = 0
prev = None
n_ter = 0
with open(out, "w") as f:
    for ln in open(target, errors="ignore"):
        if ln[:6] in ("ATOM  ", "HETATM"):
            rn = ln[17:20].strip()
            key = (ln[21], ln[22:26].strip(), rn)
            brk = False
            if key != prev:
                if prev is not None and rn in STD:
                    # only protein-internal breaks from the template; the packed file
                    # already has TER between molecules and the OL/PA/PC fragments of
                    # one lipid MUST stay connected for tleap to rebuild POPC
                    if n_prot in breaks or prev[0] != key[0]:
                        brk = True
                if rn in STD:
                    n_prot += 1
                if brk:
                    f.write("TER\n")
                    n_ter += 1
            f.write(ln)
            prev = key
        elif ln.startswith("TER"):
            f.write(ln)          # keep existing breaks; deleting them made tleap
                                 # try to bond lipid fragments across them
        elif ln.startswith("END"):
            continue
        else:
            f.write(ln)
    f.write("END\n")

if n_prot != len(T):
    sys.exit("insert_ter_mapped: FAIL -- target has %d protein residues, template has "
             "%d; residue order does not correspond, refusing to guess" % (n_prot, len(T)))
print("insert_ter_mapped: %d protein residues matched, %d TER -> %s"
      % (n_prot, n_ter, out))
