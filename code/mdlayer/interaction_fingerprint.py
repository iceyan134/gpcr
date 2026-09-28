"""CLI wrapper for md_fingerprint (library must stay argv-free at import)."""
import sys
import json
import numpy as np
from openmm import unit
import openmm.app as app

from md_fingerprint import (setup, fingerprint, summarize, retention,
                            DECISION_TYPES, selftest)


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--selftest":
        raise SystemExit(0 if selftest() else 1)
    PRM, INP = args[0], args[1]
    DCD = args[2] if len(args) > 2 and not args[2].startswith("--") else None
    EVERY, DT, OUT = 1, None, None
    for i, a in enumerate(args):
        if a == "--every":
            EVERY = int(args[i + 1])
        if a == "--dt":
            DT = float(args[i + 1])
        if a == "--out":
            OUT = args[i + 1]

    info = setup(PRM, INP)
    print("setup: %s" % info)
    inpcrd = app.AmberInpcrdFile(INP)
    xyz0 = np.array(inpcrd.positions.value_in_unit(unit.angstrom))
    box0 = np.array(inpcrd.boxVectors.value_in_unit(unit.angstrom))
    ref = fingerprint(xyz0, box0)
    print("reference (t=0): %d interactions  %s" % (len(ref), dict(summarize(ref))))
    for t in DECISION_TYPES + ("halogen", "pipi", "pication"):
        sel = sorted("%s@%s" % (a, b) for tt, a, b in ref if tt == t)
        if sel:
            print("   %-9s %s" % (t, sel[:8]))
    nd, nref = retention(ref, ref)
    print("   decision types (ionic+hbond) = %d  <-- the readout that matters" % nref)

    rows = []
    if DCD:
        import mdtraj as md
        if DT is None:
            sys.exit("ERROR: --dt required with a DCD. The driver's DCD stride is "
                     "n_prod//200 = 50 ps; 0.25 ns is the audit cadence, a different thing.")
        traj = md.load(DCD, top=PRM)
        n = traj.n_frames
        print("\n%d frames, dt=%.3f ns" % (n, DT))
        print(" t(ns)  n_int  overall  ionic  hbond")
        for f in range(0, n, EVERY):
            cur = fingerprint(traj.xyz[f] * 10.0, traj.unitcell_vectors[f] * 10.0)
            per = {}
            for typ in DECISION_TYPES:
                rt = {x for x in ref if x[0] == typ}
                per[typ] = round(len(rt & cur) / max(1, len(rt)), 3)
            ov = len(ref & cur) / max(1, len(ref))
            rows.append({"t_ns": round(f * DT, 4), "n_int": len(cur),
                         "overall": round(ov, 3), "per_type": per})
            if f % max(1, EVERY * 4) == 0 or f == n - 1:
                print("%6.2f  %5d   %6.3f  %5.3f  %5.3f"
                      % (f * DT, len(cur), ov, per["ionic"], per["hbond"]))
        val, nref = retention(ref, fingerprint(traj.xyz[-1] * 10.0, traj.unitcell_vectors[-1] * 10.0))
        print("\nfinal decision-type retention (ionic+hbond) = %.3f over %d reference interactions"
              % (val, nref))
    if OUT:
        json.dump({"setup": info, "reference": sorted("%s|%s|%s" % t for t in ref),
                   "reference_counts": dict(summarize(ref)), "trajectory": rows},
                  open(OUT, "w"), indent=1)
        print("written:", OUT)


if __name__ == "__main__":
    main()
