"""Numeric verdict classifier for the 2x2 environment/preparation grid.

Priority-ordered exhaustive rules (see docs/VALIDATION_LAYER.md); selftest: --selftest.
"""
import sys, os, json, glob

LOW, HI, DELTA = 0.2, 0.5, 0.10
CONDS = ("A", "B", "C", "D")
VERDICTS = {
    1: "ENVIRONMENT_DOMINANT -> R2a, report stratified by environment",
    2: "PREPARATION_DOMINANT",
    3: "MEMBRANE_PLUS_PREP (synergy: D above both single-factor cells) -> R2a full benchmark",
    4: "CANCEL_MD_GATE -> demote MD to finalists characterization (review sec.15 third priority)",
    5: "NO_PREREGISTERED_PATTERN -> no effect claim; MD gate not enabled",
}


def _m(cells, s, c):
    v = cells.get((s, c))
    return None if v is None else v[0]


def classify(cells, systems):
    """cells: {(system, cond): (mean_F, n_seeds)}. Returns a dict verdict record."""
    rec = {"cells": {"%s|%s" % k: v for k, v in cells.items()}, "systems": sorted(systems)}
    if not systems or not cells:
        # empty input is a FAIL, not a vacuous pass
        rec.update({"verdict_id": 0, "verdict": "NO_DATA",
                    "reason": "no systems/cells supplied (%d cells)" % len(cells)})
        return rec
    missing = [(s, c) for s in systems for c in CONDS if (s, c) not in cells]
    if missing:
        rec.update({"verdict_id": 0, "verdict": "INSUFFICIENT_CELLS",
                    "missing": ["%s|%s" % m for m in missing]})
        return rec

    def mean2(s, cs):
        return sum(_m(cells, s, c) for c in cs) / len(cs)

    ev = {}
    # priority 1: environment dominant (C ~ D, both far above the water arms)
    r1 = all(mean2(s, ("C", "D")) - mean2(s, ("A", "B")) >= DELTA
             and abs(_m(cells, s, "C") - _m(cells, s, "D")) < DELTA for s in systems)
    ev["p1_env_effect"] = {s: round(mean2(s, ("C", "D")) - mean2(s, ("A", "B")), 3) for s in systems}
    ev["p1_CD_gap"] = {s: round(abs(_m(cells, s, "C") - _m(cells, s, "D")), 3) for s in systems}
    if r1:
        rec.update({"verdict_id": 1, "verdict": VERDICTS[1], "evidence": ev})
        return rec
    # priority 2: preparation dominant (B ~ D, jointly above A and C)
    r2 = all(mean2(s, ("B", "D")) - mean2(s, ("A", "C")) >= DELTA
             and abs(_m(cells, s, "B") - _m(cells, s, "D")) < DELTA for s in systems)
    ev["p2_prep_effect"] = {s: round(mean2(s, ("B", "D")) - mean2(s, ("A", "C")), 3) for s in systems}
    ev["p2_BD_gap"] = {s: round(abs(_m(cells, s, "B") - _m(cells, s, "D")), 3) for s in systems}
    if r2:
        rec.update({"verdict_id": 2, "verdict": VERDICTS[2], "evidence": ev})
        return rec
    # priority 3: synergy -- D above BOTH single-factor cells
    r3 = all(_m(cells, s, "D") - max(_m(cells, s, "B"), _m(cells, s, "C")) >= DELTA
             and _m(cells, s, "D") >= HI for s in systems)
    ev["p3_D_minus_best_single"] = {
        s: round(_m(cells, s, "D") - max(_m(cells, s, "B"), _m(cells, s, "C")), 3) for s in systems}
    ev["p3_D_value"] = {s: _m(cells, s, "D") for s in systems}
    if r3:
        rec.update({"verdict_id": 3, "verdict": VERDICTS[3], "evidence": ev})
        return rec
    # priority 4
    if all(_m(cells, s, c) < LOW for s in systems for c in CONDS):
        rec.update({"verdict_id": 4, "verdict": VERDICTS[4], "evidence": ev})
        return rec
    # priority 5 (catch-all)
    rec.update({"verdict_id": 5, "verdict": VERDICTS[5], "evidence": ev})
    return rec


COND_OF_PREP = {"weak": ("A", "C"), "staged": ("B", "D")}   # (water arm, membrane arm)


def load_runs(rundir):
    """Read <SYSTEM>-<PREP>-s<SEED>/result.json files into cell means.

    The cell is decided by the run's OWN recorded 'membrane' flag plus its prep name,
    not by parsing conventions: weak->A/C, staged->B/D (water/membrane). Runs that
    cannot be attributed are counted and reported, never silently dropped -- the
    first version of this loader matched nothing, handed an empty grid to classify(),
    and `all(... for s in [])` returned a vacuous ENVIRONMENT_DOMINANT.
    """
    cells, unparsed = {}, []
    for rj in sorted(glob.glob(os.path.join(rundir, "*", "result.json"))):
        name = os.path.basename(os.path.dirname(rj))
        try:
            parts = name.split("-")          # SYSTEM-PREP-sSEED
            sysname, prep = parts[0], parts[1].lower()
            d = json.load(open(rj))
            f = d.get("F_native")
            if prep not in COND_OF_PREP or not isinstance(f, (int, float)):
                unparsed.append(name)
                continue
            cond = COND_OF_PREP[prep][1 if d.get("membrane") else 0]
            cells.setdefault((sysname, cond), []).append(float(f))
        except Exception:
            unparsed.append(name)
    means = {k: (round(sum(v) / len(v), 3), len(v)) for k, v in cells.items()}
    return means, unparsed


def selftest(verbose=True):
    S = ("S1", "S2")

    def mk(**over):
        base = {(s, c): (0.5, 3) for s in S for c in CONDS}
        for k, v in over.items():
            s, c = k.split("|")
            base[(s, c)] = (v, 3)
        return base

    cases = [
        ("branch1 environment (C~D >> A,B)",
         mk(**{"S1|C": 0.9, "S1|D": 0.85, "S1|A": 0.3, "S1|B": 0.25,
               "S2|C": 0.8, "S2|D": 0.75, "S2|A": 0.2, "S2|B": 0.3}), 1),
        ("branch2 prep (B ~ D >> A,C)",
         mk(**{"S1|B": 0.9, "S1|D": 0.9, "S1|A": 0.3, "S1|C": 0.3,
               "S2|B": 0.8, "S2|D": 0.8, "S2|A": 0.3, "S2|C": 0.3}), 2),
        ("branch3 synergy (D above B and C)",
         mk(**{"S1|A": 0.1, "S1|B": 0.1, "S1|C": 0.4, "S1|D": 0.9,
               "S2|A": 0.1, "S2|B": 0.15, "S2|C": 0.35, "S2|D": 0.8}), 3),
        ("branch4 cancel gate",   mk(**{f"S{i}|{c}": 0.1 for i in (1, 2) for c in CONDS}), 4),
        ("branch5 gap (review's example)",
         mk(**{"S1|A": 0.60, "S1|B": 0.65, "S1|C": 0.40, "S1|D": 0.35,
               "S2|A": 0.60, "S2|B": 0.65, "S2|C": 0.40, "S2|D": 0.35}), 5),
        ("overlap env&mem -> environment wins",
         mk(**{"S1|A": 0.1, "S1|B": 0.1, "S1|C": 0.9, "S1|D": 0.95,
               "S2|A": 0.1, "S2|B": 0.1, "S2|C": 0.85, "S2|D": 0.9}), 1),
        ("low but real difference -> 4 (review: A=.05,D=.19)",
         mk(**{f"S{i}|{c}": 0.05 for i in (1, 2) for c in ("A", "B", "C")}
            | {f"S{i}|D": 0.19 for i in (1, 2)}), 4),
        ("one system disagrees -> 5",
         mk(**{"S1|D": 0.9, "S1|A": 0.1, "S2|D": 0.3, "S2|A": 0.3}), 5),
    ]
    ok = True
    if verbose:
        print("md_verdict selftest (priority-ordered, exhaustive, numeric)")
    for name, cells, want in cases:
        got = classify(cells, S)["verdict_id"]
        good = got == want
        ok &= good
        if verbose:
            print("  %-38s -> %d (expect %d)  %s" % (name, got, want, "PASS" if good else "FAIL"))
    partial = {(("S1", c)): (0.5, 3) for c in ("A", "B")}
    r = classify(partial, S)
    good = r["verdict_id"] == 0 and "missing" in r
    ok &= good
    if verbose:
        print("  %-38s -> %s  %s" % ("partial grid (water column only)",
                                     r["verdict"], "PASS" if good else "FAIL"))
    r = classify({}, [])
    good = r["verdict_id"] == 0 and r["verdict"] == "NO_DATA"
    ok &= good
    if verbose:
        print("  %-38s -> %s  %s" % ("EMPTY grid (the vacuous-pass bug)",
                                     r["verdict"], "PASS" if good else "FAIL"))
    if verbose:
        print("  => %s" % ("ALL PASS" if ok else "SELFTEST_FAILED"))
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        raise SystemExit(0 if selftest() else 1)
    rundir = sys.argv[1]
    means, unparsed = load_runs(rundir)
    if not means:
        print("ERROR: 0 runs parsed from %s (unparsed: %s) -- refusing to classify"
              % (rundir, unparsed[:5]))
        raise SystemExit(2)
    if unparsed:
        print("WARNING: %d unattributed run dir(s) skipped: %s" % (len(unparsed), unparsed[:5]))
    systems = sorted({k[0] for k in means})
    print("cell means loaded: %s" % {("%s|%s" % k): v for k, v in sorted(means.items())})
    rec = classify(means, systems)
    print(json.dumps(rec, indent=1))
    print("VERDICT:", rec["verdict"])
