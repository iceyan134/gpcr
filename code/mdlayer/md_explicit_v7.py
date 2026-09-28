"""Explicit-solvent MD validation driver (10 ns production).

Protocol: staged minimisation under protein+ligand heavy-atom restraints,
NVT heating to 310 K, NPT restraint release (100 -> 30 -> 10 -> 0 kJ/mol/A^2),
then unrestrained production with 0.25 ns audit blocks. Emits status.json,
audit.csv and result.json (F_native + block SE).

Usage: md_explicit_v7.py LABEL SEED PREP PRMTOP INPCRD WD [--membrane] [--ticket T.json]
"""
import sys, os, json, time, hashlib, traceback
import numpy as np

LABEL = sys.argv[1]
SEED = int(sys.argv[2])
PREP = sys.argv[3]
PRM, INP, WD = sys.argv[4:7]
MEMBRANE = "--membrane" in sys.argv[7:]
TICKET = None
if "--ticket" in sys.argv[7:]:
    TICKET = sys.argv[sys.argv.index("--ticket") + 1]
SMOKE = os.environ.get("SMOKE", "") == "1"
PREP_ONLY = "--prep-only" in sys.argv[7:]
WD = os.path.abspath(WD)
# this script is deployed in _audit_scripts (with md_posemetric / md_fingerprint /
# sibling modules (md_metric_v2, mdboxutil, md_fingerprint) live next to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from openmm import unit, CustomExternalForce, LangevinMiddleIntegrator, Platform
from openmm import MonteCarloBarostat, MonteCarloMembraneBarostat
import openmm.app as app
import md_metric_v2 as M

SCRIPT_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
METRIC_SHA = hashlib.sha256(open(os.path.abspath(M.__file__), "rb").read()).hexdigest()[:16]
try:
    import md_posemetric as PM
    POSE_SHA = hashlib.sha256(open(os.path.abspath(PM.__file__), "rb").read()).hexdigest()[:16]
except Exception:
    PM, POSE_SHA = None, "absent"
try:
    import md_fingerprint as IFP_MOD
    IFP_SHA = hashlib.sha256(open(os.path.abspath(IFP_MOD.__file__), "rb").read()).hexdigest()[:16]
except Exception:
    IFP_MOD, IFP_SHA = None, "absent"

STD = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","HID","HIE","HIP",
       "CYX","CYM","ASH","GLH","LYN","ILE","LEU","LYS","MET","PHE","PRO","SER",
       "THR","TRP","TYR","VAL","NMET","CARG","NLYS","NALA","NASP","NGLU","NSER",
       "NTHR","NLEU","NVAL","NILE","NPRO","NPHE","NTYR","NTRP","NGLY","NASN",
       "NGLN","NARG","NHIS","NCYS","NA","CL","MG","ZN"}
WATER = {"HOH", "WAT", "TIP3", "HO4"}
IONS = {"Na+", "Cl-", "NA", "CL", "K", "MG", "ZN", "CA"}


def is_phosphate(name):
    n = name.strip().upper()
    return n == "P" or (n.startswith("P") and n[1:].isdigit())


LIPIDS = {"POP","POPC","POPE","POPS","POPG","DPPC","DOPC","PC","PE","PS","PA","PGR","PH-",
          "OL","OLA","OLC","PLM","MYR","STE","CHL","CHOL","CLR","POPA","DOPG","DPPE",
          "DSPC","SAPI","SAPL","TLCL","CHL1"}

N_PROD_NS = 1.0 if SMOKE else 10.0     # SMOKE=1 -> technical smoke test
AUDIT_EVERY_NS = 0.25
PE_BLOCK_NS = 2.0
PE_DRIFT_MAX = 0.25
DENSITY_RANGE = (0.90, 1.15)
DMIN0_RANGE = (1.8, 5.0)
POSE_NATIVE_MAX = 2.0
RETENTION_MIN = 0.50
PREP_ITERS = {"weak": [(1000.0, 1000)],                      # single short pass
              "staged": [(1000.0, 3000), (500.0, 3000), (100.0, 3000)]}
RELEASE = [(100.0, 75.0), (30.0, 75.0), (10.0, 75.0), (0.0, 75.0)]   # monotone, ps each
HEAT_PS, HEAT_FS, HEAT_STEPS = 21.0, 0.0005, 14
COM_REFRESH_STEPS = 500


def status(state, extra=None):
    d = {"label": LABEL, "seed": SEED, "prep": PREP, "membrane": MEMBRANE,
         "state": state, "t": time.time(), "script_sha": SCRIPT_SHA, "metric_sha": METRIC_SHA,
         "pose_metric_sha": POSE_SHA, "fingerprint_sha": IFP_SHA}
    if extra:
        d.update(extra)
    json.dump(d, open(os.path.join(WD, "status.json"), "w"), indent=1)


print(f"[{LABEL}] v7 prep={PREP} membrane={MEMBRANE} seed={SEED} sha={SCRIPT_SHA} "
      f"pose_metric={POSE_SHA} fingerprint={IFP_SHA}", flush=True)

# ---- provenance / gate ticket (P5) -------------------------------------------
if TICKET and os.path.exists(TICKET):
    tk = json.load(open(TICKET))
    want = tk.get("inpcrd_sha256")
    got = hashlib.sha256(open(INP, "rb").read()).hexdigest()
    if want and want != got:
        status("TECH_FAIL", {"reason": "gate_ticket_hash_mismatch", "want": want, "got": got})
        sys.exit(1)
    print(f"[{LABEL}] gate ticket OK (build_id={tk.get('build_id','?')})", flush=True)
elif TICKET:
    status("TECH_FAIL", {"reason": "gate_ticket_missing", "ticket": TICKET})
    sys.exit(1)

if IFP_MOD is not None:
    _ifp_info = IFP_MOD.setup(PRM, INP)
    print(f"[{LABEL}] fingerprint setup: {_ifp_info}", flush=True)
prmtop = app.AmberPrmtopFile(PRM)
inpcrd = app.AmberInpcrdFile(INP)
system = prmtop.createSystem(nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)

for _f in system.getForces():
    _n = _f.__class__.__name__
    if _n in ("HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "CMAPTorsionForce"):
        _f.setForceGroup(1)
    elif _n == "NonbondedForce":
        _f.setForceGroup(2)
    elif _n.startswith("MonteCarlo"):
        _f.setForceGroup(4)
    else:
        _f.setForceGroup(0)          # CMMotionRemover etc: no energy
atoms = list(prmtop.topology.atoms())
lig_idx, prot_idx, prot_res, lip_p_idx = [], [], [], []
for i, a in enumerate(atoms):
    rn = a.residue.name
    heavy = a.element is not None and a.element.symbol != "H"
    if rn in STD:
        if heavy:
            prot_idx.append(i)
            prot_res.append((a.residue.name, a.residue.id, a.residue.chain.id))
    elif rn in WATER or rn in IONS:
        pass
    elif rn in LIPIDS:
        if is_phosphate(a.name) and heavy:
            lip_p_idx.append(i)
    else:
        if heavy:
            lig_idx.append(i)
print(f"[{LABEL}] lig={len(lig_idx)} prot={len(prot_idx)} lipid_P={len(lip_p_idx)}", flush=True)
if not lig_idx:
    status("TECH_FAIL", {"reason": "no ligand atoms classified"})
    sys.exit(1)

# restrain protein + ligand heavy atoms during prep/heating (addendum5 sec.2);
# for the release the protein is held on the backbone while the ligand stays on
# heavy atoms, then both are released monotonically.
ca_idx = [i for i, a in enumerate(atoms) if a.residue.name in STD and a.name == "CA"]
restrained = sorted(set(prot_idx) | set(lig_idx))
restraint = CustomExternalForce("0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2")
# NOTE: assigning force groups by class name in a later loop is what silently moved the
# restraint into group 0 and hid its energy. Assign explicitly and never re-assign.
restraint.addGlobalParameter("k", 1000.0)
for p in ("x0", "y0", "z0"):
    restraint.addPerParticleParameter(p)
X_IN = np.array(inpcrd.positions.value_in_unit(unit.angstrom))   # this OpenMM returns a list
# Anchors start at the input positions (t=0 restraint energy is exactly 0); the COM
# reference is realised by shifting anchors with the system COM, never pre-offsetting.
COM0 = X_IN[restrained].mean(0)
REF = {i: X_IN[i].copy() for i in restrained}
# ANCHORS MUST BE IN NANOMETRES: addParticle takes bare numbers in the Context's
# default length unit. Passing angstroms put every anchor 10x too far away, which made
# the restraint push the backbone outward instead of holding it (2.5e6 kJ/mol at t=0 for
# 281 CA) and invalidated every preparation run of this project until now.
from mdboxutil import anchors_nm, group_energy_audit
ANCH = anchors_nm(np.array([REF[i] for i in restrained]))
for j, i in enumerate(restrained):
    restraint.addParticle(i, list(ANCH[j]))
restraint.setForceGroup(3)
system.addForce(restraint)

integrator = LangevinMiddleIntegrator(310 * unit.kelvin, 1.0 / unit.picosecond,
                                      0.002 * unit.picoseconds)
integrator.setRandomNumberSeed(SEED)
sim = app.Simulation(prmtop.topology, system, integrator, Platform.getPlatformByName("CUDA"))
sim.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

total_mass = sum(system.getParticleMass(i).value_in_unit(unit.dalton)
                 for i in range(system.getNumParticles()))
BAROSTAT = None


def add_barostat():
    """Enabled only once the system is at 310 K (NVT before that)."""
    global BAROSTAT
    if BAROSTAT is not None:
        return
    if MEMBRANE:
        BAROSTAT = MonteCarloMembraneBarostat(
            1.0 * unit.bar, 0.0 * unit.bar * unit.nanometer,
            310 * unit.kelvin, MonteCarloMembraneBarostat.XYIsotropic,
            MonteCarloMembraneBarostat.ZFree, 25)
    else:
        BAROSTAT = MonteCarloBarostat(1.0 * unit.bar, 310 * unit.kelvin, 25)
    BAROSTAT.setForceGroup(31)
    system.addForce(BAROSTAT)
    sim.context.reinitialize(preserveState=True)


def box_now():
    v = sim.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    return np.array([v[0][0], v[1][1], v[2][2]])


def pos_now():
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)


def refresh_reference():
    """Shift the anchors with the system's COM so the barostat's volume moves do not
    fight a fixed lab-frame anchor. Anchors are input positions + the COM drift."""
    X = pos_now()
    shift = X[restrained].mean(0) - COM0
    for j, i in enumerate(restrained):
        v = (REF[i] + shift) / 10.0        # nanometres
        restraint.setParticleParameters(j, i, [v[0], v[1], v[2]])
    restraint.updateParametersInContext(sim.context)


def membrane_metrics(pos):
    if not lip_p_idx:
        return None
    pz = pos[lip_p_idx][:, 2]
    mid = float(np.median(pz))
    upper, lower = pz[pz > mid], pz[pz <= mid]
    thick = float(upper.mean() - lower.mean()) if len(upper) and len(lower) else None
    return {"midplane_z": mid, "bilayer_thickness": thick,
            "n_upper_P": int(len(upper)), "n_lower_P": int(len(lower))}


def restraint_energy(pos, k):
    """0.5*k*sum(min-image displacement^2) of the restrained atoms -- computed here
    from coordinates rather than read from a force group, so it is independent of
    how OpenMM accounts for the force."""
    d = pos[restrained] - np.array([REF[i] for i in restrained])
    d -= np.round(d / np.array(box_now())) * np.array(box_now())
    return float(0.5 * k * (d ** 2).sum()), float(np.sqrt((d ** 2).sum(1).max()))


def audit():
    box = box_now()
    pos = pos_now()
    m = M.frame_metrics(pos[lig_idx], pos[prot_idx], prot_res, POCKET, TOUCH, box)
    pe = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    vol = float(np.prod(box)) / 1000.0
    dens = total_mass / vol / 602.2
    pose, disp = PM.pose_rmsd(REF_LIG, REF_PROT, pos[lig_idx], pos[prot_idx], box)
    fp_now = IFP_MOD.fingerprint(pos, box) if IFP_MOD is not None else set()
    fp_ret, _nref = IFP_MOD.retention(FP_REF, fp_now) if IFP_MOD is not None else (float("nan"), 0)
    _k = float(sim.context.getParameter("k"))
    restr_e, restr_maxd = restraint_energy(pos, _k)
    row = {"restr_e": restr_e, "restr_maxd": restr_maxd, "k": _k,
           "retention": m["retention"], "pose_rmsd": pose, "pose_disp": disp,
           "fp_retention": fp_ret, "d_pocket": m["d_pocket"], "d_min": m["d_min"],
           "d_min_p50": m["d_min_p50"], "n_contact": m["n_contact_res"],
           "l_rmsd_legacy": M.ligand_rmsd_local(REF_LIG, REF_TOUCH, pos[lig_idx], m["_touch_img"]),
           "pe": pe, "density": dens}
    mm = membrane_metrics(pos)
    if mm:
        row.update({"midplane_z": mm["midplane_z"], "bilayer_thickness": mm["bilayer_thickness"],
                    "ligand_z_vs_midplane": float(pos[lig_idx].mean(0)[2] - mm["midplane_z"])})
    return row


box0 = box_now()
pos0 = pos_now()
POCKET, TOUCH = M.build_pocket(pos0[lig_idx], pos0[prot_idx], prot_res, box0)
REF_LIG = pos0[lig_idx].copy()
REF_PROT = pos0[prot_idx].copy()
REF_TOUCH = M.touch_imaged(REF_LIG.mean(0), pos0[prot_idx], TOUCH, box0)
FP_REF = IFP_MOD.fingerprint(pos0, box0) if IFP_MOD is not None else set()
_m0 = M.frame_metrics(pos0[lig_idx], pos0[prot_idx], prot_res, POCKET, TOUCH, box0)
DMIN0 = float(_m0["d_min"])
NCONT0 = int(_m0["n_contact_res"])
print(f"[{LABEL}] pocket={len(POCKET)} touch={len(TOUCH)} d_min0={DMIN0:.2f} A "
      f"n_contact0={NCONT0} fp_ref={len(FP_REF)}", flush=True)
status("LOADED", {"n_pocket": len(POCKET), "d_min0": round(DMIN0, 2),
                  "n_contact0": NCONT0, "n_fp_ref": len(FP_REF),
                  "n_fp_decision": len([x for x in FP_REF if x[0] in ("ionic", "hbond")]),
                  "prep": PREP})
if not (DMIN0_RANGE[0] <= DMIN0 <= DMIN0_RANGE[1]):
    status("TECH_FAIL", {"reason": "input_contact_out_of_range", "d_min0": round(DMIN0, 2)})
    sys.exit(1)

timings = {}
_a0 = audit()
print("[%s] t=0 audit: pose_rmsd=%.3f retention=%.3f fp=%.3f d_min=%.2f density=%.4f "
      "k=%.0f restr_E=%.4g restr_max_disp=%.3f A"
      % (LABEL, _a0["pose_rmsd"], _a0["retention"], _a0["fp_retention"], _a0["d_min"],
         _a0["density"], _a0["k"], _a0["restr_e"], _a0["restr_maxd"]), flush=True)
_tot0, _per0, _sum0 = group_energy_audit(sim, unit.kilojoule_per_mole)
print("[%s] t=0 energy audit: total=%.6g  sum-of-groups=%.6g  per-group=%s"
      % (LABEL, _tot0, _sum0, {k: round(v, 3) for k, v in _per0.items()}), flush=True)
if abs(_tot0 - _sum0) > 1e-3 * max(1.0, abs(_tot0)):
    status("TECH_FAIL", {"reason": "force_group_audit_mismatch", "total": _tot0, "sum": _sum0})
    print("[%s] AUDIT FAIL: per-group sum does not reproduce the total -- some force is "
          "in an unqueried group" % LABEL, flush=True)
    sys.exit(1)
_re = _per0.get(3, 0.0)
if abs(_re) > 1.0:
    status("TECH_FAIL", {"reason": "restraint_energy_at_t0", "restraint_group_energy": _re})
    print("[%s] AUDIT FAIL: restraint energy from OpenMM is %.6g kJ/mol at t=0 "
          "(anchors off / wrong units)" % (LABEL, _re), flush=True)
    sys.exit(1)

if _a0["restr_e"] > 1.0:
    status("TECH_FAIL", {"reason": "restraint_reference_inconsistent_at_t0",
                         "restr_e": _a0["restr_e"], "restr_max_disp": _a0["restr_maxd"]})
    print("[%s] RESTRAINT GATE FAIL: t=0 restraint energy %.4g kJ/mol, max displacement %.3f A"
          % (LABEL, _a0["restr_e"], _a0["restr_maxd"]), flush=True)
    sys.exit(1)

try:
    # ---- preparation: NVT, protein+ligand heavy atoms restrained -----------------
    t0 = time.time()
    for k, iters in PREP_ITERS[PREP]:
        sim.context.setParameter("k", k)
        sim.minimizeEnergy(maxIterations=iters)
    sim.context.setVelocitiesToTemperature(310 * unit.kelvin, SEED)
    timings["min_s"] = round(time.time() - t0)

    t0 = time.time()
    if PREP == "weak":
        # fast variant: one short minimisation, straight to 310 K at 2 fs
        sim.context.setVelocitiesToTemperature(310 * unit.kelvin, SEED)
        timings["heat_s"] = 0
    else:
        integrator.setStepSize(HEAT_FS * unit.picoseconds)
        sim.context.setVelocitiesToTemperature(50 * unit.kelvin, SEED)
        for T in np.linspace(50, 310, HEAT_STEPS):
            integrator.setTemperature(float(T) * unit.kelvin)
            sim.step(3000)
        integrator.setStepSize(0.002 * unit.picoseconds)
        timings["heat_s"] = round(time.time() - t0)
    a_heat = audit()
    print("[%s] after heating (NVT): pose_rmsd=%.2f retention=%.3f fp=%.3f d_min=%.2f "
          "density=%.4f k=%.0f restr_E=%.4g restr_max_disp=%.2f A"
          % (LABEL, a_heat["pose_rmsd"], a_heat["retention"], a_heat["fp_retention"],
             a_heat["d_min"], a_heat["density"], a_heat["k"], a_heat["restr_e"],
             a_heat["restr_maxd"]), flush=True)

    # ---- restraint release: NPT, monotone k, COM-refreshed anchors ---------------
    add_barostat()
    for k, ps in RELEASE:
        sim.context.setParameter("k", k)
        n = int(ps / 0.002)
        done = 0
        while done < n:
            step = min(COM_REFRESH_STEPS, n - done)
            sim.step(step)
            done += step
            refresh_reference()
    a_eq = audit()
    print("[%s] after release: pose_rmsd=%.2f retention=%.3f fp=%.3f d_min=%.2f "
          "density=%.4f k=%.0f restr_E=%.4g restr_max_disp=%.2f A"
          % (LABEL, a_eq["pose_rmsd"], a_eq["retention"], a_eq["fp_retention"],
             a_eq["d_min"], a_eq["density"], a_eq["k"], a_eq["restr_e"],
             a_eq["restr_maxd"]), flush=True)
    status("PREPARED", {"pose_rmsd": round(a_eq["pose_rmsd"], 2),
                        "density": round(a_eq["density"], 3)})
    if not (DENSITY_RANGE[0] <= a_eq["density"] <= DENSITY_RANGE[1]):
        status("TECH_FAIL", {"reason": "density_collapse_at_prep", "density": a_eq["density"]})
        sys.exit(1)
    PE_REF = a_eq["pe"]

    if PREP_ONLY:
        st = sim.context.getState(getPositions=True)
        with open(os.path.join(WD, "prepped.pdb"), "w") as f:
            app.PDBFile.writeFile(prmtop.topology, st.getPositions(), f)
        status("PREP_ONLY_DONE", {"pose_rmsd": round(a_eq["pose_rmsd"], 2)})
        print("[%s] PREP_ONLY done: pose_rmsd=%.2f d_min=%.2f density=%.4f"
              % (LABEL, a_eq["pose_rmsd"], a_eq["d_min"], a_eq["density"]), flush=True)
        sys.exit(0)

    # ---- production: free, no pose-based early stop ------------------------------
    n_prod = int(N_PROD_NS * 500000)
    audit_every = int(AUDIT_EVERY_NS * 500000)
    sim.reporters.append(app.DCDReporter(os.path.join(WD, "traj.dcd"), n_prod // 200))
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, 50000, step=True, speed=True, progress=True, remainingTime=True,
        totalSteps=sim.currentStep + n_prod, separator="\t"))

    cols = ["time_ns", "retention", "pose_rmsd", "pose_disp", "fp_retention", "d_pocket",
            "d_min", "d_min_p50", "n_contact", "l_rmsd_legacy", "pe", "density",
            "midplane_z", "bilayer_thickness", "ligand_z_vs_midplane"]
    fcsv = open(os.path.join(WD, "audit.csv"), "w")
    fcsv.write(",".join(cols) + "\n")
    ivals = ("n_contact",)
    t0 = time.time()
    pes = []
    for kk in range(1, int(N_PROD_NS / AUDIT_EVERY_NS) + 1):
        sim.step(audit_every)
        t_ns = kk * AUDIT_EVERY_NS
        a = audit()
        pes.append(a["pe"])
        fcsv.write(",".join(
            ("%.2f" % t_ns) if c == "time_ns" else
            ("%d" % a.get(c, 0)) if c in ivals else
            ("%.6g" % a[c]) if c == "pe" else
            ("%.4f" % a[c]) if c == "density" else
            ("%.3f" % a.get(c, float("nan")))
            for c in cols) + "\n")
        fcsv.flush()
        print("[%s] t=%.2fns ret=%.3f pose=%.2f fp=%.2f dmin=%.2f disp=%.2f" %
              (LABEL, t_ns, a["retention"], a["pose_rmsd"], a["fp_retention"],
               a["d_min"], a["pose_disp"]), flush=True)
        native_now = (a["retention"] >= RETENTION_MIN) and (a["pose_rmsd"] <= POSE_NATIVE_MAX)
        status("RUNNING", {"t_ns": t_ns, "retention": round(a["retention"], 3),
                           "pose_rmsd": round(a["pose_rmsd"], 2), "native": bool(native_now)})
        # technical aborts only; block-averaged PE drift
        per_block = int(PE_BLOCK_NS / AUDIT_EVERY_NS)
        if len(pes) >= 2 * per_block:
            b1 = float(np.mean(pes[-2 * per_block:-per_block]))
            b2 = float(np.mean(pes[-per_block:]))
            if abs(b1) > 0 and abs(b2 - b1) / abs(b1) > PE_DRIFT_MAX:
                fcsv.close()
                status("TECH_FAIL", {"reason": "pe_drift_block", "t_ns": t_ns,
                                     "block_delta_frac": round(abs(b2 - b1) / abs(b1), 3)})
                sys.exit(1)
        if not (DENSITY_RANGE[0] <= a["density"] <= DENSITY_RANGE[1]):
            fcsv.close()
            status("TECH_FAIL", {"reason": "density_collapse", "t_ns": t_ns})
            sys.exit(1)
    fcsv.close()
    timings["prod_s"] = round(time.time() - t0)
    timings["ns_per_day"] = round(N_PROD_NS / ((time.time() - t0) / 86400), 1)

    st = sim.context.getState(getPositions=True)
    with open(os.path.join(WD, "final.pdb"), "w") as f:
        app.PDBFile.writeFile(prmtop.topology, st.getPositions(), f)

    hdr = open(os.path.join(WD, "audit.csv")).read().splitlines()[0].split(",")
    rows = [ln.split(",") for ln in open(os.path.join(WD, "audit.csv")).read().splitlines()[1:] if ln.strip()]
    ix = {h: i for i, h in enumerate(hdr)}
    ret = np.array([float(r[ix["retention"]]) for r in rows])
    pos_r = np.array([float(r[ix["pose_rmsd"]]) for r in rows])
    fp_r = np.array([float(r[ix["fp_retention"]]) for r in rows])
    t_all = np.array([float(r[ix["time_ns"]]) for r in rows])
    native = (ret >= RETENTION_MIN) & (pos_r <= POSE_NATIVE_MAX)
    F_native = float(native.mean())
    # block standard error over 4 blocks of 2.5 ns
    nb = 4
    blk = [float(native[i * len(native) // nb:(i + 1) * len(native) // nb].mean()) for i in range(nb)]
    F_se = float(np.std(blk, ddof=1) / np.sqrt(nb))
    first_exit = None
    if native.any():
        first_native = int(np.argmax(native))
        for i in range(first_native, len(native) - 1):
            if not native[i] and not native[i + 1]:
                first_exit = float(t_all[i])
                break
    best = cur = 0
    for v in native:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    seen_exit, reb = False, 0
    for v in native:
        if not v:
            seen_exit = True
        elif seen_exit:
            reb += 1
            seen_exit = False
    result = {"label": LABEL, "seed": SEED, "prep": PREP, "membrane": MEMBRANE,
              "protocol": "v7-addendum5", "script_sha": SCRIPT_SHA, "metric_sha": METRIC_SHA,
              "pose_metric_sha": POSE_SHA, "fingerprint_sha": IFP_SHA,
              "native_like": f"retention>={RETENTION_MIN} and pose_rmsd<={POSE_NATIVE_MAX}A",
              "n_pocket": len(POCKET), "d_min0": round(DMIN0, 2),
              "F_native": round(F_native, 2), "F_native_block_se": round(F_se, 3),
              "F_native_blocks": [round(b, 3) for b in blk],
              "first_exit_ns": first_exit,
              "longest_dwell_ns": round(best * AUDIT_EVERY_NS, 2),
              "rebinding_events": reb,
              "retention_last2ns": round(float(ret[-8:].mean()), 2),
              "pose_rmsd_last2ns": round(float(pos_r[-8:].mean()), 2),
              "fp_retention_last2ns": round(float(fp_r[-8:].mean()), 2),
              "md": timings, "n_frames": len(rows)}
    json.dump(result, open(os.path.join(WD, "result.json"), "w"), indent=1)
    status("COMPLETED", {"F_native": result["F_native"], "F_se": result["F_native_block_se"]})
    print(f"[{LABEL}] DONE F_native={result['F_native']} (SE {result['F_native_block_se']}) "
          f"first_exit={first_exit}", flush=True)

except Exception as e:
    status("TECH_FAIL", {"reason": type(e).__name__, "detail": str(e)[:300]})
    traceback.print_exc()
    sys.exit(1)
