"""Library registry: format-agnostic library onboarding with versioning.

Designed 2026-08-31 after the new-library onboarding time sink (missing
SMILES props, NBSP paths, hand-edited YAML, per-run SDF re-parsing).
Architecture (four orthogonal layers, reviewed and approved):

  ① format detection  — extension + content sniff
  ② extractor         — per-format module -> RawRecord(ids, structure|None)
  ③ normalizer        — SMILES/InChI/molblock -> RDKit mol -> canonical
                         SMILES + InChIKey (sanitize fallback lives here ONLY)
  ④ resolver          — identifier-only records: local InChIKey index ->
                         online PubChem (default ON, auto-degrades offline,
                         ambiguities go to pending_review.csv for agent
                         review) -> unresolved list (never silently dropped)

Approved decisions: 5% unparseable+unresolved gate (--force overrides with
recorded reason); cache at cache/libraries/ (registry.json travels with it);
versioned library ids ("L5610", "L5610@v2", default = latest).

Runs load ONLY the canonical cache — original SDF paths and their encoding
quirks are structurally disconnected from run time.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Si"}
CACHE_ROOT = Path(__file__).parent.parent / "cache" / "libraries"
GATE_FRACTION = 0.05
ONLINE_TIMEOUT_S = 4.0
ONLINE_SLEEP_S = 0.22  # <=5 req/s PubChem etiquette


@dataclass
class RawRecord:
    ids: dict = field(default_factory=dict)      # {"cas":..., "name":..., "inchikey":...}
    structure: str | None = None                 # SMILES / InChI / molblock
    props: dict = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_text_guess_encoding(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ── extractors ───────────────────────────────────────────────────────────
class SdfExtractor:
    format_id = "sdf"

    def probe(self, path: Path) -> bool:
        return path.suffix.lower() in (".sdf", ".sd") or (
            path.suffix == "" and b"$$$$" in path.read_bytes()[:65536])

    def extract(self, path: Path):
        from rdkit import Chem
        from rdkit.Chem import MolToSmiles
        for m in Chem.SDMolSupplier(str(path), sanitize=False, strictParsing=False):
            if m is None:
                yield RawRecord(ids={"parse_error": "sdf_mol"})
                continue
            ids = {}
            for k in ("Name", "ID", "CAS", "CAS_NO"):
                if m.HasProp(k):
                    ids[k.lower()] = m.GetProp(k).strip()
            if m.HasProp("SMILES"):
                yield RawRecord(ids=ids, structure=m.GetProp("SMILES"))
            elif m.HasProp("InChI") and m.GetProp("InChI").startswith("InChI="):
                yield RawRecord(ids=ids, structure=m.GetProp("InChI"))
            else:
                mm = Chem.Mol(m)
                smi = None
                try:
                    from rdkit import Chem as _C
                    _C.SanitizeMol(mm)
                    smi = MolToSmiles(mm)
                except Exception:
                    smi = MolToSmiles(m)
                yield RawRecord(ids=ids, structure=smi)


KNOWN_COLS = {"smiles", "canonicalsmiles", "smile", "inchi", "inchikey",
              "name", "cas", "casno", "id"}


class SmiCsvExtractor:
    format_id = "smi_csv"

    def probe(self, path: Path) -> bool:
        return path.suffix.lower() in (".smi", ".csv", ".tsv", ".txt")

    def extract(self, path: Path):
        text = _read_text_guess_encoding(path)
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return
        first = lines[0]
        delim = "\t" if "\t" in first else ("," if "," in first else None)
        if delim:
            head = first.split(delim)[0].strip().lower()
            if head in KNOWN_COLS:
                rows = csv.DictReader(lines, delimiter=delim)
            else:  # delimited without header: first column is smiles
                rows = ({"smiles": l.split(delim)[0].strip()} for l in lines)
        else:
            if first.split()[0].lower() in KNOWN_COLS:  # header line
                lines = lines[1:]
            rows = ({"smiles": l.split()[0]} for l in lines)
        for row in rows:
            low = {str(k).strip().lower().replace(" ", ""): (v or "").strip()
                   for k, v in row.items() if k}
            smi = low.get("smiles") or next(
                (v for k, v in low.items() if k in ("canonicalsmiles", "smile")), None)
            if smi:
                ids = {k: v for k, v in low.items()
                       if k in ("name", "cas", "casno", "inchikey") and v}
                yield RawRecord(ids=ids, structure=smi)
            else:
                first = next(iter(low.values()), "")
                yield RawRecord(ids={"raw": first} if first else {})


class XlsxExtractor:
    format_id = "xlsx_table"

    def probe(self, path: Path) -> bool:
        return path.suffix.lower() in (".xlsx", ".xlsm")

    def extract(self, path: Path):
        try:
            from openpyxl import load_workbook
        except ImportError as e:
            raise RuntimeError("xlsx support needs openpyxl: pip install openpyxl") from e
        wb = load_workbook(str(path), read_only=True, data_only=True)
        for ws in wb.worksheets:
            rows = ws.iter_rows(values_only=True)
            header = [str(h).strip().lower().replace(" ", "") if h else "" for h in next(rows, [])]
            if not header:
                continue
            idx = {name: i for i, name in enumerate(header) if name}
            for r in rows:
                if r is None or all(v is None for v in r):
                    continue
                rec = {}
                ids = {}
                for name in ("smiles", "canonicalsmiles", "inchi", "inchikey", "name", "cas", "casno", "id"):
                    i = idx.get(name)
                    if i is not None and i < len(r) and r[i] is not None:
                        v = str(r[i]).strip()
                        if not v:
                            continue
                        if name in ("smiles", "canonicalsmiles", "inchi"):
                            rec["structure_hint"] = v
                        elif name == "inchikey":
                            ids["inchikey"] = v
                        else:
                            ids[name] = v
                structure = rec.get("structure_hint")
                if not structure and ids.get("inchikey", "").startswith("InChI="):
                    structure, ids = ids["inchikey"], {}
                yield RawRecord(ids=ids, structure=structure)


class PlainInchikeyExtractor:
    format_id = "plain_inchikey"

    IK = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]")

    def probe(self, path: Path) -> bool:
        try:
            head = _read_text_guess_encoding(path).splitlines()[:5]
        except Exception:
            return False
        vals = [l.split()[0] for l in head if l.strip()]
        return bool(vals) and all(self.IK.match(v) for v in vals)

    def extract(self, path: Path):
        for line in _read_text_guess_encoding(path).splitlines():
            if line.strip():
                yield RawRecord(ids={"inchikey": line.split()[0]})


EXTRACTORS = [PlainInchikeyExtractor(), XlsxExtractor(), SdfExtractor(), SmiCsvExtractor()]


def detect_format(path: Path):
    for ex in EXTRACTORS:
        try:
            if ex.probe(path):
                return ex
        except Exception:
            continue
    return None


# ── normalizer ───────────────────────────────────────────────────────────
def normalize(structure: str) -> tuple[str, str, str] | None:
    """-> (canonical_smiles, inchikey, problem|'') ; None = fatal."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MolToSmiles, MolToInchiKey
    RDLogger.DisableLog("rdApp.*")
    mol = None
    if structure.startswith("InChI="):
        mol = Chem.MolFromInchi(structure)
    else:
        mol = Chem.MolFromSmiles(structure)
        if mol is None:
            mol = Chem.MolFromSmiles(structure, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    return None, "", "kekulize"
    if mol is None:
        return None, "", "unparseable"
    if any(a.GetSymbol() not in ORGANIC for a in mol.GetAtoms()):
        return None, "", "metal"
    smi = MolToSmiles(mol)
    try:
        ik = MolToInchiKey(mol)
    except Exception:
        ik = ""
    return smi, ik, ""


# ── resolver ─────────────────────────────────────────────────────────────
class Resolver:
    """Identifier-only records: local index -> PubChem -> pending/unresolved."""

    def __init__(self, index_path: Path, online: bool = True, verbose: bool = True):
        self.index_path = index_path
        self.index = {}
        if index_path.exists():
            for line in index_path.read_text(encoding="utf-8").splitlines():
                p = line.split("\t")
                if len(p) >= 2 and p[0]:
                    self.index[p[0]] = p[1]
        self.online = online
        self._degraded = False
        self.verbose = verbose
        self.stats = {"local": 0, "online": 0, "ambiguous": 0, "unresolved": 0}

    def _pubchem(self, url: str) -> dict | list | None:
        import requests
        try:
            r = requests.get(url, timeout=ONLINE_TIMEOUT_S)
            time.sleep(ONLINE_SLEEP_S)
            if r.status_code == 200:
                return r.json()
            return None
        except Exception:
            self._degraded = True
            return None

    def resolve(self, ids: dict) -> tuple[str | None, str, str]:
        """-> (smiles|None, provenance, note). Ambiguity note starts 'ambiguous'."""
        ik = ids.get("inchikey")
        if ik and ik in self.index:
            self.stats["local"] += 1
            return self.index[ik], f"local", ""
        if self.online and not self._degraded and ik:
            data = self._pubchem(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
                f"{ik}/property/CanonicalSMILES/JSON")
            props = (data or {}).get("PropertyTable", {}).get("Properties") or []
            smi = _pc_smiles(props[0]) if props else None
            if smi:
                self.stats["online"] += 1
                return smi, "online:pubchem", ""
        name = ids.get("name") or ids.get("cas")
        if self.online and not self._degraded and name:
            data = self._pubchem(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{requests_quote(name)}/cids/JSON")
            cids = (data or {}).get("IdentifierList", {}).get("CID", []) if data else []
            if len(cids) == 1:
                d2 = self._pubchem(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                    f"{cids[0]}/property/CanonicalSMILES/JSON")
                props2 = (d2 or {}).get("PropertyTable", {}).get("Properties") or []
                smi2 = _pc_smiles(props2[0]) if props2 else None
                if smi2:
                    self.stats["online"] += 1
                    return smi2, "online:pubchem:name", ""
            if len(cids) > 1:
                self.stats["ambiguous"] += 1
                return None, "ambiguous", f"ambiguous:{len(cids)}_cids_for:{name}"
        self.stats["unresolved"] += 1
        return None, "unresolved", ""


def _pc_smiles(props: dict) -> str | None:
    """PubChem property key casing varies across API versions."""
    for k in ("CanonicalSMILES", "canonical_smiles", "ConnectivitySMILES",
              "connectivity_smiles", "SMILES", "smiles"):
        if k in props:
            return props[k]
    return None


def requests_quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s)


# ── registry ─────────────────────────────────────────────────────────────
def _lib_id_from_name(name: str) -> str:
    m = re.match(r"^\d+\.([A-Za-z0-9]+)", name)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Za-z0-9]+)", name)
    return m.group(1) if m else "LIB"


class LibraryRegistry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else CACHE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index").mkdir(exist_ok=True)
        self.path = self.root / "registry.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) \
            if self.path.exists() else {"libraries": {}}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False),
                             encoding="utf-8")

    def add(self, source: str | Path, lib_id: str | None = None,
            version: str | None = None, online: bool = True,
            force: bool = False, note: str = "") -> dict:
        source = Path(source)
        files = sorted(p for p in (source.rglob("*") if source.is_dir() else [source])
                       if p.is_file() and detect_format(p))
        if not files:
            raise ValueError(f"no recognizable library files under {source}")
        # plate-format twins: same stem library in 96/384 layouts -> pick 96-well
        if len(files) > 1 and source.is_dir():
            picked = [p for p in files if "96-well" in p.name] or files[:1]
            files = picked
        f = files[0]
        lib_id = lib_id or _lib_id_from_name(source.name if source.is_dir() else f.stem)
        sha = _sha256(f)
        entry = self.data["libraries"].get(lib_id)
        if entry:
            for ver, v in entry["versions"].items():
                if v["sources"][0]["sha256_16"] == sha:
                    return {"status": "idempotent", "lib_id": lib_id, "version": ver}
        ex = detect_format(f)
        resolver = Resolver(self.root / "index" / "inchikey.tsv", online=online)

        n_raw = n_direct = n_local = n_online = n_amb = n_unres = n_metal = n_bad = 0
        rows, pending, unresolved = [], [], []
        for rec in ex.extract(f):
            n_raw += 1
            smi = prov = ""
            if rec.structure:
                smi, ik, problem = normalize(rec.structure)
                if smi is None:
                    if problem == "metal":
                        n_metal += 1
                    else:
                        n_bad += 1
                    continue
                prov = "direct"
                n_direct += 1
            else:
                smi, prov, note_r = resolver.resolve(rec.ids)
                if smi is None:
                    if prov == "ambiguous":
                        n_amb += 1
                        pending.append({**rec.ids, "note": note_r})
                    else:
                        n_unres += 1
                        unresolved.append(rec.ids)
                    continue
                ik = ""
                from rdkit import Chem
                m = Chem.MolFromSmiles(smi)
                if m is None or any(a.GetSymbol() not in ORGANIC for a in m.GetAtoms()):
                    n_bad += 1
                    continue
                from rdkit.Chem import MolToInchiKey
                ik = MolToInchiKey(m)
                n_local += prov.startswith("local")
                n_online += prov.startswith("online")
            rows.append((smi, ik, prov))

        drop = n_bad + n_unres
        drop_frac = drop / max(n_raw, 1)
        if drop_frac > GATE_FRACTION and not force:
            return {"status": "rejected", "lib_id": lib_id,
                    "reason": f"drop {drop}/{n_raw} ({drop_frac:.1%}) > {GATE_FRACTION:.0%} gate",
                    "detail": {"bad": n_bad, "unresolved": n_unres},
                    "hint": "re-run with --force <reason> to register anyway"}

        versions = (entry or {"versions": {}})["versions"]
        ver = version or f"v{len(versions) + 1}"
        cache_file = self.root / f"{lib_id}_{ver}.smi"
        seen = set()
        with open(cache_file, "w", encoding="utf-8") as out:
            out.write("smiles\tinchikey\tprovenance\n")
            for smi, ik, prov in rows:
                if smi in seen:
                    continue
                seen.add(smi)
                out.write(f"{smi}\t{ik}\t{prov}\n")
        # update global index
        idx_path = self.root / "index" / "inchikey.tsv"
        have = set()
        if idx_path.exists():
            have = {l.split("\t")[0] for l in idx_path.read_text(encoding="utf-8").splitlines()[1:]}
        with open(idx_path, "a", encoding="utf-8") as idx:
            for smi, ik, prov in rows:
                if ik and ik not in have:
                    have.add(ik)
                    idx.write(f"{ik}\t{smi}\t{lib_id}\n")
        if pending:
            (self.root / "pending_review.csv").parent.mkdir(exist_ok=True)
            with open(self.root / "pending_review.csv", "a", newline="", encoding="utf-8") as pf:
                w = csv.DictWriter(pf, fieldnames=["lib_id", *sorted({k for p in pending for k in p})])
                if pf.tell() == 0:
                    w.writeheader()
                for p in pending:
                    w.writerow({"lib_id": lib_id, **p})
        if unresolved:
            (self.root / f"{lib_id}_{ver}_unresolved.csv").write_text(
                "\n".join(json.dumps(u, ensure_ascii=False) for u in unresolved), encoding="utf-8")

        card = {"format": ex.format_id, "n_raw": n_raw, "n_direct": n_direct,
                "resolved_local": n_local, "resolved_online": n_online,
                "ambiguous": n_amb, "unresolved": n_unres,
                "metal": n_metal, "unparseable": n_bad,
                "unique_cached": len(seen), "drop_fraction": round(drop_frac, 4),
                "online_degraded": resolver._degraded, "note": note}
        self.data["libraries"].setdefault(
            lib_id, {"display_name": source.name if source.is_dir() else f.stem,
                     "versions": {}})["versions"][ver] = {
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sources": [{"path": str(f), "sha256_16": sha}], "card": card,
            "cache_file": cache_file.name, "n_molecules": len(seen)}
        self.data["libraries"][lib_id]["default_version"] = ver
        self.save()
        return {"status": "registered", "lib_id": lib_id, "version": ver, "card": card}



    # ── review: apply agent decisions to pending/unresolved records ──────
    def review(self, decisions: list[dict]) -> dict:
        """Apply decisions (action: both|smiles|skip) to pending_review and
        <lib>_<ver>_unresolved.csv records. Resolved molecules are appended
        as a NEW library version (audit trail preserved)."""
        import csv as _csv
        pending_path = self.root / "pending_review.csv"
        pending = []
        if pending_path.exists():
            pending = list(_csv.DictReader(open(pending_path, encoding="utf-8")))
        unres_files = sorted(self.root.glob("*_unresolved.csv"))
        unres = []  # (file, row)
        for fp in unres_files:
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    import json as _j
                    unres.append((fp, _j.loads(line)))
        handled_pending, handled_unres = [], []
        added: dict[str, list] = {}
        skipped = []
        for dec in decisions:
            def match(row):
                return all((row.get(k) or "") == str(v) for k, v in dec.items()
                           if k in ("cas", "name", "id") and v)
            action = dec.get("action", "skip")
            mols = []
            if action == "smiles" and dec.get("smiles"):
                smi, ik, problem = normalize(dec["smiles"])
                if smi:
                    mols = [(smi, ik, "review:manual")]
            elif action == "both":
                r = Resolver(self.root / "index" / "inchikey.tsv", online=True)
                name = dec.get("name")
                if name:
                    import requests
                    try:
                        d = requests.get(
                            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                            f"name/{requests_quote(name)}/cids/JSON",
                            timeout=ONLINE_TIMEOUT_S).json()
                        for cid in d.get("IdentifierList", {}).get("CID", []):
                            d2 = requests.get(
                                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                                f"cid/{cid}/property/CanonicalSMILES/JSON",
                                timeout=ONLINE_TIMEOUT_S).json()
                            props = d2.get("PropertyTable", {}).get("Properties") or []
                            smi = _pc_smiles(props[0]) if props else None
                            if smi:
                                s2, ik2, _ = normalize(smi)
                                if s2:
                                    mols.append((s2, ik2, f"review:pubchem:cid{cid}"))
                            time.sleep(ONLINE_SLEEP_S)
                    except Exception:
                        pass
            lib_id = dec.get("lib_id")
            hit_any = False
            for row in pending[:]:
                if match(row) and (not lib_id or row.get("lib_id") == lib_id):
                    pending.remove(row)
                    handled_pending.append(row)
                    added.setdefault(row["lib_id"], []).extend(mols)
                    hit_any = True
            for fp, row in unres[:]:
                lib = fp.name.split("_unresolved")[0].rsplit("_", 1)[0]
                if match(row) and (not lib_id or lib == lib_id):
                    unres.remove((fp, row))
                    handled_unres.append((fp.name, row))
                    if mols:
                        added.setdefault(lib, []).extend(mols)
                    else:
                        skipped.append({**row, "file": fp.name})
                    hit_any = True
            if not hit_any:
                skipped.append({**dec, "note": "no matching record"})
        # write new versions
        versions_made = []
        for lib_id, mols in added.items():
            entry = self.data["libraries"][lib_id]
            cur_ver = entry.get("default_version")
            v = entry["versions"][cur_ver]
            new_ver = f"v{len(entry['versions']) + 1}"
            cache_new = self.root / f"{lib_id}_{new_ver}.smi"
            lines = (self.root / v["cache_file"]).read_text(encoding="utf-8").splitlines()
            have = {l.split("	")[0] for l in lines[1:]}
            for smi, ik, prov in mols:
                if smi not in have:
                    have.add(smi)
                    lines.append(f"{smi}	{ik}	{prov}")
            cache_new.write_text("\n".join(lines) + "\n", encoding="utf-8")
            card = dict(v["card"])
            card["note"] = (f"review-applied: +{len(mols)} mols from "
                            f"pending/unresolved decisions")
            entry["versions"][new_ver] = {
                "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "sources": v["sources"], "card": card,
                "cache_file": cache_new.name, "n_molecules": len(lines) - 1}
            entry["default_version"] = new_ver
            versions_made.append(f"{lib_id}@{new_ver}:+{len(mols)}")
        # rewrite pending + unresolved files
        if pending:
            w = _csv.DictWriter(open(pending_path, "w", newline="", encoding="utf-8"),
                                fieldnames=["lib_id", *sorted({k for p in pending for k in p})])
            w.writeheader(); w.writerows(pending)
        elif pending_path.exists():
            pending_path.unlink()
        keep_by_file: dict = {}
        for fp, row in unres:
            keep_by_file.setdefault(str(fp), []).append(row)
        for fp in unres_files:
            if str(fp) in keep_by_file:
                fp.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                                        for r in keep_by_file[str(fp)]), encoding="utf-8")
            else:
                fp.unlink()
        self.save()
        return {"versions": versions_made, "handled_pending": len(handled_pending),
                "handled_unresolved": len(handled_unres), "skipped": skipped}

    def resolve_ids(self, library_ids: list[str]) -> tuple[list[Path], list[dict]]:
        """'L5610' or 'L5610@v2' -> cache paths (default = latest) + meta."""
        paths, metas = [], []
        for token in library_ids:
            lib_id, _, ver = token.partition("@")
            entry = self.data["libraries"].get(lib_id)
            if not entry:
                raise KeyError(f"library '{lib_id}' not registered — run: "
                               f"python -m app.library_registry add <dir>")
            ver = ver or entry.get("default_version") or sorted(entry["versions"])[-1]
            if ver not in entry["versions"]:
                raise KeyError(f"version '{ver}' of '{lib_id}' not in "
                               f"{sorted(entry['versions'])}")
            v = entry["versions"][ver]
            p = self.root / v["cache_file"]
            if not p.exists():
                raise FileNotFoundError(f"cache missing for {lib_id}@{ver}: {p}")
            paths.append(p)
            metas.append({"lib_id": lib_id, "version": ver, **{k: v[k] for k in ("n_molecules", "card")}})
        return paths, metas


def load_cached(paths: list[Path]) -> tuple[list[str], dict[str, str]]:
    """Union of cache files, dedup by InChIKey first-seen. -> (smiles, lib_of)."""
    seen_ik, seen_smi, lib_of = set(), set(), {}
    out = []
    for p in paths:
        lib = p.stem.rsplit("_", 1)[0]
        lines = p.read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            smi, ik, _prov = (line.split("\t") + ["", ""])[:3]
            if not smi or smi in seen_smi:
                continue
            if ik and ik in seen_ik:
                continue
            seen_smi.add(smi)
            if ik:
                seen_ik.add(ik)
            lib_of[smi] = lib
            out.append(smi)
    return out, lib_of


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m app.library_registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("source")
    a.add_argument("--id")
    a.add_argument("--version")
    a.add_argument("--offline", action="store_true")
    a.add_argument("--force", nargs="?", const="unspecified", default=None)
    a.add_argument("--note", default="")
    sub.add_parser("list")
    rv = sub.add_parser("review")
    rv.add_argument("--decisions", required=True,
                    help="CSV: lib_id,cas,name,id,action(both|smiles|skip),smiles")
    args = ap.parse_args(argv)
    reg = LibraryRegistry()
    if args.cmd == "add":
        res = reg.add(args.source, lib_id=args.id, version=args.version,
                      online=not args.offline, force=args.force is not None,
                      note=args.note or (args.force if isinstance(args.force, str) else ""))
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 0 if res.get("status") in ("registered", "idempotent") else 1
    if args.cmd == "review":
        import csv as _csv
        decisions = list(_csv.DictReader(open(args.decisions, encoding="utf-8")))
        print(json.dumps(reg.review(decisions), indent=1, ensure_ascii=False))
        return 0
    if args.cmd == "list":
        for lib_id, e in sorted(reg.data["libraries"].items()):
            for ver, v in e["versions"].items():
                mark = "*" if ver == e.get("default_version") else " "
                c = v["card"]
                print(f"{mark} {lib_id}@{ver}: {v['n_molecules']} mols "
                      f"[{c['format']}] drop={c['drop_fraction']:.1%} "
                      f"(amb {c['ambiguous']}/unres {c['unresolved']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
