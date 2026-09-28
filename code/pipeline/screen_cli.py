"""Screen CLI: `python -m app.screen_cli --config run.yaml [--dry-run]`.

One command runs the full audited flow (see app/screen_flow.py):
manifest -> library load -> L1 (cache-pinned) -> adaptive top-K ->
co-fold + gate + Nesso -> per-molecule results.jsonl (full signals,
consensus default ranking, failure classes) -> Z' -> annotated hit list.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m app.screen_cli",
        description="Full audited virtual-screening run (W3 canonical flow)")
    parser.add_argument("--config", required=True, help="YAML/JSON ScreenConfig")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve + print config, do not run")
    parser.add_argument("--preflight", action="store_true",
                        help="validate config/libraries + print time estimates, do not run")
    parser.add_argument("--cofold-k", type=int, default=None,
                        help="override co-fold batch size")
    parser.add_argument("--out-root", default="/workspace/output",
                        help="output root (default /workspace/output)")
    args = parser.parse_args(argv)

    from app.screen_flow import ScreenConfig, ScreenFlow
    cfg = ScreenConfig.load(args.config)
    if args.cofold_k is not None:
        cfg.cofold_batch_k = args.cofold_k

    if args.dry_run:
        print(json.dumps(cfg.resolved(), indent=2))
        return 0

    if args.preflight:
        problems = []
        if not cfg.library_ids and not cfg.library_sdfs:
            problems.append("no library: set library_ids or library_sdfs")
        if not cfg.positives:
            report["warnings"].append(
                "no positive control — run has no assay anchor (Z' not computable)")
        report = {"run_id": cfg.run_id, "target": cfg.target_name,
                  "problems": problems, "libraries": [], "warnings": []}
        if cfg.library_ids:
            try:
                from app.library_registry import LibraryRegistry, load_cached
                paths, metas = LibraryRegistry().resolve_ids(cfg.library_ids)
                smiles, _ = load_cached(paths)
                for m in metas:
                    report["libraries"].append(f"{m['lib_id']}@{m['version']}: {m['n_molecules']}")
                    leftover = m["card"]["unresolved"] + m["card"]["ambiguous"]
                    if leftover:
                        report["warnings"].append(
                            f"{m['lib_id']}@{m['version']}: {leftover} identifier records "
                            f"unresolved/ambiguous — molecules absent from this run")
                n = len(smiles) + len(cfg.positives) + cfg.n_null_control
                l1_s = n * 0.43
                cf_s = (cfg.cofold_batch_k + len(cfg.positives) + cfg.n_null_control) * 145
                report["library_mols_after_dedup"] = len(smiles)
                report["estimates"] = {"l1_h": round(l1_s / 3600, 1),
                                       "cofold_h": round(cf_s / 3600, 1),
                                       "total_h": round((l1_s + cf_s) / 3600, 1)}
            except (KeyError, FileNotFoundError) as e:
                report["problems"].append(str(e))
        print(json.dumps(report, indent=1))
        return 1 if report["problems"] else 0

    summary = ScreenFlow(cfg, out_root=args.out_root).run()
    print(f"[screen_cli] run complete -> see summary.json in run dir")
    return 0


if __name__ == "__main__":
    sys.exit(main())
