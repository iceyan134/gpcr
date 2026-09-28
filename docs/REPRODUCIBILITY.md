# Reproducibility status

This inventory describes the delivered source, not the completeness of the authors' original development environment.

## Included

- Screening, scoring, preparation, benchmarking, and MD Python source.
- Original Docker recipes and shell scripts.
- Dependency declarations and historical build/operations notes.

## Required before an end-to-end reproduction claim

1. Reconcile `app.*` imports with the delivered `code/pipeline/` package and verify installation in a clean environment.
2. Supply an example FASTA, compound library, run configuration, and expected output with redistribution rights and source attribution.
3. Identify exact model versions/checkpoints and official acquisition/access instructions for each enabled model; record weight checksums where permitted.
4. Supply missing Docker build inputs, base-image provenance/digests, and Compose configuration, or replace the recipes with independently buildable versions.
5. Record a tested dependency environment, external command-line tools, GPU/driver/CUDA versions, and resource requirements.
6. Supply manuscript-specific configurations, random seeds, benchmark splits, raw scores, and analysis scripts. Do not treat numbers in source comments as verified results.
7. Replace `code/figures/make_all_figures.py` with the actual figure-generation implementation and provide its input data and panel specifications.
8. Document the boundary between default screening and separate MD validation. The canonical screening flow disables MM-GBSA; the short-MD MM-GBSA branch is unimplemented. The independent explicit-solvent MD driver is a separate program.
9. Confirm the declared license and provide copyright attribution, manuscript metadata, and a citation file.
10. After publication, record the release tag and full commit SHA. An archival DOI can be added when an archive has actually been deposited.

## Additional implementation caveats

- `boltz_scorer.py` currently invokes Nesso despite its historical filename.
- Adaptive top-K decisions are recorded, while the co-fold candidate batch is controlled by `cofold_batch_k`.
- `screen_cli.py --preflight` references `report` before assignment when no positive controls are supplied.
- MD launcher paths should be absolute. The production script removes an existing output directory with the same system/preparation/seed name; use fresh output locations when preserving prior runs.
- The batch script checks `output/.../results.jsonl`, whereas the production script writes under `runs/.../result.json`; its skip behavior needs reconciliation before relying on it for resume.

## Validation of this source snapshot

Local validation on 2026-09-29: all 73 Python files passed AST syntax parsing; all 10 cases in `python code/mdlayer/md_verdict.py --selftest` passed. A narrow credential-pattern scan found no GitHub-token or private-key patterns in the 88 supplied files. These checks do not establish successful model inference, Docker builds, GPU execution, or reproduction of paper figures.
