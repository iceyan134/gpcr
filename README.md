# ReCAST-GPCR

Research source snapshot for a GPCR virtual-screening workflow combining library prescreening, protein–ligand structure prediction, confidence assessment, affinity scoring, and a separate molecular-dynamics validation layer.

## Release scope

This repository preserves the supplied research code under `code/`. The initial release identifier is `v1.0.0`; check the [release page](https://github.com/iceyan134/gpcr/releases/tag/v1.0.0) for publication status. This is a source-code snapshot, not a validated turnkey reproduction of the manuscript. Scientific source files have not been modified for this release preparation.

## Contents

| Location | Purpose |
| --- | --- |
| `code/pipeline/screen_flow.py` | Main screening orchestration |
| `code/pipeline/screen_cli.py` | Screening command-line entry point |
| `code/pipeline/cascade.py` | Structure-generation routes, confidence handling, and scoring |
| `code/pipeline/consensus.py` | Combined ranking signals |
| `code/pipeline/hit_prep.py` | Candidate follow-up processing |
| `code/pipeline/benchmark_engine.py` | Benchmark execution and metrics |
| `code/mdlayer/` | MD preparation, simulation, metrics, and classification |
| `code/pipeline/docker/` | Original Docker recipes |
| `code/scripts/` | MD launch scripts and operations notes |

## Start here

The source can be inspected without installing GPU dependencies. A dependency-free check of the MD classification rules is available from the repository root:

```bash
python code/mdlayer/md_verdict.py --selftest
```

This checks classification logic only; it does not run molecular dynamics or validate scientific performance.

For full screening, read [reproducibility status](docs/REPRODUCIBILITY.md) first. The original entry point is `python -m app.screen_cli --config <run.json>`, but the supplied package contains `pipeline/` rather than `app/`. It therefore requires packaging repair and external resources before this command can run. Installing `requirements.txt` alone is insufficient.

No Jupyter notebooks or example screening data were included in the supplied package. Full runs are intended for a configured Linux/NVIDIA GPU environment; see the original `code/pipeline/BUILD.md` and `code/scripts/RUNBOOK.md` for historical environment notes, subject to the limitations below.

## Reproducibility and licensing

Model weights, benchmark data, manuscript results, and several original build inputs are not included. The figure script is a placeholder rather than a complete figure-reproduction implementation. See [the detailed checklist](docs/REPRODUCIBILITY.md).

The supplied `code/pyproject.toml` declares MIT, but no LICENSE file or copyright attribution accompanied the source. The authors should supply the corresponding license text and attribution. Third-party software and model weights remain subject to their own terms.

Author names, manuscript title, DOI, and an archival DOI have not been supplied. No author attribution or scientific validation is inferred from the hosting account.
