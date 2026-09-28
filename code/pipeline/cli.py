"""
Interactive CLI for ESMFold2 structure prediction.
Quick test folding for proteins, protein-ligand, and multi-chain complexes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _build_request(args) -> "PredictionRequest":
    from app.models import ChainInput, FoldingConfig, MoleculeType, PredictionRequest

    chains = []

    # Protein chain(s)
    for spec in args.protein or []:
        if ":" in spec:
            chain_id, seq = spec.split(":", 1)
        else:
            chain_id = "A"
            seq = spec
        chains.append(ChainInput(id=chain_id, sequence=seq.strip(), type=MoleculeType.protein))

    # DNA chains
    for spec in args.dna or []:
        if ":" in spec:
            chain_id, seq = spec.split(":", 1)
        else:
            chain_id, seq = "B", spec
        chains.append(ChainInput(id=chain_id, sequence=seq.strip(), type=MoleculeType.dna))

    # RNA chains
    for spec in args.rna or []:
        if ":" in spec:
            chain_id, seq = spec.split(":", 1)
        else:
            chain_id, seq = "C", spec
        chains.append(ChainInput(id=chain_id, sequence=seq.strip(), type=MoleculeType.rna))

    # Ligands (small molecules via CCD codes)
    for spec in args.ligand or []:
        if ":" in spec:
            chain_id, ccd_list = spec.split(":", 1)
        else:
            chain_id = "L"
            ccd_list = spec
        ccd_codes = [c.strip() for c in ccd_list.split(",")]
        chains.append(ChainInput(id=chain_id, sequence="", type=MoleculeType.ligand, ccd=ccd_codes))

    # FASTA file input
    if args.fasta:
        from Bio import SeqIO
        records = list(SeqIO.parse(args.fasta, "fasta"))
        for i, rec in enumerate(records):
            chain_id = chr(65 + i)  # A, B, C, ...
            chains.append(ChainInput(
                id=chain_id,
                sequence=str(rec.seq),
                type=MoleculeType.protein,
            ))

    if not chains:
        console.print("[red]Error:[/] No chains specified. Use --protein, --fasta, --dna, --rna, or --ligand.")
        sys.exit(1)

    config = FoldingConfig(
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=args.num_samples,
        seed=args.seed,
    )

    name = args.name or f"fold_{chains[0].id}"
    return PredictionRequest(name=name, chains=chains, config=config)


def _run_local(request):
    from app.local_inference import LocalInferenceEngine
    engine = LocalInferenceEngine(device="cuda" if args.gpu else "cpu")
    return engine.predict(request)


def _run_api(request):
    from app.api_client import BiohubAPIClient
    client = BiohubAPIClient()
    return client.predict(request)


def main():
    parser = argparse.ArgumentParser(
        description="ESMFold2 Interactive CLI — Predict biomolecular structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single protein
  python -m app.cli --protein MKTAYIAKQRQISFVKSHFSR

  # Protein + ligand (CCD code)
  python -m app.cli --protein MKTAYIAKQRQISFVKSHFSR --ligand SAH

  # SMILES to protein-ligand binding (auto CCD lookup)
  python -m app.cli --protein MKTAYIAKQRQISFVKSHFSR --smiles "CC(=O)O"

  # SMILES only (CCD lookup test)
  python -m app.cli --smiles "c1ccccc1" --search-only

  # Protein + DNA complex
  python -m app.cli --protein "A:MKTAYIAKQR" --dna "B:GATAGCGCTATC"

  # From FASTA file
  python -m app.cli --fasta input.fasta --output results/

  # API mode (Biohub cloud)
  python -m app.cli --mode api --protein MKTAYIAKQRQISFVKSHFSR
        """,
    )
    parser.add_argument("--protein", action="append", metavar="[CHAIN:]SEQ",
                        help="Protein chain (format: '[chain_id:]sequence')")
    parser.add_argument("--smiles", metavar="SMILES",
                        help="SMILES string of compound → auto-lookup CCD code")
    parser.add_argument("--dna", action="append", metavar="[CHAIN:]SEQ",
                        help="DNA chain")
    parser.add_argument("--rna", action="append", metavar="[CHAIN:]SEQ",
                        help="RNA chain")
    parser.add_argument("--ligand", action="append", metavar="[CHAIN:]CCD[,CCD...]",
                        help="Ligand(s) by CCD code(s)")
    parser.add_argument("--fasta", metavar="FILE", help="FASTA file input")
    parser.add_argument("--name", default=None, help="Output name prefix")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--mode", choices=["local", "api"], default="local",
                        help="Inference mode: local GPU or Biohub API")
    parser.add_argument("--search-only", action="store_true",
                        help="Only search CCD code for SMILES, don't run folding")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU (local mode)")
    parser.add_argument("--cpu", action="store_false", dest="gpu",
                        help="Force CPU (slow)")
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    global args
    args = parser.parse_args()

    # ── SMILES search-only mode ──────────────────────────────────────
    if args.smiles and args.search_only:
        from app.ccd_database import get_ccd_db
        db = get_ccd_db()
        result = db.search(args.smiles)
        console.print(f"[bold]SMILES:[/] {args.smiles}")
        console.print(f"[bold]Match type:[/] {result['match_type']}")
        if result.get("ccd_code"):
            console.print(f"[bold green]CCD Code: {result['ccd_code']}[/]")
        if result.get("candidates"):
            cand_table = Table(title="Similar CCD Candidates")
            cand_table.add_column("CCD", style="cyan")
            cand_table.add_column("SMILES", style="dim")
            cand_table.add_column("Score", style="green")
            for c in result["candidates"]:
                cand_table.add_row(c["ccd"], c["smiles"][:50], str(c["score"]))
            console.print(cand_table)
        return

    # ── Handle SMILES -> CCD lookup for ligand ───────────────────────
    if args.smiles and args.protein:
        from app.ccd_database import get_ccd_db
        db = get_ccd_db()
        search_result = db.search(args.smiles)
        ccd_code = search_result.get("ccd_code")
        if ccd_code:
            console.print(f"[bold green]SMILES → CCD: {ccd_code}[/] (match: {search_result['match_type']})")
            if not args.ligand:
                args.ligand = []
            args.ligand.append(f"L:{ccd_code}")
        else:
            console.print("[red]No CCD code found for this SMILES[/]")
            if search_result.get("candidates"):
                console.print("Did you mean one of these CCD codes?")
                for c in search_result["candidates"][:5]:
                    console.print(f"  {c['ccd']} (score: {c['score']})")
            return
    elif args.smiles and not args.protein:
        console.print("[red]--smiles requires --protein for binding prediction[/]")
        console.print("Use --search-only for CCD lookup without folding")
        return

    # Build request
    console.print("[bold cyan]Building prediction input...[/]")
    request = _build_request(args)

    # Show what we're folding
    table = Table(title="Prediction Input")
    table.add_column("Chain", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Sequence / CCD", style="dim")
    for ch in request.chains:
        seq_display = ch.sequence if ch.sequence else ", ".join(ch.ccd)
        if len(seq_display) > 60:
            seq_display = seq_display[:57] + "..."
        table.add_row(ch.id, ch.type.value, seq_display)
    console.print(table)

    # Run prediction
    console.print(f"\n[bold yellow]Running prediction ({args.mode} mode)...[/]")
    t0 = time.time()

    if args.mode == "local":
        result = _run_local(request)
    else:
        result = _run_api(request)

    elapsed = time.time() - t0

    # Save result
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.name}_{result.job_id}.cif"
    out_path.write_text(result.mmcif)

    # Report
    console.print(Panel.fit(
        f"[bold green]Prediction complete[/]\n\n"
        f"  Job ID:      {result.job_id}\n"
        f"  Chains:      {result.num_chains}\n"
        f"  Residues:    {result.num_residues}\n"
        f"  pLDDT mean:  {result.plddt_mean}\n"
        f"  pTM:         {result.ptm}\n"
        f"  ipTM:        {result.iptm or 'N/A'}\n"
        f"  Wall time:   {result.wall_time_s:.1f}s\n"
        f"  Output:      {out_path}",
        title="ESMFold2 Result",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
