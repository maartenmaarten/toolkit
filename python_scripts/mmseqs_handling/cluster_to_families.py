#!/usr/bin/env python3
"""
Run mmseqs easy-cluster on an input FASTA and write one FASTA file per
cluster into <output_dir>/protein_families/.

Usage:
    python cluster_to_families.py input.fasta output_dir/ [options]

Output:
    <output_dir>/clusters_cluster.tsv   - mmseqs cluster table (rep, member)
    <output_dir>/clusters_rep_seq.fasta - cluster representative sequences
    <output_dir>/protein_families/      - one FASTA per cluster, named by rep ID
"""

import argparse
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO


def run_mmseqs(fasta, prefix, tmp_dir, min_seq_id, coverage, cov_mode, threads):
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--threads", str(threads),
    ]
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def load_clusters(cluster_tsv):
    """Return dict mapping rep_id -> [member_ids] from a two-column mmseqs TSV."""
    clusters = defaultdict(list)
    with open(cluster_tsv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            clusters[rep].append(member)
    return clusters


def _safe_filename(seq_id):
    """Strip characters that are problematic in filenames."""
    return seq_id.replace("|", "_").replace("/", "_").replace(" ", "_").replace(":", "_")


def write_families(clusters, fasta, families_dir):
    families_dir = Path(families_dir)
    families_dir.mkdir(parents=True, exist_ok=True)

    seqs = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))
    print(f"Loaded {len(seqs)} sequences from {fasta}")

    missing = 0
    for rep, members in clusters.items():
        out_file = families_dir / f"{_safe_filename(rep)}.fasta"
        records = []
        for member in members:
            if member in seqs:
                records.append(seqs[member])
            else:
                missing += 1
                print(f"  Warning: member '{member}' not found in FASTA", file=sys.stderr)

        if records:
            SeqIO.write(records, out_file, "fasta")

    print(f"Wrote {len(clusters)} cluster FASTA files to {families_dir}/")
    if missing:
        print(f"Warning: {missing} member ID(s) not found in input FASTA", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Run mmseqs easy-cluster and split results into per-cluster FASTA files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("fasta", help="Input protein FASTA file")
    parser.add_argument("output_dir", help="Directory for all outputs")
    parser.add_argument(
        "--prefix", default=None,
        help="Prefix for mmseqs output files (default: <output_dir>/clusters)"
    )
    parser.add_argument("--min-seq-id", type=float, default=0.3, help="Minimum sequence identity (0.0-1.0)")
    parser.add_argument("--coverage", "-c", type=float, default=0.3, help="Minimum coverage (0.0-1.0)")
    parser.add_argument(
        "--cov-mode", type=int, default=0,
        help="Coverage mode: 0=bidirectional, 1=query, 2=target, 3=max(query,target)"
    )
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads")
    parser.add_argument(
        "--keep-tmp", action="store_true",
        help="Keep mmseqs temporary files in <output_dir>/mmseqs_tmp/"
    )
    args = parser.parse_args()

    fasta = Path(args.fasta)
    if not fasta.exists():
        print(f"Error: FASTA file not found: {fasta}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(args.prefix) if args.prefix else out_dir / "clusters"

    # Run mmseqs easy-cluster
    if args.keep_tmp:
        tmp_dir = out_dir / "mmseqs_tmp"
        tmp_dir.mkdir(exist_ok=True)
        run_mmseqs(fasta, prefix, tmp_dir, args.min_seq_id, args.coverage, args.cov_mode, args.threads)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_mmseqs(fasta, prefix, tmp_dir, args.min_seq_id, args.coverage, args.cov_mode, args.threads)

    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        print(f"Error: mmseqs cluster TSV not found: {cluster_tsv}", file=sys.stderr)
        sys.exit(1)

    clusters = load_clusters(cluster_tsv)
    print(f"\nFound {len(clusters)} clusters")

    write_families(clusters, fasta, out_dir / "protein_families")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
