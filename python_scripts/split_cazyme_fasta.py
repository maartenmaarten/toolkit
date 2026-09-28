#!/usr/bin/env python3
"""
Split a protein FASTA file into per-CAZy-family FASTA files, using a
consolidated annotation TSV (e.g. sequences_ALL.tsv) to look up each
sequence's family.

The annotation TSV has a header and is joined to the FASTA on accession
(default columns: GenBank / Family). The Family cell may be a plain string
("GH43") or a Python-list repr for multi-family proteins
("['GH0', 'GH43']") - sequences with multiple families are written to each
family's output file.

Usage:
    python split_cazyme_fasta.py --fasta all_cazymes.fa --outdir split_by_family
"""

import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO

DEFAULT_ANNOTATION = (
    "/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/"
    "2026-06-29_download/consolidated/sequences_ALL.tsv"
)

UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def parse_families(raw):
    """Parse a Family cell that may be a plain string or a Python-list repr."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            pass
    return [raw]


def load_family_map(annotation_path, id_col, family_col, wanted_ids):
    """Map accession -> list of families, for accessions in `wanted_ids`.

    Streams the (potentially huge) annotation TSV rather than loading it
    fully, since only a subset of its accessions are usually present in
    any given input FASTA.
    """
    family_map = {}
    with open(annotation_path, newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            id_idx = header.index(id_col)
            family_idx = header.index(family_col)
        except ValueError as e:
            raise SystemExit(
                f"Column not found in {annotation_path} header {header}: {e}"
            )
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(id_idx, family_idx):
                continue
            accession = fields[id_idx]
            if accession not in wanted_ids:
                continue
            family_map[accession] = parse_families(fields[family_idx])
    return family_map


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fasta", required=True, type=Path,
                        help="Input protein FASTA file to split")
    parser.add_argument("--annotation", type=Path, default=Path(DEFAULT_ANNOTATION),
                        help="Annotation TSV mapping accession to CAZy family")
    parser.add_argument("--id-col", default="GenBank",
                        help="Annotation column matching FASTA sequence IDs (default: GenBank)")
    parser.add_argument("--family-col", default="Family",
                        help="Annotation column giving the CAZy family (default: Family)")
    parser.add_argument("--outdir", type=Path, default=Path("split_by_family"),
                        help="Directory to write per-family FASTA files into")
    args = parser.parse_args()

    print(f"Loading sequences from {args.fasta}")
    records = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"  {len(records)} sequences")

    print(f"Loading family annotations from {args.annotation}")
    family_map = load_family_map(
        args.annotation, args.id_col, args.family_col, {r.id for r in records}
    )
    print(f"  {len(family_map)} matching annotation rows")

    records_by_family = defaultdict(list)
    n_missing = 0
    for record in records:
        families = family_map.get(record.id)
        if not families:
            n_missing += 1
            continue
        for family in families:
            records_by_family[family].append(record)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for family, family_records in sorted(records_by_family.items()):
        safe_name = UNSAFE_FILENAME_CHARS.sub("_", family)
        out_path = args.outdir / f"{safe_name}.fasta"
        SeqIO.write(family_records, out_path, "fasta")
        print(f"  {family}: {len(family_records)} sequences -> {out_path}")

    print(f"Wrote {len(records_by_family)} family FASTA files to {args.outdir}")
    if n_missing:
        print(f"  WARNING: {n_missing} sequence(s) had no matching family annotation - skipped")


if __name__ == "__main__":
    main()
