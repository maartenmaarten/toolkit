#!/usr/bin/env python3
"""
Combine a FASTA file with a CAZy annotation TSV into one ML-ready TSV: one
row per sequence, with its accession, sequence, and relevant annotations.
This tabular shape (accession/protein_name/organism/ec/length/sequence/...)
is what most ML tooling expects as input, rather than a bare FASTA.

Usage:
    python fasta_to_annotated_tsv.py \\
        --fasta GH43_characterized.fa \\
        --annotation characterized_ALL.tsv \\
        -o GH43_characterized_annotated.tsv
"""

import argparse
import ast
import csv
from pathlib import Path

from Bio import SeqIO

DEFAULT_FASTA = (
    "/Users/maartenboneschansker/Documents/PhD/projects/cazyme_deep_dive/"
    "GH43_deep_dive/data/genbank/GH43_characterized.fa"
)
DEFAULT_ANNOTATION = (
    "/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/"
    "2026-06-29_download/consolidated/characterized_ALL.tsv"
)

# Output columns, and which annotation TSV column each is sourced from.
# "ec" and "cazy_family" may be a plain string or a Python-list repr for
# multifunctional/multi-family proteins (e.g. "['3.2.1.37', '3.2.1.55']").
ANNOTATION_COLUMNS = {
    "protein_name": "Protein Name",
    "organism": "Organism",
    "ec": "EC",
    "domain": "Domain",
    "cazy_family": "cazy_family",
    "uniprot": "Uniprot",
}
LIST_VALUED_COLUMNS = {"ec", "cazy_family", "uniprot"}


def parse_cell(raw, sep):
    """Parse an annotation cell that may be a plain string or a Python-list
    repr, returning a `sep`-joined string either way.
    """
    raw = (raw or "").strip()
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return sep.join(str(v).strip() for v in parsed if str(v).strip())
        except (ValueError, SyntaxError):
            pass
    return raw


def load_annotations(path, id_col):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row[id_col]: row for row in reader if row[id_col]}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta", type=Path, default=Path(DEFAULT_FASTA),
                        help="Input FASTA file")
    parser.add_argument("--annotation", type=Path, default=Path(DEFAULT_ANNOTATION),
                        help="Annotation TSV with one row per sequence")
    parser.add_argument("--id-col", default="GenBank",
                        help="Column in the annotation TSV matching FASTA sequence IDs (default: GenBank)")
    parser.add_argument("--list-sep", default=";",
                        help="Separator used when an annotation cell is a multi-value list (default: ;)")
    parser.add_argument("-o", "--output", type=Path, default=Path("annotated.tsv"),
                        help="Output TSV path (default: annotated.tsv)")
    args = parser.parse_args()

    print(f"Loading annotations from {args.annotation}")
    annotations = load_annotations(args.annotation, args.id_col)
    print(f"  {len(annotations)} annotation rows")

    print(f"Loading sequences from {args.fasta}")
    records = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"  {len(records)} sequences")

    columns = ["accession", *ANNOTATION_COLUMNS, "length", "sequence"]

    n_missing = 0
    n_written = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for record in records:
            row_data = annotations.get(record.id)
            if row_data is None:
                n_missing += 1
                continue

            sequence = str(record.seq)
            row = [record.id]
            for out_col, src_col in ANNOTATION_COLUMNS.items():
                raw = row_data.get(src_col, "")
                row.append(parse_cell(raw, args.list_sep) if out_col in LIST_VALUED_COLUMNS else raw)
            row.extend([len(sequence), sequence])
            writer.writerow(row)
            n_written += 1

    print(f"Written {n_written} rows to {args.output}")
    if n_missing:
        print(f"  WARNING: {n_missing} sequence(s) had no matching annotation row — skipped")


if __name__ == "__main__":
    main()
