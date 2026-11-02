#!/usr/bin/env python3
"""
Convert any annotation column in a TSV/CSV to an iTOL DATASET_COLORSTRIP file.

Input: plain TSV/CSV with a header row.
    seq ID column auto-detected from: id, seq_id, sequence_id, protein_id, protein, accession
    (or pass --id-col explicitly)

Pass one or more --col arguments, each produces its own DATASET_COLORSTRIP file.

Pass --tree to only write entries for the leaf labels in that tree (Newick by default,
see --tree-format). Colors and legends are then built from the tree's leaves only.

Usage:
    python itol_from_tsv.py --tsv annotations.tsv --outdir itol_annotations/ \
        --col EC --col Phylum --col Genus --tree tree.nwk
"""

import csv
import sys
from pathlib import Path
from typing import Optional
import seaborn as sns

ID_ALIASES = {"id", "seq_id", "sequence_id", "protein_id", "protein", "accession"}

PALETTE = sns.color_palette("tab20").as_hex()

def _match_col(headers: list, aliases: set) -> Optional[str]:
    for h in headers:
        if h.strip().lower() in {a.lower() for a in aliases}:
            return h
    return None


def _sniff_dialect(path: str):
    with open(path) as fh:
        sample = fh.read(4096)
    return csv.Sniffer().sniff(sample, delimiters="\t")


def read_tree_leaves(tree_path: str, tree_format: str = "newick") -> set:
    from Bio import Phylo

    tree = Phylo.read(tree_path, tree_format)
    leaves = {t.name.strip() for t in tree.get_terminals() if t.name}
    if not leaves:
        sys.exit(f"ERROR: no leaf labels found in tree {tree_path}")
    return leaves


def assign_colors(values: list) -> dict:
    unique = sorted(set(v for v in values if v))
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(unique)}


def write_colorstrip(out_path: str, dataset_label: str, id_to_value: dict, colors: dict):
    lines = []
    lines.append("DATASET_COLORSTRIP")
    lines.append("SEPARATOR TAB")
    lines.append(f"DATASET_LABEL\t{dataset_label}")
    lines.append("COLOR\t#000000")
    lines.append("")
    lines.append(f"LEGEND_TITLE\t{dataset_label}")
    sorted_vals = sorted(colors)
    lines.append("LEGEND_SHAPES\t" + "\t".join("1" for _ in sorted_vals))
    lines.append("LEGEND_COLORS\t" + "\t".join(colors[v] for v in sorted_vals))
    lines.append("LEGEND_LABELS\t" + "\t".join(sorted_vals))
    lines.append("")
    lines.append("DATA")
    for seq_id, value in sorted(id_to_value.items()):
        if not value:
            continue
        lines.append(f"{seq_id}\t{colors[value]}\t{value}")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}: {len(id_to_value)} sequences, {len(colors)} unique values")


def convert_columns(
    tsv_path: str,
    outdir: str,
    columns: list,
    id_col: Optional[str] = None,
    keep_ids: Optional[set] = None,
):
    dialect = _sniff_dialect(tsv_path)
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        headers = reader.fieldnames or []
        if not headers:
            sys.exit("ERROR: annotations file appears to be empty or has no header row")

        resolved_id_col = id_col or _match_col(headers, ID_ALIASES)
        if resolved_id_col is None:
            sys.exit(
                f"ERROR: could not auto-detect sequence ID column.\n"
                f"Available columns: {headers}\n"
                "Pass --id-col explicitly."
            )

        missing = [c for c in columns if c not in headers]
        if missing:
            sys.exit(f"ERROR: column(s) not found: {missing}. Available columns: {headers}")

        value_maps = {col: {} for col in columns}
        for row in reader:
            seq_id = row[resolved_id_col].strip()
            if not seq_id:
                continue
            if keep_ids is not None and seq_id not in keep_ids:
                continue
            for col in columns:
                value_maps[col][seq_id] = row[col].strip()

    if keep_ids is not None:
        found = set(value_maps[columns[0]])
        print(f"  Tree leaves: {len(keep_ids)}, matched in annotations: {len(found)}")
        not_found = sorted(keep_ids - found)
        if not_found:
            preview = ", ".join(not_found[:10]) + (" ..." if len(not_found) > 10 else "")
            print(f"  WARNING: {len(not_found)} tree leaves not in annotations: {preview}")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    for col in columns:
        id_to_value = value_maps[col]
        colors = assign_colors(list(id_to_value.values()))
        safe_name = col.lower().replace(" ", "_")
        write_colorstrip(
            str(Path(outdir) / f"itol_{safe_name}.txt"),
            dataset_label=col,
            id_to_value=id_to_value,
            colors=colors,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TSV column(s) → iTOL DATASET_COLORSTRIP files")
    parser.add_argument("--tsv", required=True, help="Input TSV/CSV with annotations")
    parser.add_argument("--outdir", required=True, help="Output directory for iTOL files")
    parser.add_argument(
        "--col", action="append", required=True, dest="columns",
        help="Column name to convert (repeatable, e.g. --col EC --col Phylum)"
    )
    parser.add_argument("--id-col", help="Override auto-detected sequence ID column")
    parser.add_argument("--tree", help="Tree file; only its leaf labels are written")
    parser.add_argument(
        "--tree-format", default="newick",
        help="Tree format for --tree, as accepted by Bio.Phylo (default: newick)"
    )
    args = parser.parse_args()

    keep_ids = read_tree_leaves(args.tree, args.tree_format) if args.tree else None
    convert_columns(args.tsv, args.outdir, args.columns, args.id_col, keep_ids)