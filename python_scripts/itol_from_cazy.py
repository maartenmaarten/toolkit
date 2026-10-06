#!/usr/bin/env python3
"""
Build iTOL DATASET_COLORSTRIP files (one per column, default EC and Domain)
for the sequences in a FASTA, taken from the consolidated CAZy characterized
table (all families).

The CAZy table is subset to the FASTA IDs by matching on its GenBank column
(or --id-col). Multi-value entries are stored as Python-list strings, e.g.
"['3.2.1.37', '3.2.1.55']"; these are sorted and joined with "/" so the same
combination always gets one label and colour. Strips show colour only (the
value is in the legend and on hover), except columns in LETTER_COLS, which
print the value's first letter in the strip (Domain: B, E, U).

Usage:
    python itol_from_cazy.py --fasta data/sequence/GH43_characterized.fa \
        --cazy characterized_ALL.tsv --outdir itol/ [--col EC --col Domain]
"""

import argparse
import ast
import csv
import sys
from pathlib import Path

import seaborn as sns

PALETTE = sns.color_palette("tab20").as_hex()
DEFAULT_COLS = ["EC", "Domain"]
# columns whose strips print the value's first letter (Bacteria -> B, ...)
LETTER_COLS = {"Domain"}


def read_fasta_ids(path: str) -> list:
    with open(path) as fh:
        return [line[1:].split()[0] for line in fh if line.startswith(">")]


def _value_key(value: str) -> tuple:
    # EC numbers sort numerically per field ('-' last); other values alphabetically
    parts = value.split(".")
    if all(p.isdigit() or p == "-" for p in parts):
        return (0, tuple(int(p) if p.isdigit() else float("inf") for p in parts))
    return (1, value.lower())


def split_values(raw: str) -> set:
    raw = raw.strip()
    if raw.startswith("["):
        return set(ast.literal_eval(raw))
    return {raw} if raw else set()


def join_values(values: set) -> str:
    return "/".join(sorted(values, key=_value_key))


def read_columns(cazy_path: str, id_col: str, columns: list, wanted: set) -> dict:
    """Return {column: {seq_id: set of values}}, merging values over duplicate rows."""
    col_values = {col: {} for col in columns}
    with open(cazy_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in [id_col, *columns] if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"ERROR: column(s) {missing} not in {cazy_path}: {reader.fieldnames}")
        for row in reader:
            seq_id = row[id_col].strip()
            if seq_id not in wanted:
                continue
            for col in columns:
                col_values[col].setdefault(seq_id, []).append(split_values(row[col]))
    return col_values


def write_colorstrip(out_path: Path, dataset_label: str, id_to_value: dict, letters: bool = False):
    labels = sorted(set(id_to_value.values()), key=lambda lab: [_value_key(v) for v in lab.split("/")])
    if len(labels) > len(PALETTE):
        print(f"  WARNING: {len(labels)} labels but {len(PALETTE)} colours; colours will repeat")
    colors = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}

    lines = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        f"DATASET_LABEL\t{dataset_label}",
        "COLOR\t#000000",
        "",
        "BORDER_WIDTH\t1",
        "BORDER_COLOR\t#000000",
        "STRIP_WIDTH\t40",
        "",
        f"LEGEND_TITLE\t{dataset_label}",
        "LEGEND_SHAPES\t" + "\t".join("1" for _ in labels),
        "LEGEND_COLORS\t" + "\t".join(colors[lab] for lab in labels),
        "LEGEND_LABELS\t" + "\t".join(labels),
        "",
    ]
    if letters:
        lines += [
            "SHOW_STRIP_LABELS\t1",
            "STRIP_LABEL_POSITION\tcenter",
            "STRIP_LABEL_SIZE_FACTOR\t0.8",
            "STRIP_LABEL_COLOR\t#000000",
            # white halo keeps black text readable on the dark colours
            "STRIP_LABEL_OUTLINE\t1",
        ]
    else:
        lines.append("SHOW_STRIP_LABELS\t0")
    lines += ["", "DATA"]
    for seq_id, value in sorted(id_to_value.items()):
        text = value[0].upper() if letters else value
        lines.append(f"{seq_id}\t{colors[value]}\t{text}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}: {len(id_to_value)} sequences, {len(labels)} labels")
    for lab in labels:
        print(f"  {lab:<32} {colors[lab]}  {sum(1 for v in id_to_value.values() if v == lab)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAZy characterized table → iTOL DATASET_COLORSTRIP per column")
    parser.add_argument("--fasta", required=True, help="FASTA whose IDs select the rows to annotate")
    parser.add_argument("--cazy", required=True, help="Consolidated CAZy characterized TSV (all families)")
    parser.add_argument("--outdir", required=True, help="Output directory; writes itol_<column>.txt per column")
    parser.add_argument("--col", action="append", dest="columns",
                        help=f"Column to turn into a colour strip (repeatable, default: {' '.join(DEFAULT_COLS)})")
    parser.add_argument("--id-col", default="GenBank", help="CAZy column matching the FASTA IDs (default: GenBank)")
    args = parser.parse_args()
    columns = args.columns or DEFAULT_COLS

    fasta_ids = read_fasta_ids(args.fasta)
    if not fasta_ids:
        sys.exit(f"ERROR: no sequences in {args.fasta}")
    col_values = read_columns(args.cazy, args.id_col, columns, set(fasta_ids))

    missing = [i for i in fasta_ids if i not in col_values[columns[0]]]
    if missing:
        print(f"  WARNING: {len(missing)} FASTA ID(s) not in CAZy table (no strip): "
              + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        conflicting = [i for i, rows in col_values[col].items() if len({frozenset(r) for r in rows}) > 1]
        if conflicting:
            # same ID listed with different values (e.g. under several families): merge them
            print(f"  WARNING: {col}: {len(conflicting)} ID(s) with differing rows, merged: "
                  + ", ".join(sorted(conflicting)[:5]))
        id_to_value = {i: join_values(set().union(*rows)) for i, rows in col_values[col].items()}
        id_to_value = {i: v for i, v in id_to_value.items() if v}
        safe_name = col.lower().replace(" ", "_").replace("/", "_")
        write_colorstrip(outdir / f"itol_{safe_name}.txt", col, id_to_value, letters=col in LETTER_COLS)
