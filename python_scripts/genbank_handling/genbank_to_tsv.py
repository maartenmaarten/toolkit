#!/usr/bin/env python3
# Usage: python genbank_to_tsv.py <input.gb> [output.tsv] [col1,col2,...]
#
# Columns are dot-notation paths into the GenBank record:
#   id, name, description, dbxrefs
#   annotations.<key>         e.g. annotations.organism, annotations.taxonomy
#   features.<type>.<qualifier>  e.g. features.CDS.product, features.CDS.db_xref
#
# Default columns (if not specified):
#   id, description, annotations.organism, annotations.taxonomy,
#   annotations.keywords, annotations.comment,
#   features.CDS.product, features.CDS.gene, features.CDS.db_xref,
#   features.CDS.EC_number, features.CDS.note
#
# Pass --list-columns to print all available columns from the first record.

import sys
import csv
from Bio import SeqIO

DEFAULT_COLUMNS = [
    "id",
    "description",
    "annotations.organism",
    "annotations.taxonomy",
    "annotations.keywords",
    "annotations.comment",
    "features.CDS.product",
    "features.CDS.gene",
    "features.CDS.db_xref",
    "features.CDS.EC_number",
    "features.CDS.note",
]


def extract_value(record, column):
    """Extract a value from a SeqRecord using a dot-notation column spec."""
    parts = column.split(".", 2)

    if parts[0] == "annotations":
        key = parts[1] if len(parts) > 1 else None
        if key is None:
            return ""
        val = record.annotations.get(key, "")
        # taxonomy is a list — join it
        if isinstance(val, list):
            val = "; ".join(str(v) for v in val)
        return str(val) if val is not None else ""

    if parts[0] == "features":
        feat_type = parts[1] if len(parts) > 1 else None
        qualifier = parts[2] if len(parts) > 2 else None
        if feat_type is None:
            return ""
        values = []
        for feat in record.features:
            if feat.type == feat_type:
                if qualifier is None:
                    values.append(feat_type)
                else:
                    q = feat.qualifiers.get(qualifier, [])
                    values.extend(q)
        return "; ".join(values)

    # Top-level attributes: id, name, description, dbxrefs
    val = getattr(record, parts[0], "")
    if isinstance(val, list):
        val = "; ".join(str(v) for v in val)
    return str(val) if val is not None else ""


def collect_all_columns(record):
    """Return all available column paths for a record."""
    cols = ["id", "name", "description", "dbxrefs"]
    for key in record.annotations:
        cols.append(f"annotations.{key}")
    seen_types = {}
    for feat in record.features:
        if feat.type not in seen_types:
            seen_types[feat.type] = set()
        seen_types[feat.type].update(feat.qualifiers.keys())
    for ftype, qualifiers in sorted(seen_types.items()):
        for q in sorted(qualifiers):
            cols.append(f"features.{ftype}.{q}")
    return cols


def genbank_to_tsv(input_path, output_path, columns):
    records = SeqIO.parse(input_path, "genbank")

    out = open(output_path, "w", newline="") if output_path != "-" else sys.stdout
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)

    count = 0
    for record in records:
        writer.writerow([extract_value(record, col) for col in columns])
        count += 1

    if output_path != "-":
        out.close()
        print(f"Written {count} records to {output_path}", file=sys.stderr)
    return count


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    input_path = args[0]

    # --list-columns: print available columns from first record and exit
    if "--list-columns" in args:
        first = next(SeqIO.parse(input_path, "genbank"), None)
        if first is None:
            print("No records found.", file=sys.stderr)
            sys.exit(1)
        for col in collect_all_columns(first):
            print(col)
        sys.exit(0)

    output_path = args[1] if len(args) >= 2 and not args[1].startswith("--") else "-"
    columns_arg = next((a for a in args[2:] if not a.startswith("--")), None)
    columns = columns_arg.split(",") if columns_arg else DEFAULT_COLUMNS

    genbank_to_tsv(input_path, output_path, columns)


if __name__ == "__main__":
    main()
