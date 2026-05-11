#!/usr/bin/env python3
"""
Convert HMMER hmmscan domtblout output to GFF3 format.

Assumes hmmscan usage where:
    - target = HMM profile (the database you searched against)
    - query  = your protein sequence

Usage:
    python domtblout_to_gff3.py <domtblout_file> -s <source_name> [-e <evalue_cutoff>] [-o <output.gff3>]

Examples:
    python domtblout_to_gff3.py dbcan_results.domtblout -s dbCAN -e 1e-5 -o dbcan.gff3
    python domtblout_to_gff3.py pfam_results.domtblout -s Pfam -e 1e-5 -o pfam.gff3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# --- Column definition for HMMER domtblout (hmmscan) ---
# Columns 0-21 are whitespace-delimited fixed fields.
# Column 22+ is a free-text description that may contain spaces.
DOMTBLOUT_COLUMNS = [
    # --- target (HMM) ---
    ("hmm_name",          str),    #  0  target name
    ("hmm_accession",     str),    #  1  target accession
    ("hmm_len",           int),    #  2  tlen
    # --- query (your protein) ---
    ("seq_name",          str),    #  3  query name
    ("seq_accession",     str),    #  4  query accession
    ("seq_len",           int),    #  5  qlen
    # --- full sequence ---
    ("full_evalue",       float),  #  6  E-value (full sequence)
    ("full_score",        float),  #  7  score (full sequence)
    ("full_bias",         float),  #  8  bias (full sequence)
    # --- this domain ---
    ("domain_number",     int),    #  9  # (domain index)
    ("domain_of",         int),    # 10  of (total domains)
    ("domain_cevalue",    float),  # 11  c-Evalue
    ("domain_ievalue",    float),  # 12  i-Evalue
    ("domain_score",      float),  # 13  score
    ("domain_bias",       float),  # 14  bias
    # --- coordinates ---
    ("hmm_from",          int),    # 15  hmm coord from
    ("hmm_to",            int),    # 16  hmm coord to
    ("ali_from",          int),    # 17  ali coord from
    ("ali_to",            int),    # 18  ali coord to
    ("env_from",          int),    # 19  env coord from
    ("env_to",            int),    # 20  env coord to
    # --- other ---
    ("acc",               float),  # 21  mean posterior probability
    ("description",       str),    # 22  description of target (free text)
]

EXPECTED_MIN_FIELDS = 23
COL_NAMES = [name for name, _ in DOMTBLOUT_COLUMNS]
COL_TYPES = {name: dtype for name, dtype in DOMTBLOUT_COLUMNS}


def parse_domtblout(filepath: Path) -> pd.DataFrame:
    """Parse HMMER domtblout into a DataFrame with validated columns."""
    rows = []
    parse_errors = []

    with open(filepath) as fh:
        for line_num, line in enumerate(fh, 1):
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            if len(fields) < EXPECTED_MIN_FIELDS:
                parse_errors.append(
                    f"  Line {line_num}: expected >= {EXPECTED_MIN_FIELDS} "
                    f"fields, got {len(fields)}"
                )
                continue

            # First 22 fields are fixed; remainder is free-text description
            row = fields[:22] + [" ".join(fields[22:])]
            rows.append(row)

    if parse_errors:
        print(
            f"WARNING: {len(parse_errors)} lines skipped due to parsing issues:",
            file=sys.stderr,
        )
        for err in parse_errors[:10]:
            print(err, file=sys.stderr)
        if len(parse_errors) > 10:
            print(f"  ... and {len(parse_errors) - 10} more", file=sys.stderr)

    if not rows:
        print("ERROR: no valid data lines found in domtblout file.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows, columns=COL_NAMES)

    # --- Type casting and validation ---
    casting_errors = []
    for col, dtype in DOMTBLOUT_COLUMNS:
        if col == "description":
            continue
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError) as e:
            casting_errors.append(f"  Column '{col}' (expected {dtype.__name__}): {e}")

    if casting_errors:
        print(
            "ERROR: column type validation failed. This file may not be a "
            "valid hmmscan domtblout:",
            file=sys.stderr,
        )
        for err in casting_errors:
            print(err, file=sys.stderr)
        sys.exit(1)

    # --- Sanity checks on parsed values ---
    problems = []

    # Coordinate checks: start <= end, positive values
    for prefix in ("hmm", "ali", "env"):
        col_from, col_to = f"{prefix}_from", f"{prefix}_to"
        if (df[col_from] > df[col_to]).any():
            n_bad = (df[col_from] > df[col_to]).sum()
            problems.append(
                f"  {n_bad} rows where {col_from} > {col_to}"
            )
        if (df[col_from] < 1).any():
            problems.append(f"  {col_from} contains values < 1")

    # E-values should be non-negative
    for col in ("full_evalue", "domain_cevalue", "domain_ievalue"):
        if (df[col] < 0).any():
            problems.append(f"  {col} contains negative values")

    # Domain numbering
    if (df["domain_number"] > df["domain_of"]).any():
        problems.append("  domain_number exceeds domain_of in some rows")

    # Envelope within sequence length
    if (df["env_to"] > df["seq_len"]).any():
        n_bad = (df["env_to"] > df["seq_len"]).sum()
        problems.append(
            f"  {n_bad} rows where env_to exceeds seq_len — possible "
            f"target/query swap (are you sure this is hmmscan, not hmmsearch?)"
        )

    # Accuracy should be between 0 and 1
    if (df["acc"] < 0).any() or (df["acc"] > 1).any():
        problems.append("  accuracy values outside [0, 1] range")

    if problems:
        print("WARNING: sanity check issues detected:", file=sys.stderr)
        for p in problems:
            print(p, file=sys.stderr)
        print(
            "These may indicate the file is not from hmmscan, or is "
            "corrupted. Proceeding, but inspect your output carefully.",
            file=sys.stderr,
        )

    print(
        f"Parsed {len(df)} domain hits for {df['seq_name'].nunique()} "
        f"unique sequences.",
        file=sys.stderr,
    )

    return df


def to_gff3(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Convert parsed domtblout DataFrame to GFF3 columns.

    For hmmscan:
        seqid  = query (your protein)
        Name   = target (HMM hit)
    Coordinates are envelope positions on the protein.
    """
    gff = pd.DataFrame({
        "seqid":      df["seq_name"],
        "source":     source,
        "type":       "protein_hmm_match",
        "start":      df["env_from"],
        "end":        df["env_to"],
        "score":      df["domain_score"].map(lambda x: f"{x:.1f}"),
        "strand":     ".",
        "phase":      ".",
        "attributes": (
            "Name=" + df["hmm_name"]
            + ";Accession=" + df["hmm_accession"]
            + ";Domain=" + df["domain_number"].astype(str)
            + "_of_" + df["domain_of"].astype(str)
            + ";Evalue=" + df["domain_ievalue"].map(lambda x: f"{x:.2e}")
            + ";Score=" + df["domain_score"].map(lambda x: f"{x:.1f}")
            + ";Accuracy=" + df["acc"].map(lambda x: f"{x:.2f}")
            + ";Note=" + df["description"]
        ),
    })
    return gff


def main():
    parser = argparse.ArgumentParser(
        description="Convert HMMER hmmscan domtblout to GFF3 format."
    )
    parser.add_argument("domtblout", help="HMMER domtblout file (from hmmscan)")
    parser.add_argument(
        "-s", "--source", required=True,
        help="Source database name for GFF3 column 2 (e.g., 'dbCAN', 'Pfam')",
    )
    parser.add_argument(
        "-e", "--evalue", type=float, default=None,
        help="Domain i-Evalue cutoff (default: no filtering)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output GFF3 file (default: stdout)",
    )
    args = parser.parse_args()

    if not Path(args.domtblout).exists():
        print(f"Error: file not found: {args.domtblout}", file=sys.stderr)
        sys.exit(1)

    # Parse and validate
    df = parse_domtblout(Path(args.domtblout))

    # Filter by E-value
    n_before = len(df)
    if args.evalue is not None:
        df = df[df["domain_ievalue"] <= args.evalue].copy()
    n_after = len(df)

    if n_after == 0:
        print(
            "WARNING: all hits filtered out. Consider relaxing the E-value "
            "cutoff.",
            file=sys.stderr,
        )

    # Convert to GFF3
    gff = to_gff3(df, args.source)

    # Write output
    out = open(args.output, "w") if args.output else sys.stdout
    out.write("##gff-version 3\n")
    gff.to_csv(out, sep="\t", header=False, index=False)

    if args.output:
        out.close()

    print(
        f"Wrote {n_after}/{n_before} domain hits "
        f"(i-Evalue cutoff: {args.evalue if args.evalue else 'none'})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()