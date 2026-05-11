#!/usr/bin/env python3
"""
Resolve overlapping Pfam domain hits using clan-based filtering.

Within the same clan and the same sequence, overlapping domains are
resolved by keeping only the best-scoring hit. Hits from different
clans, or families not assigned to any clan, are allowed to overlap.

Requires:
    - A Pfam hmmscan domtblout file
    - Pfam-A.clans.tsv (download from Pfam/InterPro FTP)
      Columns: pfam_accession, clan_accession, clan_name, pfam_name, pfam_description

Usage:
    python resolve_pfam_overlaps.py <domtblout> <clans_tsv> [-e EVALUE] [-o OUTPUT]

Examples:
    python resolve_pfam_overlaps.py pfam.domtblout Pfam-A.clans.tsv -e 1e-5 -o pfam_resolved.domtblout
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# --- domtblout column definitions (hmmscan) ---
DOMTBLOUT_COLUMNS = [
    ("hmm_name",       str),    #  0
    ("hmm_accession",  str),    #  1
    ("hmm_len",        int),    #  2
    ("seq_name",       str),    #  3
    ("seq_accession",  str),    #  4
    ("seq_len",        int),    #  5
    ("full_evalue",    float),  #  6
    ("full_score",     float),  #  7
    ("full_bias",      float),  #  8
    ("domain_number",  int),    #  9
    ("domain_of",      int),    # 10
    ("domain_cevalue", float),  # 11
    ("domain_ievalue", float),  # 12
    ("domain_score",   float),  # 13
    ("domain_bias",    float),  # 14
    ("hmm_from",       int),    # 15
    ("hmm_to",         int),    # 16
    ("ali_from",       int),    # 17
    ("ali_to",         int),    # 18
    ("env_from",       int),    # 19
    ("env_to",         int),    # 20
    ("acc",            float),  # 21
    ("description",    str),    # 22
]

COL_NAMES = [name for name, _ in DOMTBLOUT_COLUMNS]


def parse_domtblout(filepath: Path) -> pd.DataFrame:
    """Parse domtblout into a DataFrame, preserving raw lines for output."""
    rows = []
    raw_lines = []

    with open(filepath) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            if len(fields) < 23:
                continue
            row = fields[:22] + [" ".join(fields[22:])]
            rows.append(row)
            raw_lines.append(line.rstrip("\n"))

    if not rows:
        print("ERROR: no valid data lines found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows, columns=COL_NAMES)

    # Type casting
    for col, dtype in DOMTBLOUT_COLUMNS:
        if col == "description":
            continue
        df[col] = df[col].astype(dtype)

    # Keep the raw line for faithful output
    df["_raw_line"] = raw_lines

    return df


def parse_clans_tsv(filepath: Path) -> dict[str, str]:
    """Parse Pfam-A.clans.tsv into a dict mapping Pfam accession -> clan accession.

    Pfam-A.clans.tsv columns (tab-separated):
        0: Pfam accession  (e.g., PF00001.23)
        1: Clan accession   (e.g., CL0192 or empty if no clan)
        2: Clan name
        3: Pfam name
        4: Pfam description

    Returns mapping with versioned accessions stripped to root
    (PF00001.23 -> PF00001) for robust matching.
    """
    clan_map = {}

    with open(filepath) as fh:
        for line in fh:
            fields = line.strip().split("\t")
            if len(fields) < 2:
                continue

            pfam_acc = fields[0].split(".")[0]  # strip version
            clan_acc = fields[1].strip()

            if clan_acc:  # only map families that belong to a clan
                clan_map[pfam_acc] = clan_acc

    if not clan_map:
        print("ERROR: no clan mappings found. Check file format.", file=sys.stderr)
        sys.exit(1)

    n_clans = len(set(clan_map.values()))
    print(
        f"Loaded {len(clan_map)} Pfam->clan mappings across {n_clans} clans.",
        file=sys.stderr,
    )

    return clan_map


def overlaps(row_a: pd.Series, row_b: pd.Series, min_overlap: int = 1) -> bool:
    """Check if two domain hits overlap on envelope coordinates by at
    least min_overlap residues."""
    overlap_len = (
        min(row_a["env_to"], row_b["env_to"])
        - max(row_a["env_from"], row_b["env_from"])
        + 1
    )
    return overlap_len >= min_overlap


def resolve_clan_overlaps(
    df: pd.DataFrame,
    clan_map: dict[str, str],
) -> pd.DataFrame:
    """Greedy clan-based overlap resolution per sequence.

    Algorithm:
        1. Assign each hit its clan (or None if the family has no clan).
        2. Per sequence, sort hits by domain score descending (best first).
        3. Accept hits greedily: skip a hit only if it overlaps with an
           already-accepted hit AND both belong to the same clan.
        4. Hits without a clan assignment never conflict with each other
           on clan grounds — they are kept unless they are the weaker
           partner in an overlap with a same-clan hit (which can't happen
           if they have no clan). So effectively, clan-less hits are
           always kept.
    """
    # Map accessions (strip version for matching)
    df = df.copy()
    df["_pfam_root"] = df["hmm_accession"].str.split(".").str[0]
    df["_clan"] = df["_pfam_root"].map(clan_map)

    n_with_clan = df["_clan"].notna().sum()
    n_without = df["_clan"].isna().sum()
    print(
        f"Clan assignment: {n_with_clan} hits in clans, "
        f"{n_without} hits without clan (always kept).",
        file=sys.stderr,
    )

    keep_indices = []

    for seq_name, group in df.groupby("seq_name"):
        # Sort best score first
        group_sorted = group.sort_values("domain_score", ascending=False)

        accepted = []  # list of (index, row) tuples

        for idx, row in group_sorted.iterrows():
            dominated = False
            row_clan = row["_clan"]

            # Hits without a clan can never be dominated on clan grounds
            if pd.notna(row_clan):
                for acc_idx, acc_row in accepted:
                    acc_clan = acc_row["_clan"]
                    if acc_clan == row_clan and overlaps(row, acc_row):
                        dominated = True
                        break

            if not dominated:
                accepted.append((idx, row))

        keep_indices.extend([idx for idx, _ in accepted])

    return df.loc[keep_indices].drop(columns=["_pfam_root", "_clan"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resolve overlapping Pfam domain hits using clan-based filtering. "
            "Within the same clan and sequence, only the best-scoring hit per "
            "overlapping region is kept."
        )
    )
    parser.add_argument("domtblout", help="Pfam hmmscan domtblout file")
    parser.add_argument("clans_tsv", help="Pfam-A.clans.tsv file")
    parser.add_argument(
        "-e", "--evalue", type=float, default=None,
        help="Pre-filter by domain i-Evalue before resolving (default: none)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file in domtblout format (default: stdout)",
    )
    args = parser.parse_args()

    for f in (args.domtblout, args.clans_tsv):
        if not Path(f).exists():
            print(f"Error: file not found: {f}", file=sys.stderr)
            sys.exit(1)

    # Parse inputs
    df = parse_domtblout(Path(args.domtblout))
    clan_map = parse_clans_tsv(Path(args.clans_tsv))
    n_initial = len(df)

    # Optional E-value pre-filter
    if args.evalue is not None:
        df = df[df["domain_ievalue"] <= args.evalue].copy()
        print(
            f"E-value filter: {len(df)}/{n_initial} hits retained "
            f"(cutoff: {args.evalue:.0e}).",
            file=sys.stderr,
        )

    n_before_resolve = len(df)

    # Resolve overlaps
    df = resolve_clan_overlaps(df, clan_map)

    n_removed = n_before_resolve - len(df)
    print(
        f"Overlap resolution: removed {n_removed} dominated hits, "
        f"{len(df)} hits retained.",
        file=sys.stderr,
    )

    # Write output — use the preserved raw lines for faithful domtblout format
    out = open(args.output, "w") if args.output else sys.stdout
    for raw_line in df["_raw_line"]:
        out.write(raw_line + "\n")

    if args.output:
        out.close()
        print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()