#!/usr/bin/env python3
"""
parse_domtblout.py — Parse HMMER domtblout, resolve overlapping domain hits,
and output a clean per-protein domain architecture table.

Overlap resolution: greedy algorithm, sorted by i-Evalue (ascending).
Two hits are considered overlapping if they share more than `--overlap-threshold`
fraction of the shorter hit's length.

Usage:
    python parse_domtblout.py input.domtblout -o results.tsv
    python parse_domtblout.py input.domtblout -o results.tsv --evalue 1e-5 --overlap-threshold 0.3
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DomainHit:
    """A single domain hit from HMMER domtblout."""
    domain_name: str
    domain_acc: str
    protein_name: str
    protein_len: int
    domain_num: int
    domain_of: int
    i_evalue: float
    domain_score: float
    hmm_from: int
    hmm_to: int
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int
    acc: float
    description: str
    raw_line: str = field(default="")

    @property
    def env_len(self) -> int:
        return self.env_to - self.env_from + 1


def parse_domtblout(filepath: str):
    """Parse a HMMER domtblout file into a list of DomainHit objects.

    The domtblout format is space-delimited with 22 fixed columns,
    followed by a free-text description field. We split on whitespace
    and rejoin everything from column 22 onward as the description.

    Returns (header_lines, hits) where header_lines preserves the original
    comment/header block for writing to the resolved output.
    """
    hits = []
    header_lines = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line)
                continue
            raw_line = line
            line = line.strip()
            if not line:
                continue

            cols = line.split()
            if len(cols) < 22:
                print(f"WARNING: skipping malformed line: {line[:80]}...",
                      file=sys.stderr)
                continue

            try:
                hit = DomainHit(
                    domain_name=cols[0],
                    domain_acc=cols[1],
                    protein_name=cols[3],
                    protein_len=int(cols[5]),
                    domain_num=int(cols[9]),
                    domain_of=int(cols[10]),
                    i_evalue=float(cols[12]),
                    domain_score=float(cols[13]),
                    hmm_from=int(cols[15]),
                    hmm_to=int(cols[16]),
                    ali_from=int(cols[17]),
                    ali_to=int(cols[18]),
                    env_from=int(cols[19]),
                    env_to=int(cols[20]),
                    acc=float(cols[21]),
                    description=" ".join(cols[22:]),
                    raw_line=raw_line,
                )
                hits.append(hit)
            except (ValueError, IndexError) as e:
                print(f"WARNING: could not parse line: {e}\n  {line[:80]}...",
                      file=sys.stderr)
    return header_lines, hits


def overlap_fraction(a: DomainHit, b: DomainHit) -> float:
    """Calculate overlap as a fraction of the shorter hit's envelope length.

    Returns 0.0 if no overlap, up to 1.0 if one is fully contained in the other.
    """
    overlap_start = max(a.env_from, b.env_from)
    overlap_end = min(a.env_to, b.env_to)
    overlap_len = max(0, overlap_end - overlap_start + 1)
    shorter = min(a.env_len, b.env_len)
    if shorter == 0:
        return 0.0
    return overlap_len / shorter


def resolve_overlaps(
    hits: List[DomainHit],
    threshold: float = 0.5,
) -> List[DomainHit]:
    """Greedy overlap resolution for a single protein's hits.

    1. Sort hits by i-Evalue (ascending = best first).
    2. Accept best hit, reject anything overlapping with it above threshold.
    3. Repeat until no hits remain.

    Returns accepted hits sorted by env_from (positional order).
    """
    remaining = sorted(hits, key=lambda h: h.i_evalue)
    accepted = []

    while remaining:
        best = remaining.pop(0)
        accepted.append(best)
        remaining = [
            h for h in remaining
            if overlap_fraction(best, h) < threshold
        ]

    return sorted(accepted, key=lambda h: h.env_from)


def main():
    parser = argparse.ArgumentParser(
        description="Parse HMMER domtblout and resolve overlapping domain hits."
    )
    parser.add_argument(
        "input",
        help="Path to HMMER domtblout file",
    )
    parser.add_argument(
        "-o", "--output",
        default="-",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "-e", "--evalue",
        type=float,
        default=1e-5,
        help="i-Evalue threshold for inclusion (default: 1e-5)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum overlap fraction (of shorter hit) to consider two hits "
            "as conflicting (default: 0.5). Set to 0.0 to disallow any overlap."
        ),
    )
    args = parser.parse_args()

    # Parse
    _, all_hits = parse_domtblout(args.input)
    print(f"Parsed {len(all_hits)} domain hits", file=sys.stderr)

    # Filter by e-value
    filtered = [h for h in all_hits if h.i_evalue <= args.evalue]
    print(
        f"After i-Evalue filter (<= {args.evalue}): {len(filtered)} hits",
        file=sys.stderr,
    )

    # Group by protein
    proteins: Dict[str, List[DomainHit]] = {}
    for hit in filtered:
        proteins.setdefault(hit.protein_name, []).append(hit)

    # Resolve overlaps per protein
    resolved: Dict[str, List[DomainHit]] = {}
    total_kept = 0
    for protein, hits in proteins.items():
        resolved[protein] = resolve_overlaps(hits, args.overlap_threshold)
        total_kept += len(resolved[protein])

    print(
        f"After overlap resolution: {total_kept} hits across "
        f"{len(resolved)} proteins",
        file=sys.stderr,
    )

    # Output — tab-separated table
    tsv_header = "\t".join([
        "protein_name", "protein_len",
        "domain_name", "domain_acc",
        "domain_num", "domain_of",
        "i_evalue", "domain_score",
        "hmm_from", "hmm_to",
        "ali_from", "ali_to",
        "env_from", "env_to",
        "acc", "description",
    ])
    out = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        out.write(tsv_header + "\n")
        for protein in sorted(resolved.keys()):
            for h in resolved[protein]:
                row = "\t".join([
                    h.protein_name, str(h.protein_len),
                    h.domain_name, h.domain_acc,
                    str(h.domain_num), str(h.domain_of),
                    str(h.i_evalue), str(h.domain_score),
                    str(h.hmm_from), str(h.hmm_to),
                    str(h.ali_from), str(h.ali_to),
                    str(h.env_from), str(h.env_to),
                    str(h.acc), h.description,
                ])
                out.write(row + "\n")
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()