#!/usr/bin/env python3
"""
Convert a Stockholm alignment (e.g. from hmmalign or Pfam) to aligned FASTA.

The "." gaps of Stockholm become "-" and every residue is written in
uppercase, so hmmalign's lowercase insert-state residues look like any other
residue; --keep-case keeps lowercase and "." as they are. Stockholm
annotation lines (#=GF, #=GS, #=GR, #=GC) are dropped, as FASTA has no
place for them.

Usage:
    python stockholm_to_afa.py alignment.sto                 # writes alignment.afa
    python stockholm_to_afa.py alignment.sto -o out.afa
    python stockholm_to_afa.py alignment.sto --keep-case
"""
import argparse
from pathlib import Path

from Bio import AlignIO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stockholm", help="Stockholm alignment (one alignment per file)")
    parser.add_argument("-o", "--output", default=None, help="Aligned FASTA (default: input with .afa extension)")
    parser.add_argument("--keep-case", action="store_true",
                        help="Keep lowercase residues and '.' gaps instead of uppercase and '-'")
    return parser.parse_args()


def main():
    args = parse_args()
    alignment = AlignIO.read(args.stockholm, "stockholm")
    out = Path(args.output) if args.output else Path(args.stockholm).with_suffix(".afa")
    with open(out, "w") as fh:
        for rec in alignment:
            seq = str(rec.seq) if args.keep_case else str(rec.seq).upper().replace(".", "-")
            fh.write(f">{rec.id}\n{seq}\n")
    print(f"Written {out} ({len(alignment)} sequences, {alignment.get_alignment_length()} columns)")


if __name__ == "__main__":
    main()
