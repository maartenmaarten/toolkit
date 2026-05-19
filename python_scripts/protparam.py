#!/usr/bin/env python3
"""
Compute BioPython ProtParam properties for every sequence in a FASTA file
and write the results as a TSV alongside the input file.

Output: <input_stem>.protparam.tsv  (written in the same directory as the FASTA)

Usage:
    python protparam.py proteins.fasta [--pH 7.0]

Columns written
---------------
sequence_id, length, molecular_weight, isoelectric_point, instability_index,
aromaticity, gravy, helix_fraction, turn_fraction, sheet_fraction,
molar_ext_reduced, molar_ext_oxidized, charge_at_pH
"""

import argparse
import csv
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMBIGUOUS = set("XBZJOU*")

COLUMNS = [
    "sequence_id",
    "length",
    "molecular_weight",
    "isoelectric_point",
    "instability_index",
    "aromaticity",
    "gravy",
    "helix_fraction",
    "turn_fraction",
    "sheet_fraction",
    "molar_ext_reduced",
    "molar_ext_oxidized",
    "charge_at_pH",
]


def analyse(record, pH):
    """Return a dict of ProtParam values for one SeqRecord, or NaN on failure."""
    seq = str(record.seq).upper().replace("*", "")
    base = {
        "sequence_id": record.id,
        "length": len(seq),
    }
    nan = {c: "NA" for c in COLUMNS if c not in base}

    if not seq:
        return {**base, **nan}

    # Warn but continue if ambiguous residues are present; ProtParam will fail
    # on some calculations in that case so we catch per-property.
    has_ambiguous = bool(set(seq) & AMBIGUOUS)
    if has_ambiguous:
        print(
            f"  Warning: {record.id} contains ambiguous residues "
            f"({set(seq) & AMBIGUOUS}) — some properties may be NA",
            file=sys.stderr,
        )
        # Strip ambiguous residues for calculations that require clean sequence
        clean = "".join(aa for aa in seq if aa not in AMBIGUOUS)
    else:
        clean = seq

    def safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return "NA"

    pa_full = ProteinAnalysis(seq)
    pa_clean = ProteinAnalysis(clean) if has_ambiguous else pa_full

    ss = safe(pa_clean.secondary_structure_fraction)
    helix = ss[0] if ss != "NA" else "NA"
    turn  = ss[1] if ss != "NA" else "NA"
    sheet = ss[2] if ss != "NA" else "NA"

    ext = safe(pa_clean.molar_extinction_coefficient)
    ext_red = ext[0] if ext != "NA" else "NA"
    ext_ox  = ext[1] if ext != "NA" else "NA"

    return {
        "sequence_id":       record.id,
        "length":            len(seq),
        "molecular_weight":  safe(pa_clean.molecular_weight),
        "isoelectric_point": safe(pa_clean.isoelectric_point),
        "instability_index": safe(pa_clean.instability_index),
        "aromaticity":       safe(pa_clean.aromaticity),
        "gravy":             safe(pa_clean.gravy),
        "helix_fraction":    helix,
        "turn_fraction":     turn,
        "sheet_fraction":    sheet,
        "molar_ext_reduced": ext_red,
        "molar_ext_oxidized": ext_ox,
        "charge_at_pH":      safe(pa_clean.charge_at_pH, pH),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute ProtParam properties for all sequences in a FASTA file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fasta", help="Input protein FASTA file")
    parser.add_argument(
        "--pH", type=float, default=7.0,
        help="pH for charge_at_pH calculation"
    )
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"Error: file not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)

    out_path = fasta_path.parent / (fasta_path.stem + ".protparam.tsv")

    records = list(SeqIO.parse(fasta_path, "fasta"))
    print(f"Loaded {len(records)} sequences from {fasta_path}")

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for i, rec in enumerate(records, 1):
            row = analyse(rec, args.pH)
            writer.writerow(row)
            if i % 500 == 0:
                print(f"  Processed {i}/{len(records)}...")

    print(f"Done. Written to {out_path}")


if __name__ == "__main__":
    main()
