import argparse
import os
import re
from collections import defaultdict

from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a CAZyme FASTA file into per-subfamily FASTAs based on a CAZy TSV."
    )
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("tsv", help="CAZy subfamily TSV (columns: Family, Domain, Species, GenBank, Source)")
    parser.add_argument("-o", "--outdir", default=None,
                        help="Output directory (default: <fasta_dir>/subfamilies)")
    return parser.parse_args()


def build_subfamily_map(tsv_path):
    acc_to_subfams = defaultdict(set)
    genbank_pattern = re.compile(r"[A-Za-z]")
    with open(tsv_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            family, genbank = parts[0], parts[3]
            if not genbank_pattern.search(genbank):
                continue
            acc_to_subfams[genbank].add(family)
    return acc_to_subfams


def split_fasta(fasta_path, acc_to_subfams, outdir):
    os.makedirs(outdir, exist_ok=True)

    subfam_records = defaultdict(list)
    unmatched = 0

    for record in SeqIO.parse(fasta_path, "fasta"):
        subfams = acc_to_subfams.get(record.id)
        if not subfams:
            unmatched += 1
            continue
        for subfam in subfams:
            subfam_records[subfam].append(record)

    for subfam, records in subfam_records.items():
        outpath = os.path.join(outdir, f"{subfam}.fa")
        SeqIO.write(records, outpath, "fasta")

    return subfam_records, unmatched


def subfam_sort_key(name):
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return (1, int(parts[-1]))
    return (0, 0)


def main():
    args = parse_args()
    outdir = args.outdir or os.path.join(os.path.dirname(args.fasta), "subfamilies")

    acc_to_subfams = build_subfamily_map(args.tsv)
    print(f"TSV: {len(acc_to_subfams)} unique GenBank accessions mapped")

    subfam_records, unmatched = split_fasta(args.fasta, acc_to_subfams, outdir)

    print(f"Unmatched FASTA entries: {unmatched}")
    print(f"\nSubfamily files written to {outdir}:")
    for subfam in sorted(subfam_records.keys(), key=subfam_sort_key):
        print(f"  {subfam}.fa: {len(subfam_records[subfam])} sequences")


if __name__ == "__main__":
    main()
