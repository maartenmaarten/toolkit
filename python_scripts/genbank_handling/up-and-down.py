#!/usr/bin/env python3
"""
Usage: python up-and-down.py <input.gbk/.gbff or a folder of them> <protein_ids.txt> <output_dir> [--flank N]
       python up-and-down.py <input.gbk/.gbff or a folder of them> ID1,ID2,ID3 <output_dir> [--flank N]
       cat protein_ids.txt | python up-and-down.py <input.gbk/.gbff or a folder of them> - <output_dir> [--flank N]

For each protein_id in the list, locates the gene encoding it and extracts the
genomic region spanning that gene plus N genes upstream and N genes downstream
(ordered by genomic coordinate, not transcription direction), keeping all
intergenic sequence in between. Warns per protein_id if fewer than N genes are
available on either side (record boundary reached).

Writes one GenBank file per protein_id to <output_dir>, containing the sliced
DNA sequence and its features (CDS entries retain their /translation, i.e. the
amino acid record, along with genes, RNAs, etc.).
"""

import argparse
import os
import re
import sys

from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation

FLANKING_FEATURE_TYPES = ("CDS", "tRNA", "rRNA", "ncRNA", "tmRNA")


def read_protein_ids(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


GENBANK_EXTENSIONS = (".gbk", ".gbff", ".gb", ".genbank")


def collect_genbank_files(path):
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(GENBANK_EXTENSIONS)
        )
        if not files:
            print(f"No {'/'.join(GENBANK_EXTENSIONS)} files found in directory {path}", file=sys.stderr)
            sys.exit(1)
        return files
    return [path]


def resolve_protein_ids(args_list):
    """args_list is argparse nargs='+': a single '-' to read IDs (one per line)
    from stdin, a single path to a file of IDs, or one or more IDs given
    directly (space- and/or comma-separated)."""
    if len(args_list) == 1 and args_list[0] == "-":
        return [line.strip() for line in sys.stdin if line.strip() and not line.startswith("#")]

    if len(args_list) == 1 and os.path.isfile(args_list[0]):
        return read_protein_ids(args_list[0])

    ids = []
    for item in args_list:
        ids.extend(part.strip() for part in item.split(",") if part.strip())
    return ids


def build_gene_lists(records):
    """One sorted 'gene' list per record: proper gene features if present,
    otherwise CDS/RNA features used as gene stand-ins."""
    per_record_genes = []
    for record in records:
        gene_feats = [f for f in record.features if f.type == "gene"]
        if gene_feats:
            genes = gene_feats
        else:
            genes = [f for f in record.features if f.type in FLANKING_FEATURE_TYPES]
        per_record_genes.append(sorted(genes, key=lambda f: int(f.location.start)))
    return per_record_genes


def locate_gene(genes, cds_feature):
    locus_tag = cds_feature.qualifiers.get("locus_tag", [None])[0]
    cds_start = int(cds_feature.location.start)
    cds_end = int(cds_feature.location.end)

    if locus_tag:
        for i, g in enumerate(genes):
            if g.qualifiers.get("locus_tag", [None])[0] == locus_tag:
                return i

    for i, g in enumerate(genes):
        if int(g.location.start) == cds_start and int(g.location.end) == cds_end:
            return i

    for i, g in enumerate(genes):
        if int(g.location.start) < cds_end and int(g.location.end) > cds_start:
            return i

    return None


def index_proteins(records, per_record_genes):
    """protein_id -> (record_index, gene_index)"""
    protein_locations = {}
    for r_idx, record in enumerate(records):
        genes = per_record_genes[r_idx]
        for feat in record.features:
            if feat.type != "CDS":
                continue
            pid = feat.qualifiers.get("protein_id", [None])[0]
            if not pid or pid in protein_locations:
                continue
            gene_idx = locate_gene(genes, feat)
            if gene_idx is not None:
                protein_locations[pid] = (r_idx, gene_idx)
    return protein_locations


def extract_flanks(record, genes, gene_idx, flank, protein_id):
    upstream = genes[max(0, gene_idx - flank):gene_idx]
    downstream = genes[gene_idx + 1: gene_idx + 1 + flank]

    if len(upstream) < flank:
        print(
            f"WARNING: {protein_id}: only {len(upstream)} upstream gene(s) available "
            f"(requested {flank}); start of record '{record.id}' reached.",
            file=sys.stderr,
        )
    if len(downstream) < flank:
        print(
            f"WARNING: {protein_id}: only {len(downstream)} downstream gene(s) available "
            f"(requested {flank}); end of record '{record.id}' reached.",
            file=sys.stderr,
        )

    selected = upstream + [genes[gene_idx]] + downstream
    start = min(int(f.location.start) for f in selected)
    end = max(int(f.location.end) for f in selected)
    return start, end, len(upstream), len(downstream)


def build_output_record(record, start, end, protein_id, n_up, n_down, flank):
    sub = record[start:end]
    sub.annotations["molecule_type"] = record.annotations.get("molecule_type", "DNA")
    sub.annotations["topology"] = "linear"
    for key in ("organism", "source", "taxonomy"):
        if key in record.annotations:
            sub.annotations[key] = record.annotations[key]

    safe_locus = re.sub(r"[^A-Za-z0-9_.-]", "_", protein_id)[:16] or "region"
    sub.id = safe_locus
    sub.name = safe_locus
    sub.description = (
        f"{record.id}:{start + 1}-{end} region around {protein_id} "
        f"({n_up} upstream / {n_down} downstream of {flank} requested genes)"
    )

    orig_source = next((f for f in record.features if f.type == "source"), None)
    source_qualifiers = dict(orig_source.qualifiers) if orig_source else {}
    sub.features.insert(
        0, SeqFeature(FeatureLocation(0, len(sub)), type="source", qualifiers=source_qualifiers)
    )
    return sub


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "genbank_file", help="Input .gbk/.gbff file, or a folder containing multiple such files"
    )
    parser.add_argument(
        "protein_ids",
        nargs="+",
        help="Protein ID(s) to extract (space- and/or comma-separated), "
        "a single path to a file with one protein_id per line, "
        "or '-' to read one protein_id per line from stdin",
    )
    parser.add_argument("output_dir", help="Directory to write one GenBank file per protein_id")
    parser.add_argument("--flank", type=int, default=10, help="Genes to include on each side (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = collect_genbank_files(args.genbank_file)
    records = []
    for fp in input_files:
        records.extend(SeqIO.parse(fp, "genbank"))
    if not records:
        print(f"No records found in {args.genbank_file}", file=sys.stderr)
        sys.exit(1)

    per_record_genes = build_gene_lists(records)
    protein_locations = index_proteins(records, per_record_genes)

    protein_ids = resolve_protein_ids(args.protein_ids)
    if not protein_ids:
        print("No protein IDs found in input list.", file=sys.stderr)
        sys.exit(1)

    written = 0
    for protein_id in protein_ids:
        loc = protein_locations.get(protein_id)
        if loc is None:
            print(f"WARNING: protein_id '{protein_id}' not found; skipping.", file=sys.stderr)
            continue

        r_idx, gene_idx = loc
        record = records[r_idx]
        genes = per_record_genes[r_idx]

        start, end, n_up, n_down = extract_flanks(record, genes, gene_idx, args.flank, protein_id)
        sub = build_output_record(record, start, end, protein_id, n_up, n_down, args.flank)

        safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", protein_id)
        out_path = os.path.join(args.output_dir, f"{safe_filename}.gbk")
        with open(out_path, "w") as out_fh:
            SeqIO.write(sub, out_fh, "genbank")
        written += 1
        print(f"{protein_id}: wrote {out_path} ({end - start} bp, {n_up} upstream / {n_down} downstream genes)")

    print(f"\nDone: {written}/{len(protein_ids)} protein IDs extracted.", file=sys.stderr)


if __name__ == "__main__":
    main()
