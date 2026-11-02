#!/usr/bin/env python3
"""
Combine DeepLocPro (prokaryotes) and DeepLoc 2.x (eukaryotes) predictions into
one iTOL DATASET_COLORSTRIP band with the predicted subcellular location.

Which prediction a protein gets is decided by the taxonomic domain in an
annotation TSV (default columns: GenBank, Domain): proteins whose domain is
Eukaryota take the DeepLoc 2.x call, all others (Bacteria, Archaea,
unclassified metagenome proteins, ...) the DeepLocPro call.

Labels are the tools' own location names, except that all cytoplasmic calls are
united into one "Cytoplasm" category: DeepLocPro's "Cytoplasmic" and every
DeepLoc 2.x multi-location call that includes the cytoplasm ("Cytoplasm|Nucleus").
Other multi-location calls are kept as a combined label ("Nucleus + Extracellular").
"Cytoplasmic Membrane" is a membrane location and stays separate.

Usage:
    python itol_from_deeploc.py --deeplocpro results-deeplocpro.csv \\
        --deeploc results-deeploc2.1.csv --annotations characterized_ALL.tsv \\
        --out itol_localization.txt
"""

import csv
import re
import sys
from typing import Optional

from itol_from_tsv import _match_col, assign_colors, write_colorstrip

ID_ALIASES  = {"acc", "protein_id", "id", "seq_id", "sequence_id", "protein", "accession"}
LOC_ALIASES = {"localization", "localizations"}

EUKARYOTE = "eukaryota"

CYTOPLASM = "Cytoplasm"
CYTOPLASM_NAMES = {"cytoplasm", "cytoplasmic"}


def _location_label(call: str) -> str:
    parts = [loc.strip() for loc in call.split("|") if loc.strip()]
    if any(p.lower() in CYTOPLASM_NAMES for p in parts):
        return CYTOPLASM
    return " + ".join(parts)


def read_deeploc(path: str) -> dict:
    """seq ID -> location label, from DeepLocPro or DeepLoc 2.x CSV output."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        col_id  = _match_col(headers, ID_ALIASES)
        col_loc = _match_col(headers, LOC_ALIASES)
        if col_id is None or col_loc is None:
            sys.exit(
                f"ERROR: could not find ID/localization columns in {path}.\n"
                f"Available columns: {headers}"
            )
        calls = {}
        for row in reader:
            seq_id = row[col_id].strip().split()[0] if row[col_id].strip() else ""
            if seq_id:
                calls[seq_id] = _location_label(row[col_loc])
    return calls


def read_domains(path: str, id_col: str, domain_col: str) -> dict:
    """seq ID -> set of taxonomic domains (lowercased) from the annotation TSV."""
    domains = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        headers = reader.fieldnames or []
        missing = [c for c in (id_col, domain_col) if c not in headers]
        if missing:
            sys.exit(f"ERROR: column(s) not found in {path}: {missing}. Available: {headers}")
        for row in reader:
            domain = (row[domain_col] or "").strip().lower()
            for seq_id in re.split(r"[\s,;|]+", row[id_col] or ""):
                if seq_id:
                    domains.setdefault(seq_id, set()).add(domain)
    return domains


def combine(
    pro_path: Optional[str],
    euk_path: Optional[str],
    annotations_path: str,
    out_path: str,
    id_col: str = "GenBank",
    domain_col: str = "Domain",
    dataset_label: str = "Localization",
):
    if not pro_path and not euk_path:
        sys.exit("ERROR: give at least one of --deeplocpro, --deeploc")

    pro = read_deeploc(pro_path) if pro_path else {}
    euk = read_deeploc(euk_path) if euk_path else {}
    domains = read_domains(annotations_path, id_col, domain_col)

    id_to_value = {}
    no_domain, conflicting, no_call = set(), set(), set()
    n_pro = n_euk = 0
    for seq_id in sorted(set(pro) | set(euk)):
        doms = domains.get(seq_id)
        if not doms:
            no_domain.add(seq_id)
            continue
        if len(doms) > 1 and EUKARYOTE in doms:
            conflicting.add(seq_id)
            continue
        is_euk = EUKARYOTE in doms
        calls = euk if is_euk else pro
        if seq_id not in calls:
            no_call.add(seq_id)
            continue
        id_to_value[seq_id] = calls[seq_id]
        n_euk += is_euk
        n_pro += not is_euk

    for msg, ids in [
        ("protein(s) not in the annotation TSV (skipped)", no_domain),
        ("protein(s) annotated as both eukaryote and non-eukaryote (skipped)", conflicting),
        ("protein(s) missing from the DeepLoc file for their domain (skipped)", no_call),
    ]:
        if ids:
            print(
                f"  WARNING: {len(ids)} {msg}: {', '.join(sorted(ids)[:5])}"
                + (" ..." if len(ids) > 5 else "")
            )
    if not id_to_value:
        sys.exit("ERROR: no proteins left to write")

    print(f"  {n_pro} proteins from DeepLocPro, {n_euk} from DeepLoc (Eukaryota)")
    colors = assign_colors(list(id_to_value.values()))
    write_colorstrip(out_path, dataset_label, id_to_value, colors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepLocPro + DeepLoc 2.x → one iTOL DATASET_COLORSTRIP localization band"
    )
    parser.add_argument("--deeplocpro", help="DeepLocPro results CSV (prokaryotes)")
    parser.add_argument("--deeploc", help="DeepLoc 2.x results CSV (eukaryotes)")
    parser.add_argument("--annotations", required=True,
                        help="Annotation TSV with seq ID and taxonomic domain columns")
    parser.add_argument("--id-col", default="GenBank",
                        help="Seq ID column in the annotation TSV (default: GenBank)")
    parser.add_argument("--domain-col", default="Domain",
                        help="Taxonomic domain column in the annotation TSV (default: Domain)")
    parser.add_argument("--out", required=True, help="Output iTOL annotation file")
    parser.add_argument("--dataset-label", default="Localization", help="iTOL dataset label")
    args = parser.parse_args()

    combine(args.deeplocpro, args.deeploc, args.annotations, args.out,
            args.id_col, args.domain_col, args.dataset_label)
