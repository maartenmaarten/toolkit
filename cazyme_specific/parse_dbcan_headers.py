#!/usr/bin/env python3
"""
parse_dbcan_headers.py

Filter the dbCAN FASTA (CAZyDB.faa) to NCBI-sourced sequences only — using
the 'source' column in cazy_data.txt — then parse every header into structured
domain annotations and write TSV output.

Usage
-----
python parse_dbcan_headers.py \\
    --fasta   CAZyDB.07142024.faa \\
    --cazy    cazy_data_20250926_uniq_taxonomy_annotated.txt \\
    --outdir  results/

Outputs
-------
<outdir>/dbcan_ncbi_filtered.faa          – FASTA containing only NCBI sequences
<outdir>/dbcan_domain_annotations.tsv     – per-protein domain annotation table

Header format assumed
---------------------
>{protein_id}|{domain1}[|{domain2}|...[|EC_number]]

Examples:
  >QJW36450.1|GH13_30
  >WMT33606.1|CE4|GT2
  >QPV83021.1|CBM35|GH5_41
  >BAA09604.1|CBM35|GH66|2.4.1.248
  >2064252|Phoaln1_GeneCatalog_proteins_20160826.aa.fasta|CBM1   ← non-NCBI, filtered out

Domain classification
---------------------
  Catalytic : GH, GT, PL, CE, AA  (any sub-family, e.g. GH13_30, AA10)
  Non-catalytic : CBM              (e.g. CBM35, CBM0)
  Ignored   : EC numbers (1.2.3.4 / 2.4.1.-), JGI filenames
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    from Bio import SeqIO
except ImportError:
    sys.exit("BioPython is required:  pip install biopython")


# ──────────────────────────────────────────────────────────────────────────────
# Regex constants
# ──────────────────────────────────────────────────────────────────────────────

#: Any catalytic CAZy family (GH / GT / PL / CE / AA), with optional sub-family
CATALYTIC_RE = re.compile(r"^(GH|GT|PL|CE|AA)\d+", re.IGNORECASE)

#: Carbohydrate-Binding Module
CBM_RE = re.compile(r"^CBM\d*", re.IGNORECASE)

#: EC number  e.g. 3.2.1.4  or  2.4.1.-  or  1.14.99.-
EC_RE = re.compile(r"^\d+\.\d+(\.\d+)?(\.\d+|\.-)?(\.-)?\s*$")


# ──────────────────────────────────────────────────────────────────────────────
# Module: load NCBI accession IDs from cazy_data
# ──────────────────────────────────────────────────────────────────────────────

def load_ncbi_ids(cazy_data_path: str | Path) -> set[str]:
    """
    Parse *cazy_data.txt* and return the set of protein accession IDs
    (``genbank_id`` column) whose ``source`` value is ``"ncbi"``.

    The file is expected to be a tab-delimited table with a header row that
    includes at least the columns ``genbank_id`` and ``source``.  Column order
    is detected from the header, so minor schema differences are tolerated.

    Parameters
    ----------
    cazy_data_path:
        Path to the cazy_data TSV file.

    Returns
    -------
    set[str]
        Protein accession IDs (e.g. ``"QJW36450.1"``) from NCBI sources.
    """
    ncbi_ids: set[str] = set()
    path = Path(cazy_data_path)

    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")

        # -- resolve column positions from the header -------------------------
        header = next(reader)
        try:
            id_col  = header.index("genbank_id")
            src_col = header.index("source")
        except ValueError:
            # Fallback: assume fixed positions (col 3 = genbank_id, col 4 = source)
            id_col, src_col = 3, 4
            print(
                f"Warning: 'genbank_id'/'source' columns not found in header; "
                f"falling back to columns {id_col} and {src_col}.",
                file=sys.stderr,
            )

        min_cols = max(id_col, src_col) + 1

        for row in reader:
            if len(row) < min_cols:
                continue
            if row[src_col].strip().lower() == "ncbi":
                acc = row[id_col].strip()
                if acc:
                    ncbi_ids.add(acc)

    return ncbi_ids


# ──────────────────────────────────────────────────────────────────────────────
# Module: parse a single dbCAN FASTA header
# ──────────────────────────────────────────────────────────────────────────────

def parse_dbcan_header(header: str) -> dict:
    """
    Parse one dbCAN FASTA header into structured domain annotation fields.

    The function splits the header on ``|`` and classifies every field after
    the first (protein ID) as one of: catalytic domain, CBM, EC number, or
    'other' (filenames from JGI etc. are silently ignored).

    Parameters
    ----------
    header:
        Raw header line, with or without the leading ``>``.
        Examples::

            >QJW36450.1|GH13_30
            >WMT33606.1|CE4|GT2
            >QPV83021.1|CBM35|GH5_41
            >BAA09604.1|CBM35|GH66|2.4.1.248

    Returns
    -------
    dict
        +---------------------------------+--------------------------------------+
        | Key                             | Description                          |
        +=================================+======================================+
        | ``protein_id``                  | Accession / identifier               |
        +---------------------------------+--------------------------------------+
        | ``raw_annotation``              | Pipe-joined string of all domain     |
        |                                 | fields exactly as in the FASTA       |
        +---------------------------------+--------------------------------------+
        | ``catalytic_domains``           | List of catalytic domains            |
        |                                 | (GH / GT / PL / CE / AA)            |
        +---------------------------------+--------------------------------------+
        | ``cbm_domains``                 | List of CBM annotations              |
        +---------------------------------+--------------------------------------+
        | ``ec_numbers``                  | List of EC numbers (may be empty)    |
        +---------------------------------+--------------------------------------+
        | ``n_domains_with_cbm``          | Total CAZy domain count              |
        |                                 | (catalytic + CBM)                    |
        +---------------------------------+--------------------------------------+
        | ``n_domains_no_cbm``            | Catalytic domain count only          |
        +---------------------------------+--------------------------------------+
        | ``n_cbm``                       | Number of CBM annotations            |
        +---------------------------------+--------------------------------------+
        | ``is_multidomain``              | True if ``n_domains_with_cbm > 1``   |
        +---------------------------------+--------------------------------------+
        | ``is_multidomain_catalytic``    | True if ``n_domains_no_cbm > 1``     |
        +---------------------------------+--------------------------------------+
        | ``domain_class``                | Architecture label; one of:          |
        |                                 | ``single_catalytic``,                |
        |                                 | ``catalytic_with_cbm``,              |
        |                                 | ``multi_catalytic``,                 |
        |                                 | ``multi_catalytic_with_cbm``,        |
        |                                 | ``cbm_only``,                        |
        |                                 | ``unknown``                          |
        +---------------------------------+--------------------------------------+
    """
    header = header.lstrip(">").strip()
    fields = header.split("|")
    protein_id = fields[0]

    catalytic: list[str] = []
    cbms: list[str]      = []
    ec_numbers: list[str] = []

    for field in fields[1:]:
        field = field.strip()
        if not field:
            continue
        if CBM_RE.match(field):
            cbms.append(field)
        elif CATALYTIC_RE.match(field):
            catalytic.append(field)
        elif EC_RE.match(field):
            ec_numbers.append(field)
        # else: JGI filename, version tag, or unrecognised — skip silently

    n_cat   = len(catalytic)
    n_cbm   = len(cbms)
    n_total = n_cat + n_cbm

    # Classify domain architecture
    if   n_cat == 0 and n_cbm == 0:
        domain_class = "unknown"
    elif n_cat == 0 and n_cbm > 0:
        domain_class = "cbm_only"
    elif n_cat == 1 and n_cbm == 0:
        domain_class = "single_catalytic"
    elif n_cat == 1 and n_cbm > 0:
        domain_class = "catalytic_with_cbm"
    elif n_cat > 1 and n_cbm == 0:
        domain_class = "multi_catalytic"
    else:  # n_cat > 1 and n_cbm > 0
        domain_class = "multi_catalytic_with_cbm"

    raw_annotation = "|".join(fields[1:]) if len(fields) > 1 else ""

    return {
        "protein_id":               protein_id,
        "raw_annotation":           raw_annotation,
        "catalytic_domains":        catalytic,
        "cbm_domains":              cbms,
        "ec_numbers":               ec_numbers,
        "n_domains_with_cbm":       n_total,
        "n_domains_no_cbm":         n_cat,
        "n_cbm":                    n_cbm,
        "is_multidomain":           n_total > 1,
        "is_multidomain_catalytic": n_cat > 1,
        "domain_class":             domain_class,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

_TSV_FIELDS = [
    "protein_id",
    "raw_annotation",
    "domain_class",
    "n_domains_with_cbm",
    "n_domains_no_cbm",
    "n_cbm",
    "is_multidomain",
    "is_multidomain_catalytic",
    "catalytic_domains",
    "cbm_domains",
    "ec_numbers",
]


def filter_and_annotate(
    fasta_path: str | Path,
    cazy_data_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Full pipeline:

    1. Load NCBI accession IDs from *cazy_data_path*.
    2. Stream through *fasta_path*; keep only sequences whose accession appears
       in the NCBI ID set.
    3. Write the filtered sequences to ``<output_dir>/dbcan_ncbi_filtered.faa``.
    4. Parse every kept header and write a tab-delimited annotation table to
       ``<output_dir>/dbcan_domain_annotations.tsv``.

    Parameters
    ----------
    fasta_path:
        Path to the full dbCAN FASTA file (e.g. ``CAZyDB.07142024.faa``).
    cazy_data_path:
        Path to the cazy_data TSV file (must contain ``genbank_id`` and
        ``source`` columns).
    output_dir:
        Directory where output files are written (created if absent).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load NCBI IDs ────────────────────────────────────────────────────────
    print(f"[1/3] Loading NCBI accession IDs from:\n      {cazy_data_path}", flush=True)
    ncbi_ids = load_ncbi_ids(cazy_data_path)
    print(f"      → {len(ncbi_ids):,} unique NCBI accessions loaded\n", flush=True)

    # 2. Stream FASTA: filter & parse ─────────────────────────────────────────
    faa_out = out / "dbcan_ncbi_filtered.faa"
    tsv_out = out / "dbcan_domain_annotations.tsv"

    print(f"[2/3] Filtering & annotating:\n      {fasta_path}", flush=True)
    print(f"      Filtered FASTA  → {faa_out}", flush=True)
    print(f"      Annotation TSV  → {tsv_out}\n", flush=True)

    n_total = 0
    n_kept  = 0
    n_skip  = 0

    with (
        open(fasta_path, encoding="utf-8") as fasta_fh,
        open(faa_out, "w", encoding="utf-8") as faa_fh,
        open(tsv_out, "w", encoding="utf-8", newline="") as tsv_fh,
    ):
        writer = csv.DictWriter(tsv_fh, fieldnames=_TSV_FIELDS, delimiter="\t")
        writer.writeheader()

        for record in SeqIO.parse(fasta_fh, "fasta"):
            n_total += 1

            # record.id  = everything up to the first space (or end of line)
            # In dbCAN there are no spaces, so record.id == full description
            protein_id = record.id.split("|")[0]

            if protein_id not in ncbi_ids:
                n_skip += 1
                continue

            n_kept += 1

            # Write filtered FASTA record
            faa_fh.write(f">{record.description}\n{record.seq}\n")

            # Parse the header
            ann = parse_dbcan_header(record.description)

            # Serialise list fields as semicolon-separated strings
            writer.writerow({
                "protein_id":               ann["protein_id"],
                "raw_annotation":           ann["raw_annotation"],
                "domain_class":             ann["domain_class"],
                "n_domains_with_cbm":       ann["n_domains_with_cbm"],
                "n_domains_no_cbm":         ann["n_domains_no_cbm"],
                "n_cbm":                    ann["n_cbm"],
                "is_multidomain":           ann["is_multidomain"],
                "is_multidomain_catalytic": ann["is_multidomain_catalytic"],
                "catalytic_domains":        ";".join(ann["catalytic_domains"]),
                "cbm_domains":              ";".join(ann["cbm_domains"]),
                "ec_numbers":               ";".join(ann["ec_numbers"]),
            })

            if n_total % 500_000 == 0:
                print(
                    f"      … {n_total:,} sequences scanned | "
                    f"{n_kept:,} kept | {n_skip:,} excluded",
                    flush=True,
                )

    # 3. Summary ──────────────────────────────────────────────────────────────
    pct = n_kept / n_total * 100 if n_total else 0.0
    print(
        f"[3/3] Complete.\n"
        f"      Sequences in FASTA          : {n_total:>12,}\n"
        f"      NCBI sequences kept         : {n_kept:>12,}  ({pct:.1f} %)\n"
        f"      Non-NCBI sequences excluded : {n_skip:>12,}\n",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--fasta", required=True,
        metavar="FAA",
        help="Path to the full dbCAN FASTA file (e.g. CAZyDB.07142024.faa)",
    )
    p.add_argument(
        "--cazy", required=True,
        metavar="TSV",
        help=(
            "Path to cazy_data TSV with 'genbank_id' and 'source' columns "
            "(e.g. cazy_data_20250926_uniq_taxonomy_annotated.txt)"
        ),
    )
    p.add_argument(
        "--outdir", default="dbcan_parsed_output",
        metavar="DIR",
        help="Output directory (default: dbcan_parsed_output/)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    filter_and_annotate(
        fasta_path=args.fasta,
        cazy_data_path=args.cazy,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
