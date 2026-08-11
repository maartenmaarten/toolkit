#!/usr/bin/env python3
"""
Convert domain annotations to iTOL DATASET_DOMAINS format.

Accepts two input formats (auto-detected):
  1. Raw HMMER domtblout  — space-delimited, lines starting with '#' are skipped.
     Columns used: query_name (seq ID), target_name (HMM, .hmm suffix stripped),
                   env_from, env_to, fs_evalue (full-sequence e-value).
  2. Plain TSV/CSV with a header row — column names are auto-detected:
       seq ID  : id, seq_id, sequence_id, protein_id, protein, accession
       domain  : domain, domain_name, name, label, hmm_name, family
       start   : start, start_pos, env_from, ali_from, begin
       end     : end, end_pos, env_to, ali_to, stop

Protein lengths are always derived from the input FASTA.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
import seaborn as sns


# ── column alias maps (plain TSV mode) ────────────────────────────────────────

ID_ALIASES     = {"id", "seq_id", "sequence_id", "protein_id", "protein", "protein_name", "accession"}
DOMAIN_ALIASES = {"domain", "domain_name", "name", "label", "hmm_name", "family"}
START_ALIASES  = {"start", "start_pos", "env_from", "ali_from", "begin"}
END_ALIASES    = {"end",   "end_pos",   "env_to",   "ali_to",   "stop"}

SHAPE   = "RE"
PALETTE = sns.color_palette("Paired").as_hex()

# HMMER domtblout fixed column positions (0-based)
_DOM_COLS = {
    "target_name":      0,
    "target_accession": 1,
    "tlen":             2,
    "query_name":       3,
    "query_accession":  4,
    "qlen":             5,
    "fs_evalue":        6,
    "fs_score":         7,
    "fs_bias":          8,
    "dom_num":          9,
    "dom_total":        10,
    "c_evalue":         11,
    "i_evalue":         12,
    "dom_score":        13,
    "dom_bias":         14,
    "hmm_from":         15,
    "hmm_to":           16,
    "ali_from":         17,
    "ali_to":           18,
    "env_from":         19,
    "env_to":           20,
    "acc":              21,
    # field 22 onward = description (free text)
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _match_col(headers: list[str], aliases: set[str]) -> Optional[str]:
    for h in headers:
        if h.strip().lower() in {a.lower() for a in aliases}:
            return h
    return None


def read_fasta_lengths(fasta_path: str) -> dict:
    lengths = {}
    current_id = None
    current_len = 0
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    lengths[current_id] = current_len
                current_id = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
    if current_id is not None:
        lengths[current_id] = current_len
    return lengths


def assign_colors(domain_names: list) -> dict:
    unique = sorted(set(domain_names))
    return {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(unique)}


def _is_domtblout(path: str) -> bool:
    """Return True if the file looks like raw HMMER domtblout output."""
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                return True
            # first non-empty line: domtblout rows have ≥23 space-delimited fields
            if line.strip():
                return len(line.split()) >= 23
    return False


# ── parsers ────────────────────────────────────────────────────────────────────

def _parse_domtblout(path: str, evalue_threshold: float) -> list:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(None, 22)
            if len(parts) < 21:
                continue
            try:
                fs_evalue = float(parts[_DOM_COLS["fs_evalue"]])
            except ValueError:
                continue
            if fs_evalue > evalue_threshold:
                continue
            domain = parts[_DOM_COLS["target_name"]].replace(".hmm", "")
            rows.append({
                "seq_id": parts[_DOM_COLS["query_name"]],
                "domain": domain,
                "start":  int(parts[_DOM_COLS["env_from"]]),
                "end":    int(parts[_DOM_COLS["env_to"]]),
            })
    return rows


def _parse_tsv(path: str) -> list:
    with open(path) as fh:
        sample = fh.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters="\t,")

    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        headers = reader.fieldnames or []
        if not headers:
            sys.exit("ERROR: domains file appears to be empty or has no header row")

        col_id     = _match_col(headers, ID_ALIASES)
        col_domain = _match_col(headers, DOMAIN_ALIASES)
        col_start  = _match_col(headers, START_ALIASES)
        col_end    = _match_col(headers, END_ALIASES)

        missing = [
            name
            for name, col in [
                ("sequence-ID", col_id),
                ("domain-name", col_domain),
                ("start",       col_start),
                ("end",         col_end),
            ]
            if col is None
        ]
        if missing:
            sys.exit(
                f"ERROR: could not auto-detect column(s) for: {', '.join(missing)}\n"
                f"Available columns: {headers}\n"
                "Rename headers to match recognised aliases (see tsv_to_itol.py)."
            )

        for row in reader:
            rows.append({
                "seq_id": row[col_id].strip(),
                "domain": row[col_domain].strip().replace(".hmm", ""),
                "start":  int(row[col_start]),
                "end":    int(row[col_end]),
            })
    return rows


# ── main conversion ────────────────────────────────────────────────────────────

def convert_domains(
    fasta_path: str,
    domains_path: str,
    out_path: str,
    evalue_threshold: float = 1e-5,
):
    lengths = read_fasta_lengths(fasta_path)

    if _is_domtblout(domains_path):
        print(f"  Detected HMMER domtblout format (e-value threshold: {evalue_threshold})")
        rows = _parse_domtblout(domains_path, evalue_threshold)
    else:
        print("  Detected plain TSV/CSV format")
        rows = _parse_tsv(domains_path)

    if not rows:
        sys.exit("ERROR: no domain entries found after parsing/filtering")

    all_domains = [r["domain"] for r in rows]
    colors = assign_colors(all_domains)

    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq_id"]].append(r)

    unknown = set(by_seq) - set(lengths)
    if unknown:
        print(
            f"  WARNING: {len(unknown)} sequence ID(s) in domain file not found in FASTA "
            f"(skipped): {', '.join(sorted(unknown)[:5])}"
            + (" ..." if len(unknown) > 5 else "")
        )

    lines: list[str] = []
    lines.append("DATASET_DOMAINS")
    lines.append("SEPARATOR TAB")
    lines.append("DATASET_LABEL\tDomains")
    lines.append("COLOR\t#000000")
    lines.append("")
    lines.append("LEGEND_TITLE\tDomain families")
    legend_shapes = "\t".join(SHAPE for _ in sorted(colors))
    legend_colors = "\t".join(colors[d] for d in sorted(colors))
    legend_labels = "\t".join(sorted(colors))
    lines.append(f"LEGEND_SHAPES\t{legend_shapes}")
    lines.append(f"LEGEND_COLORS\t{legend_colors}")
    lines.append(f"LEGEND_LABELS\t{legend_labels}")
    lines.append("")
    lines.append("DATA")

    for seq_id, domains in sorted(by_seq.items()):
        if seq_id not in lengths:
            continue
        seq_len = lengths[seq_id]
        domain_fields = "\t".join(
            f"{SHAPE}|{d['start']}|{d['end']}|{colors[d['domain']]}|{d['domain']}"
            for d in sorted(domains, key=lambda x: x["start"])
        )
        lines.append(f"{seq_id}\t{seq_len}\t{domain_fields}")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(
        f"  Wrote {len(by_seq) - len(unknown)} sequences, "
        f"{len(all_domains)} domain entries, "
        f"{len(colors)} unique domain families"
    )


# ── CLI shim ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Domain annotations → iTOL DATASET_DOMAINS")
    parser.add_argument("--fasta",  help="Input protein FASTA")
    parser.add_argument("--domains", help="Domain annotations (TSV or raw HMMER domtblout)")
    parser.add_argument("--out",    help="Output iTOL annotation file")
    parser.add_argument(
        "--evalue", type=float, default=1e-5,
        help="E-value threshold for domtblout filtering (default: 1e-5)"
    )
    args = parser.parse_args()
    convert_domains(args.fasta, args.domains, args.out, args.evalue)