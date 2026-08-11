#!/usr/bin/env python3
"""
split_by_domain.py — Split a multi-sequence protein FASTA into per-domain
subsequences using a resolved HMMER/dbCAN domain TSV.

For each protein, every domain hit's envelope (env_from-env_to) is extracted
as its own subsequence and bucketed into an output FASTA by domain group
(configurable via --group-regex, e.g. GH43_1/GH43_12 -> GH43). Any leftover
sequence not covered by a domain envelope (before/between/after hits, or the
whole sequence for proteins with no hits) is written to a separate
"unannotated" FASTA.

Input TSV: same resolved format consumed by domain_utils.parse_domains(),
i.e. an exact header row with columns:
    protein_name, protein_len, domain_name, env_from, env_to

Usage:
    python -m domain_handling.split_by_domain \
        --fasta proteins.fa \
        --domains resolved.domtblout \
        --outdir split_fasta/
"""

import argparse
import os
import re
import sys

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from domain_handling.domain_utils import parse_domains

DEFAULT_GROUP_REGEX = r"^([A-Za-z]+\d+)"


def bucket_domain(domain_name: str, regex: re.Pattern, warned: set) -> str:
    match = regex.search(domain_name)
    if not match:
        if domain_name not in warned:
            print(
                f"WARNING: domain_name '{domain_name}' did not match --group-regex, "
                f"using literal name as its own group",
                file=sys.stderr,
            )
            warned.add(domain_name)
        return domain_name
    return match.group(1) if regex.groups >= 1 else match.group(0)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return cleaned or "group"


def merge_intervals(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_unannotated_regions(seq_len: int, merged_covered):
    if not merged_covered:
        return [(1, seq_len)] if seq_len > 0 else []

    gaps = []
    cursor = 1
    for start, end in merged_covered:
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= seq_len:
        gaps.append((cursor, seq_len))
    return gaps


def extract_subsequence(record: SeqRecord, start: int, end: int, min_len: int, warned_clip: set):
    seq_len = len(record.seq)
    clipped_start = max(1, start)
    clipped_end = min(seq_len, end)

    if (clipped_start, clipped_end) != (start, end) and record.id not in warned_clip:
        print(
            f"WARNING: coordinates {start}-{end} for '{record.id}' exceed sequence "
            f"length {seq_len}, clipped to {clipped_start}-{clipped_end}",
            file=sys.stderr,
        )
        warned_clip.add(record.id)

    if clipped_start > clipped_end:
        return None

    frag = str(record.seq)[clipped_start - 1:clipped_end]
    if len(frag) < min_len:
        return None
    return frag, clipped_start, clipped_end


def build_domain_record(record, domain_name, group, start, end, frag) -> SeqRecord:
    new_id = f"{record.id}__{domain_name}__{start}-{end}"
    description = (
        f"group={group} domain={domain_name} env={start}-{end} "
        f"length={len(frag)} source={record.description}"
    )
    return SeqRecord(Seq(frag), id=new_id, description=description)


def build_unannotated_record(record, start, end, frag) -> SeqRecord:
    new_id = f"{record.id}__unannotated__{start}-{end}"
    description = (
        f"group=unannotated env={start}-{end} length={len(frag)} source={record.description}"
    )
    return SeqRecord(Seq(frag), id=new_id, description=description)


def check_length_agreement(pid: str, seq_len: int, info: dict, policy: str):
    if info is None:
        return
    tsv_len = info["length"]
    if tsv_len != seq_len:
        message = f"protein_len mismatch for {pid}: TSV={tsv_len} FASTA={seq_len}"
        if policy == "error":
            sys.exit(f"ERROR: {message}")
        print(f"WARNING: {message}", file=sys.stderr)


def write_group_fastas(groups: dict, unannotated: list, outdir: str, unannotated_basename: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    counts = {}

    for group, records in groups.items():
        path = os.path.join(outdir, f"{sanitize_filename(group)}.fasta")
        SeqIO.write(records, path, "fasta")
        counts[group] = len(records)

    unannotated_path = os.path.join(outdir, f"{sanitize_filename(unannotated_basename)}.fasta")
    SeqIO.write(unannotated, unannotated_path, "fasta")
    counts[unannotated_basename] = len(unannotated)

    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a protein FASTA into per-domain subsequences by envelope coordinates."
    )
    parser.add_argument("--fasta", required=True, help="Input multi-sequence protein FASTA")
    parser.add_argument("--domains", required=True, help="Resolved domain TSV (see domain_utils.parse_domains)")
    parser.add_argument("--outdir", required=True, help="Output directory for split FASTAs")
    parser.add_argument(
        "--group-regex",
        default=DEFAULT_GROUP_REGEX,
        help=(
            "Regex used to bucket domain_name into an output group/filename. "
            "Capture group 1 is used if present, else the whole match. "
            f"Default: {DEFAULT_GROUP_REGEX!r} (e.g. GH43_1, GH43_12 -> GH43)"
        ),
    )
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Disable --group-regex bucketing; each exact domain_name gets its own output FASTA",
    )
    parser.add_argument(
        "--min-fragment-length",
        type=int,
        default=1,
        help="Drop domain fragments shorter than this after clipping (default: 1)",
    )
    parser.add_argument(
        "--min-unannotated-length",
        type=int,
        default=10,
        help="Drop unannotated fragments shorter than this after clipping (default: 10)",
    )
    parser.add_argument(
        "--unannotated-name",
        default="unannotated",
        help="Basename (no extension) of the leftover-region output FASTA (default: unannotated)",
    )
    parser.add_argument(
        "--on-length-mismatch",
        choices=["warn", "error"],
        default="warn",
        help="Behavior when TSV protein_len disagrees with actual FASTA sequence length (default: warn)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        regex = re.compile(args.group_regex)
    except re.error as e:
        sys.exit(f"ERROR: invalid --group-regex: {e}")

    proteins = parse_domains(args.domains)

    groups = {}
    unannotated = []
    warned_bucket = set()
    warned_clip = set()
    seen_ids = set()

    for record in SeqIO.parse(args.fasta, "fasta"):
        pid = record.id
        seen_ids.add(pid)
        seq_len = len(record.seq)
        info = proteins.get(pid)

        check_length_agreement(pid, seq_len, info, args.on_length_mismatch)

        domain_hits = info["domains"] if info else []

        for domain_name, start, end in domain_hits:
            result = extract_subsequence(record, start, end, args.min_fragment_length, warned_clip)
            if result is None:
                continue
            frag, clipped_start, clipped_end = result
            group = domain_name if args.no_group else bucket_domain(domain_name, regex, warned_bucket)
            groups.setdefault(group, []).append(
                build_domain_record(record, domain_name, group, clipped_start, clipped_end, frag)
            )

        covered = merge_intervals([(start, end) for _, start, end in domain_hits])
        for gap_start, gap_end in compute_unannotated_regions(seq_len, covered):
            result = extract_subsequence(record, gap_start, gap_end, args.min_unannotated_length, warned_clip)
            if result is None:
                continue
            frag, clipped_start, clipped_end = result
            unannotated.append(build_unannotated_record(record, clipped_start, clipped_end, frag))

    missing_from_fasta = sorted(set(proteins) - seen_ids)
    if missing_from_fasta:
        example = ", ".join(missing_from_fasta[:10])
        print(
            f"WARNING: {len(missing_from_fasta)} protein(s) in --domains have no matching "
            f"FASTA record and could not be extracted (e.g. {example})",
            file=sys.stderr,
        )

    no_hit_count = sum(1 for pid in seen_ids if pid not in proteins)
    print(f"{no_hit_count} of {len(seen_ids)} FASTA records had no domain hits (fully unannotated)")

    counts = write_group_fastas(groups, unannotated, args.outdir, args.unannotated_name)
    for group, count in sorted(counts.items()):
        print(f"{group}: {count} sequence(s)")


if __name__ == "__main__":
    main()
