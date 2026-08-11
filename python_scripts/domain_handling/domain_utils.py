#!/usr/bin/env python3
"""
domain_utils.py — Shared parsing logic for resolved HMMER/dbCAN domain TSV files.

Provides functions to parse per-protein domain hits and collapse them into
unique domain architectures. Used by domain_architecture_viewer.py (schematic
diagrams) and domain_architecture_stats.py (co-occurrence statistics).

Input: TSV with an exact header row containing these columns (case-sensitive,
no auto-detection/aliasing — column names must match exactly):
    protein_name, protein_len, domain_name, env_from, env_to
    (one row per domain hit; multiple rows per protein for multi-domain proteins)
"""

import csv
import random
import statistics
import sys
from collections import defaultdict

REQUIRED_COLUMNS = ["protein_name", "protein_len", "domain_name", "env_from", "env_to"]


def parse_domains(path: str) -> dict:
    """Returns dict: protein_id -> {"length": int, "domains": [(name, start, end), ...]}

    Expects exact column names: protein_name, protein_len, domain_name, env_from, env_to.
    """
    with open(path) as fh:
        sample = fh.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters="\t,")

    proteins = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        headers = reader.fieldnames or []
        if not headers:
            sys.exit("ERROR: domains file appears empty or has no header row")

        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if missing:
            sys.exit(
                f"ERROR: missing required column(s): {missing}\n"
                f"Available columns: {headers}\n"
                f"Required (exact names): {REQUIRED_COLUMNS}"
            )

        for row in reader:
            pid = row["protein_name"].strip()
            domain = row["domain_name"].strip().replace(".hmm", "")
            start = int(row["env_from"])
            end = int(row["env_to"])
            plen = int(row["protein_len"])

            if pid not in proteins:
                proteins[pid] = {"length": plen, "domains": []}
            proteins[pid]["domains"].append((domain, start, end))

    for pid in proteins:
        proteins[pid]["domains"].sort(key=lambda d: d[1])

    return proteins


def collapse_architectures(proteins: dict):
    """Group proteins by their ordered domain-name tuple (the 'architecture').

    Returns:
        architectures: dict arch_tuple -> list of protein_ids sharing it
        representative: dict arch_tuple -> one example protein's full domain
                         list (with positions), used for drawing proportions.
                         Chosen as the protein whose length is nearest the
                         architecture's median length; ties broken at random.
    """
    architectures = defaultdict(list)

    for pid, info in proteins.items():
        arch = tuple(d[0] for d in info["domains"])
        architectures[arch].append(pid)

    representative = {}
    for arch, pids in architectures.items():
        median_len = statistics.median(proteins[pid]["length"] for pid in pids)
        diffs = {pid: abs(proteins[pid]["length"] - median_len) for pid in pids}
        min_diff = min(diffs.values())
        candidates = [pid for pid in pids if diffs[pid] == min_diff]
        representative[arch] = random.choice(candidates)

    return architectures, representative
