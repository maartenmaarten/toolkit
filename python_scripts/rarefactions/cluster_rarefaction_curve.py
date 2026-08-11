#!/usr/bin/env python3
"""
cluster_rarefaction_curve.py — A genuine rarefaction curve for the
characterized GH43 set: cumulative number of UNIQUE SEQUENCE CLUSTERS
discovered as a function of samples added, either in random order
(permutation, averaged) or in real historical (GenBank submission date)
order.

This differs from sequence_novelty_decay.py in a fundamental way: clusters
are discrete, countable units (membership is binary — a sequence belongs to
exactly one cluster), so "cumulative unique clusters seen" is monotonically
non-decreasing by construction, exactly like a species-accumulation /
gene-accumulation rarefaction curve in ecology or pan-genomics. The novelty-
decay curves in the other script are a continuous per-sequence distance
metric and are NOT monotonic — a different and complementary kind of
evidence, not a substitute for this.

INPUTS
------
--cluster  : MMseqs2 cluster TSV (e.g. from `mmseqs cluster` / `easy-cluster`),
             NO header row, two columns: cluster_representative \\t member.
             Each unique value in column 1 defines one discrete cluster.

--genbank  : GenBank flat file (.gb) containing all sequences. Same date
             extraction logic as sequence_novelty_decay.py: earliest
             "Submitted (DD-MON-YYYY)" date across all REFERENCE blocks,
             falling back to earliest JOURNAL publication year if no
             explicit Submitted date exists. Sequences with no usable date
             are dropped (and reported) for the date-ordered curve only —
             they are still included in the permutation curve, since that
             one doesn't need dates.

OUTPUTS
-------
--permutation-out  : PNG, cumulative unique clusters vs. samples (random order)
--date-ordered-out : PNG, cumulative unique clusters vs. samples (real date order)
--permutations     : number of random orderings to average (default 50)

Usage:
    python cluster_rarefaction_curve.py \\
        --cluster GH43_char_cluster_0_9-0_9.tsv \\
        --genbank GH43_characterized.gb \\
        --permutation-out cluster_rarefaction_permutation.png \\
        --date-ordered-out cluster_rarefaction_date_ordered.png
"""

import sys
import csv
import random
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

SUBMITTED_RE = re.compile(r"Submitted\s+\((\d{2})-([A-Z]{3})-(\d{4})\)")
JOURNAL_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


# ── parsing: cluster membership ─────────────────────────────────────────────

def parse_clusters(path: str) -> dict:
    """Returns dict: member_accession -> cluster_representative.
    Each unique cluster_representative value defines one discrete cluster.
    """
    member_to_cluster = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            representative, member = row[0].strip(), row[1].strip()
            member_to_cluster[member] = representative
    return member_to_cluster


# ── parsing: genbank dates (same logic as sequence_novelty_decay.py) ───────

def _extract_earliest_date_from_record_text(raw_text: str):
    submitted_dates = []
    for day, mon, year in SUBMITTED_RE.findall(raw_text):
        try:
            submitted_dates.append(datetime(int(year), MONTHS[mon], int(day)))
        except (KeyError, ValueError):
            continue
    if submitted_dates:
        return min(submitted_dates), False

    journal_years = []
    for line in raw_text.splitlines():
        if "JOURNAL" in line:
            m = JOURNAL_YEAR_RE.search(line.strip())
            if m:
                journal_years.append(int(m.group(1)))
    if journal_years:
        return datetime(min(journal_years), 1, 1), True

    return None, None


def parse_genbank_dates(path: str) -> dict:
    with open(path) as fh:
        full_text = fh.read()
    records_raw = full_text.split("//\n")

    dates = {}
    n_fallback = 0
    dropped = []

    for record_text in records_raw:
        if not record_text.strip():
            continue
        version_match = re.search(r"^VERSION\s+(\S+)", record_text, re.MULTILINE)
        if not version_match:
            continue
        accession = version_match.group(1)

        date, was_fallback = _extract_earliest_date_from_record_text(record_text)
        if date is None:
            dropped.append(accession)
            continue
        if was_fallback:
            n_fallback += 1
        dates[accession] = date

    print(f"  Parsed dates for {len(dates)} records from {path}")
    print(f"  Used JOURNAL-year fallback (no explicit Submitted date) for {n_fallback} records")
    if dropped:
        print(f"  WARNING: no usable date found for {len(dropped)} record(s), dropped: {dropped[:10]}"
              + (" ..." if len(dropped) > 10 else ""))

    return dates


# ── permutation-based rarefaction ───────────────────────────────────────────

def permutation_rarefaction(accessions: list, member_to_cluster: dict, n_permutations: int = 50):
    """For each random ordering, track cumulative unique clusters seen at
    each rank position. Returns array of shape (n_permutations, n_sequences).
    """
    n = len(accessions)
    all_curves = np.zeros((n_permutations, n), dtype=int)

    for p in range(n_permutations):
        order = accessions.copy()
        random.shuffle(order)
        seen_clusters = set()
        for i, acc in enumerate(order):
            cluster = member_to_cluster[acc]
            seen_clusters.add(cluster)
            all_curves[p, i] = len(seen_clusters)

    return all_curves


def plot_permutation_rarefaction(all_curves: np.ndarray, out_path: str, n_permutations: int, total_clusters: int):
    n = all_curves.shape[1]
    mean_curve = all_curves.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for p in range(min(n_permutations, all_curves.shape[0])):
        ax.plot(range(1, n + 1), all_curves[p], color="#AAAAAA", alpha=0.15, linewidth=0.7)
    ax.plot(range(1, n + 1), mean_curve, color=sns.color_palette("husl", 1)[0],
             linewidth=2, label=f"Mean over {n_permutations} random orderings")
    ax.axhline(total_clusters, color="black", linestyle=":", linewidth=1,
               label=f"Total unique clusters (n={total_clusters})")

    ax.set_xlabel("Number of sequences sampled (random order)")
    ax.set_ylabel("Cumulative unique clusters observed")
    ax.set_title("Cluster rarefaction curve (permutation-based)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {out_path}")

    # plateau heuristic: slope over final 10% of samples
    tail_start = int(n * 0.9)
    tail_slope = (mean_curve[-1] - mean_curve[tail_start]) / max(1, n - tail_start)
    print(f"  Final cumulative clusters: {mean_curve[-1]:.1f} of {total_clusters} total")
    print(f"  Slope over final 10% of samples: {tail_slope:.4f} new clusters/sequence")
    if tail_slope > 0.01:
        print("  -> Curve has NOT plateaued: cluster space appears undersampled")
    else:
        print("  -> Curve appears close to plateau at this sample size")


# ── date-ordered rarefaction ────────────────────────────────────────────────

def date_ordered_rarefaction(accessions_with_dates: list, member_to_cluster: dict):
    """accessions_with_dates: list of (accession, date), sorted by date.
    Returns list of (date, cumulative_unique_clusters, rank).
    """
    seen_clusters = set()
    results = []
    for rank, (acc, date) in enumerate(accessions_with_dates, start=1):
        cluster = member_to_cluster[acc]
        seen_clusters.add(cluster)
        results.append((date, len(seen_clusters), rank))
    return results


def plot_date_ordered_rarefaction(results: list, out_path: str, total_clusters: int):
    dates = [r[0] for r in results]
    cumulative = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(dates, cumulative, where="post", color=sns.color_palette("husl", 1)[0], linewidth=2)
    ax.axhline(total_clusters, color="black", linestyle=":", linewidth=1,
               label=f"Total unique clusters (n={total_clusters})")

    ax.set_xlabel("GenBank submission date (earliest Submitted date per record)")
    ax.set_ylabel("Cumulative unique clusters observed")
    ax.set_title("Cluster rarefaction curve (real historical discovery order)\n"
                 "Caution: confounded by non-uniform sequencing effort over time",
                 fontsize=10)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {out_path}")
    print(f"  Final cumulative clusters: {cumulative[-1]} of {total_clusters} total")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cluster-based rarefaction curves (permutation + date-ordered)")
    parser.add_argument("--cluster", required=True, help="MMseqs2 cluster TSV (representative, member columns, no header)")
    parser.add_argument("--genbank", required=True, help="GenBank flat file with all sequences")
    parser.add_argument("--permutation-out", required=True, help="Output PNG for permutation rarefaction curve")
    parser.add_argument("--date-ordered-out", required=True, help="Output PNG for date-ordered rarefaction curve")
    parser.add_argument("--permutations", type=int, default=50, help="Number of random orderings (default: 50)")
    args = parser.parse_args()

    print("Parsing cluster membership...")
    member_to_cluster = parse_clusters(args.cluster)
    total_clusters = len(set(member_to_cluster.values()))
    print(f"  {len(member_to_cluster)} sequences across {total_clusters} unique clusters")

    print("\nComputing permutation-based rarefaction curve...")
    all_accessions = list(member_to_cluster.keys())
    perm_curves = permutation_rarefaction(all_accessions, member_to_cluster, n_permutations=args.permutations)
    plot_permutation_rarefaction(perm_curves, args.permutation_out, args.permutations, total_clusters)

    print("\nParsing GenBank dates...")
    dates = parse_genbank_dates(args.genbank)

    accessions_with_dates = [(acc, d) for acc, d in dates.items() if acc in member_to_cluster]
    missing_cluster_info = set(dates.keys()) - set(member_to_cluster.keys())
    if missing_cluster_info:
        print(f"  WARNING: {len(missing_cluster_info)} dated accession(s) have no cluster assignment, excluded "
              f"from date-ordered curve: {sorted(missing_cluster_info)[:5]}"
              + (" ..." if len(missing_cluster_info) > 5 else ""))

    accessions_with_dates.sort(key=lambda x: x[1])
    print(f"  {len(accessions_with_dates)} sequences with both a date and a cluster assignment")

    print("\nComputing date-ordered rarefaction curve...")
    date_results = date_ordered_rarefaction(accessions_with_dates, member_to_cluster)
    plot_date_ordered_rarefaction(date_results, args.date_ordered_out, total_clusters)


if __name__ == "__main__":
    main()
