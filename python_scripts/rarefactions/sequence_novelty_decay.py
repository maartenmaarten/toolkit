#!/usr/bin/env python3
"""
sequence_novelty_decay.py — Two complementary views of NOVELTY DECAY for the
characterized GH43 set. NOTE: these are continuous nearest-neighbor distance
curves, NOT rarefaction curves — they measure how dissimilar each new sequence
is from everything seen before it, not a count of discrete newly-discovered
units. For an actual rarefaction curve (cumulative unique clusters discovered
vs. samples), see cluster_rarefaction_curve.py instead, which uses discrete
sequence clusters as the countable unit.

  1. PERMUTATION novelty-decay curve: as sequences are added in random order
     (averaged over many random orderings), how does per-sequence novelty
     (= distance to nearest already-seen sequence) behave as a function of
     SAMPLE SIZE? This is agnostic to calendar time and avoids the confound
     of non-uniform historical sequencing effort.

  2. DATE-ORDERED novelty trend: as sequences are added in their REAL
     historical discovery order (by GenBank submission date), how novel was
     each sequence relative to everything already known at that time? This
     is a genuine historical trajectory (no permutation possible/needed,
     since there is only one true order) but CONFOUNDED by the fact that
     sequencing effort itself has not been constant over time (e.g. the
     post-2010 NGS/metagenomics boom). Treat this as a discovery-history
     view, not a substitute for (1).

Novelty is defined here as: for sequence i with predecessor set P(i)
(all sequences "before" it, by sample order or by date), novelty(i) =
1 - max_{j in P(i)} (pident_ij / 100), i.e. one minus the percent identity
to the most similar predecessor. If a sequence has NO recorded hit to any
predecessor in the all-vs-all file (e.g. below the e-value threshold used
to generate it), novelty is treated as 1.0 (maximally novel) — this is a
real assumption worth checking against your e-value cutoff, since a missing
hit could also mean "alignment too short/diverged to score," not just
"genuinely unrelated."

INPUTS
------
--allvsall : TSV from `mmseqs convertalis`, NO header row, columns:
             query, target, pident, evalue, bits, qlen, tlen, alnlen
             Self-hits (query == target) are ignored automatically.

--genbank  : GenBank flat file (.gb) containing all sequences. The date used
             per sequence is the EARLIEST "Submitted (DD-MON-YYYY)" date
             found across all REFERENCE blocks for that record. If no
             explicit Submitted date exists anywhere in the record, falls
             back to the earliest REFERENCE...JOURNAL publication year
             (January 1st of that year is used as a placeholder date, and
             this fallback is logged so you can check how often it triggers).
             Sequences with NO usable date anywhere are dropped, and the
             dropped accessions are printed.

OUTPUTS
-------
--permutation-out      : PNG, the permutation-based novelty-decay curve
--date-ordered-out     : PNG, the date-ordered novelty trend
--permutations          : number of random orderings to average for (1)
                           (default 50)
--summary-out (optional): TSV with per-sequence novelty values for the
                           date-ordered analysis (accession, date, novelty,
                           rank), in case you want to inspect outliers

Usage:
    python sequence_novelty_curves.py \\
        --allvsall GH43_allvsall.tsv \\
        --genbank GH43_characterized.gb \\
        --permutation-out permutation_novelty_decay.png \\
        --date-ordered-out date_ordered_novelty.png \\
        --summary-out date_ordered_novelty.tsv
"""

import sys
import csv
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

SUBMITTED_RE = re.compile(r"Submitted\s+\((\d{2})-([A-Z]{3})-(\d{4})\)")
JOURNAL_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


# ── parsing: all-vs-all similarity ──────────────────────────────────────────

def parse_allvsall(path: str) -> dict:
    """Returns dict: (query, target) -> pident (float, 0-100), excluding self-hits.
    If multiple rows exist for the same (query, target) pair, keeps the max pident.
    """
    pident = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            query, target, pid = row[0], row[1], float(row[2])
            if query == target:
                continue
            key = (query, target)
            if key not in pident or pid > pident[key]:
                pident[key] = pid
    return pident


# ── parsing: genbank dates ──────────────────────────────────────────────────

def _extract_earliest_date_from_record_text(raw_text: str):
    """Given the raw text of a single GenBank record, find the earliest
    'Submitted (DD-MON-YYYY)' date across all REFERENCE blocks. Falls back
    to the earliest JOURNAL '(YYYY)' publication year (as Jan 1 of that
    year) if no Submitted date is found. Returns (date, was_fallback) or
    (None, None) if nothing usable is found.
    """
    submitted_dates = []
    for day, mon, year in SUBMITTED_RE.findall(raw_text):
        try:
            submitted_dates.append(datetime(int(year), MONTHS[mon], int(day)))
        except (KeyError, ValueError):
            continue
    if submitted_dates:
        return min(submitted_dates), False

    # fallback: earliest JOURNAL (YYYY) publication year anywhere in the record
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
    """Returns dict: accession.version -> datetime, using earliest Submitted
    date per record (see module docstring for fallback logic). Prints a
    summary of how many records used the fallback or were dropped entirely.
    """
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


# ── novelty computation ─────────────────────────────────────────────────────

def build_similarity_lookup(pident: dict) -> dict:
    """Returns dict: accession -> {other_accession: pident}, symmetrized
    (since mmseqs all-vs-all may not report both directions identically
    for all pairs depending on search parameters — we take the max of the
    two directions if both exist).
    """
    lookup = defaultdict(dict)
    for (q, t), pid in pident.items():
        existing = lookup[q].get(t)
        if existing is None or pid > existing:
            lookup[q][t] = pid
        existing_rev = lookup[t].get(q)
        if existing_rev is None or pid > existing_rev:
            lookup[t][q] = pid
    return lookup


def novelty_given_predecessors(accession: str, predecessor_set: set, similarity: dict) -> float:
    """novelty = 1 - (max pident to any predecessor / 100).
    If no hits to any predecessor exist, novelty = 1.0 (maximally novel) —
    see module docstring caveat about this assumption.
    """
    hits = similarity.get(accession, {})
    relevant = [pid for acc, pid in hits.items() if acc in predecessor_set]
    if not relevant:
        return 1.0
    return 1.0 - (max(relevant) / 100.0)


# ── permutation-based novelty decay ───────────────────────────────────────────

def permutation_novelty_curve(accessions: list, similarity: dict, n_permutations: int = 50):
    """For each random ordering, compute novelty(i) for each sequence given
    only predecessors EARLIER IN THAT RANDOM ORDER. Returns array of shape
    (n_permutations, n_sequences) with novelty values at each rank position.
    """
    n = len(accessions)
    all_curves = np.zeros((n_permutations, n))

    for p in range(n_permutations):
        order = accessions.copy()
        random.shuffle(order)
        seen = set()
        for i, acc in enumerate(order):
            all_curves[p, i] = novelty_given_predecessors(acc, seen, similarity)
            seen.add(acc)

    return all_curves


def plot_permutation_curve(all_curves: np.ndarray, out_path: str, n_permutations: int):
    n = all_curves.shape[1]
    mean_curve = all_curves.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for p in range(min(n_permutations, all_curves.shape[0])):
        ax.plot(range(1, n + 1), all_curves[p], color="#AAAAAA", alpha=0.15, linewidth=0.7)
    ax.plot(range(1, n + 1), mean_curve, color=sns.color_palette("husl", 1)[0],
             linewidth=2, label=f"Mean over {n_permutations} random orderings")

    # rolling median as a smoother trend through the mean curve
    window = max(5, n // 20)
    rolling = np.array([
        np.median(mean_curve[max(0, i - window):i + 1]) for i in range(n)
    ])
    ax.plot(range(1, n + 1), rolling, color="black", linewidth=1.5, linestyle="--",
             label=f"Rolling median (window={window})")

    ax.set_xlabel("Number of sequences sampled (random order)")
    ax.set_ylabel("Novelty (1 − max identity to any prior sequence)")
    ax.set_title("Permutation-based novelty decay curve")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {out_path}")

    tail_start = int(n * 0.9)
    tail_mean = mean_curve[tail_start:].mean()
    head_mean = mean_curve[:max(1, n // 10)].mean()
    print(f"  Mean novelty, first 10% of samples: {head_mean:.3f}")
    print(f"  Mean novelty, final 10% of samples: {tail_mean:.3f}")
    if tail_mean > 0.5 * head_mean:
        print("  -> Novelty has NOT substantially declined with sample size: "
              "sequence space appears undersampled")
    else:
        print("  -> Novelty declines notably with sample size: "
              "later sequences increasingly resemble earlier ones")


# ── date-ordered novelty ────────────────────────────────────────────────────

def date_ordered_novelty(accessions_with_dates: list, similarity: dict):
    """accessions_with_dates: list of (accession, date), already sorted by date.
    Same-date sequences are treated as simultaneous: none of them count as
    predecessors for any other sequence sharing that exact date, but all
    become predecessors for strictly later dates once that date's batch is done.

    Returns list of (accession, date, novelty, rank) in date order.
    """
    results = []
    seen = set()
    i = 0
    n = len(accessions_with_dates)

    while i < n:
        current_date = accessions_with_dates[i][1]
        batch = []
        while i < n and accessions_with_dates[i][1] == current_date:
            batch.append(accessions_with_dates[i][0])
            i += 1
        # compute novelty for everyone in this batch against `seen` (all strictly earlier)
        for acc in batch:
            nov = novelty_given_predecessors(acc, seen, similarity)
            results.append((acc, current_date, nov))
        # NOW add the whole batch to `seen`, so within-batch members don't see each other
        seen.update(batch)

    return [(acc, date, nov, rank + 1) for rank, (acc, date, nov) in enumerate(results)]


def plot_date_ordered_novelty(results: list, out_path: str, summary_out: str = None):
    accessions = [r[0] for r in results]
    dates = [r[1] for r in results]
    novelties = [r[2] for r in results]
    ranks = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(dates, novelties, color=sns.color_palette("husl", 1)[0], alpha=0.5, s=15)

    # rolling median trend over date-ordered sequence (window in #sequences, not calendar time)
    window = max(5, len(novelties) // 20)
    novelties_arr = np.array(novelties)
    rolling = np.array([
        np.median(novelties_arr[max(0, i - window):i + 1]) for i in range(len(novelties_arr))
    ])
    ax.plot(dates, rolling, color="black", linewidth=1.5, linestyle="--",
            label=f"Rolling median (window={window} sequences)")

    ax.set_xlabel("GenBank submission date (earliest Submitted date per record)")
    ax.set_ylabel("Novelty (1 − max identity to any earlier-dated sequence)")
    ax.set_title("Date-ordered novelty trend (real historical discovery order)\n"
                 "Caution: confounded by non-uniform sequencing effort over time",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {out_path}")

    if summary_out:
        with open(summary_out, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["accession", "submission_date", "novelty", "rank"])
            for acc, date, nov, rank in results:
                writer.writerow([acc, date.strftime("%Y-%m-%d"), f"{nov:.4f}", rank])
        print(f"  Wrote {summary_out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sequence novelty decay curves (permutation + date-ordered)")
    parser.add_argument("--allvsall", required=True, help="mmseqs convertalis TSV (no header)")
    parser.add_argument("--genbank", required=True, help="GenBank flat file with all sequences")
    parser.add_argument("--permutation-out", required=True, help="Output PNG for permutation novelty-decay curve")
    parser.add_argument("--date-ordered-out", required=True, help="Output PNG for date-ordered novelty trend")
    parser.add_argument("--permutations", type=int, default=50, help="Number of random orderings (default: 50)")
    parser.add_argument("--summary-out", help="Optional: per-sequence date-ordered novelty TSV")
    args = parser.parse_args()

    print("Parsing all-vs-all similarity file...")
    pident = parse_allvsall(args.allvsall)
    similarity = build_similarity_lookup(pident)
    print(f"  {len(pident)} pairwise hits across {len(similarity)} sequences with at least one hit")

    print("Parsing GenBank dates...")
    dates = parse_genbank_dates(args.genbank)

    all_accessions_in_similarity = set(similarity.keys())
    accessions_with_dates = [(acc, d) for acc, d in dates.items() if acc in all_accessions_in_similarity]
    missing_similarity = set(dates.keys()) - all_accessions_in_similarity
    if missing_similarity:
        print(f"  NOTE: {len(missing_similarity)} dated accession(s) have no all-vs-all hits at all "
              f"(will still be included with novelty=1.0 by definition): {sorted(missing_similarity)[:5]}"
              + (" ..." if len(missing_similarity) > 5 else ""))
        # include them anyway — absence from similarity just means no recorded hits
        accessions_with_dates += [(acc, dates[acc]) for acc in missing_similarity]

    accessions_with_dates.sort(key=lambda x: x[1])
    all_accessions = [acc for acc, _ in accessions_with_dates]
    print(f"  {len(all_accessions)} sequences with both a date and inclusion in this analysis")

    print("\nComputing permutation-based novelty decay curve...")
    perm_curves = permutation_novelty_curve(all_accessions, similarity, n_permutations=args.permutations)
    plot_permutation_curve(perm_curves, args.permutation_out, args.permutations)

    print("\nComputing date-ordered novelty trend...")
    date_results = date_ordered_novelty(accessions_with_dates, similarity)
    plot_date_ordered_novelty(date_results, args.date_ordered_out, args.summary_out)


if __name__ == "__main__":
    main()
