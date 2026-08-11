#!/usr/bin/env python3
"""
domain_architecture_stats.py — Compute numerical domain co-occurrence and
correlation statistics from a resolved HMMER/dbCAN domain TSV.

NOTE: This script assumes the input TSV has already been domain-overlap-resolved
(i.e. one row per accepted domain hit, not raw unfiltered HMMER output) — same
assumption as domain_architecture_viewer.py.

Input: TSV with an exact header row containing these columns (case-sensitive,
no auto-detection/aliasing — column names must match exactly):
    protein_name, protein_len, domain_name, env_from, env_to
    (one row per domain hit; multiple rows per protein for multi-domain proteins)

Usage:
    python domain_architecture_stats.py --domains GH43_characterized_dbcan_resolved.tsv \
        --cooccurrence-out cooccurrence.tsv \
        --frequency-out domain_frequency.tsv
"""

import argparse
import csv
import fnmatch
import math
from collections import defaultdict
from itertools import combinations

from scipy.stats import fisher_exact

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from domain_handling.domain_utils import parse_domains, collapse_architectures


def _try_bh_correction(pvalues):
    """Benjamini-Hochberg FDR correction. Uses statsmodels if available, otherwise manual."""
    try:
        from statsmodels.stats.multitest import multipletests
        _, qvalues, _, _ = multipletests(pvalues, method="fdr_bh")
        return list(qvalues)
    except ImportError:
        pass
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    qvalues = [0.0] * n
    cummin = 1.0
    for rank_from_end, (orig_idx, pval) in enumerate(reversed(indexed)):
        rank = n - rank_from_end
        bh = pval * n / rank
        cummin = min(cummin, bh)
        qvalues[orig_idx] = min(cummin, 1.0)
    return qvalues


def build_domain_protein_sets(proteins: dict) -> dict:
    """Build domain -> set(protein_ids) lookup."""
    domain_sets = defaultdict(set)
    for pid, info in proteins.items():
        for dname, _, _ in info["domains"]:
            domain_sets[dname].add(pid)
    return dict(domain_sets)


def filter_domain_sets(domain_sets: dict, patterns) -> dict:
    """Keep only domains whose name matches one of the given fnmatch pattern(s).

    patterns: a single pattern string or a list of patterns, e.g. "GH43*" or
    ["GH43*", "CBM6"]. Matches against domain names as parsed by
    domain_utils.parse_domains (".hmm" suffix already stripped, e.g. "GH43_11").
    Used to scope the frequency report to the domains actually being compared
    (see filter_cooccurrence_rows) — not to restrict what compute_cooccurrence()
    itself tests.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    return {
        name: pids for name, pids in domain_sets.items()
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    }


def filter_cooccurrence_rows(cooc_rows: list, group_a, group_b=None) -> list:
    """Keep only co-occurrence rows pairing a domain from group_a with one from group_b.

    group_a, group_b: an fnmatch pattern or list of patterns, e.g. "GH43*",
    ["CBM*"], or "GH43_1". If group_b is omitted, defaults to group_a — i.e.
    "only pairs where both domains are in group_a" (e.g. GH43-internal
    co-occurrence). Pass group_b=["*"] for "group_a vs everything else".
    Matching is order-independent (domain_a/domain_b are just alphabetical,
    not tied to group membership): a row is kept if either domain lands in
    group_a and the other in group_b.

    Run this AFTER compute_cooccurrence() on the full domain universe — the
    full pairwise matrix (and its BH q-value correction) should always be
    computed first; this only narrows what gets reported/plotted, so cherry-
    picking a subset here never inflates apparent significance.
    """
    if group_b is None:
        group_b = group_a
    if isinstance(group_a, str):
        group_a = [group_a]
    if isinstance(group_b, str):
        group_b = [group_b]

    def matches(name, patterns):
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    return [
        r for r in cooc_rows
        if (matches(r["domain_a"], group_a) and matches(r["domain_b"], group_b))
        or (matches(r["domain_a"], group_b) and matches(r["domain_b"], group_a))
    ]


def compute_cooccurrence(proteins: dict, domain_sets: dict):
    """Compute pairwise co-occurrence stats for all domain pairs."""
    n_total = len(proteins)
    domains = sorted(domain_sets.keys())

    rows = []
    for domain_a, domain_b in combinations(domains, 2):
        set_a = domain_sets[domain_a]
        set_b = domain_sets[domain_b]
        n_a = len(set_a)
        n_b = len(set_b)
        n_ab = len(set_a & set_b)

        jaccard = n_ab / (n_a + n_b - n_ab) if (n_a + n_b - n_ab) > 0 else 0.0

        # 2x2 contingency table
        both = n_ab
        a_only = n_a - n_ab
        b_only = n_b - n_ab
        neither = n_total - n_a - n_b + n_ab

        # phi coefficient (Pearson correlation on binary presence/absence)
        denom = math.sqrt((both + a_only) * (b_only + neither) *
                          (both + b_only) * (a_only + neither))
        phi = (both * neither - a_only * b_only) / denom if denom > 0 else 0.0

        odds_ratio, p_value = fisher_exact([[both, a_only], [b_only, neither]])

        rows.append({
            "domain_a": domain_a,
            "domain_b": domain_b,
            "n_a": n_a,
            "n_b": n_b,
            "n_ab": n_ab,
            "jaccard": jaccard,
            "phi": phi,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
        })

    pvalues = [r["p_value"] for r in rows]
    qvalues = _try_bh_correction(pvalues)
    for r, q in zip(rows, qvalues):
        r["q_value"] = q

    rows.sort(key=lambda r: (-r["phi"]))
    rows.sort(key=lambda r: r["q_value"])

    return rows


def compute_frequency(proteins: dict, domain_sets: dict):
    """Compute per-domain frequency summary."""
    n_total = len(proteins)
    rows = []
    for dname in sorted(domain_sets.keys()):
        n = len(domain_sets[dname])
        rows.append({
            "domain_name": dname,
            "n_proteins": n,
            "pct_of_total": 100.0 * n / n_total if n_total > 0 else 0.0,
        })
    rows.sort(key=lambda r: r["n_proteins"], reverse=True)
    return rows


def print_architecture_summary(proteins: dict, architectures: dict):
    """Print architecture-level summary statistics to stdout."""
    n_proteins = len(proteins)
    n_archs = len(architectures)
    n_singletons = sum(1 for members in architectures.values() if len(members) == 1)

    domain_counts = [len(info["domains"]) for info in proteins.values()]
    lengths = [info["length"] for info in proteins.values()]

    single_domain_lengths = [info["length"] for info in proteins.values() if len(info["domains"]) == 1]
    multi_domain_lengths = [info["length"] for info in proteins.values() if len(info["domains"]) > 1]

    def median(vals):
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0

    def sd(vals):
        if len(vals) < 2:
            return 0.0
        m = mean(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    print("\n=== Architecture-level summary ===")
    print(f"Total proteins:           {n_proteins}")
    print(f"Total unique architectures: {n_archs}")
    print(f"Singleton architectures (n=1): {n_singletons}")
    print(f"\nDomains per protein:")
    print(f"  Mean:   {mean(domain_counts):.2f}  SD: {sd(domain_counts):.2f}")
    print(f"  Median: {median(domain_counts):.1f}")
    print(f"\nProtein length (aa):")
    print(f"  All proteins:          mean={mean(lengths):.0f}  SD={sd(lengths):.0f}  median={median(lengths):.0f}  (n={len(lengths)})")
    if single_domain_lengths:
        print(f"  Single-domain:         mean={mean(single_domain_lengths):.0f}  SD={sd(single_domain_lengths):.0f}  median={median(single_domain_lengths):.0f}  (n={len(single_domain_lengths)})")
    if multi_domain_lengths:
        print(f"  Multi-domain:          mean={mean(multi_domain_lengths):.0f}  SD={sd(multi_domain_lengths):.0f}  median={median(multi_domain_lengths):.0f}  (n={len(multi_domain_lengths)})")
    if single_domain_lengths and multi_domain_lengths:
        diff = mean(multi_domain_lengths) - mean(single_domain_lengths)
        print(f"  Difference (multi-single): {diff:+.0f} aa mean")


def _heatmap_color_scale(matrix, metric):
    """Match the original fixed color scales: phi is data-driven (diverging,
    symmetric around 0); jaccard is always the fixed 0-1 similarity range."""
    if metric == "phi":
        finite = matrix[np.isfinite(matrix)]
        vmax = max(abs(finite.min()), abs(finite.max()), 0.01) if finite.size else 0.01
        return "RdBu_r", 0.0, -vmax, vmax, "phi coefficient"
    return "YlOrRd", None, 0.0, 1.0, "Jaccard index"


def draw_cooccurrence_heatmap(
    cooc_rows: list,
    out_path: str,
    metric: str = "phi",
    x_domains: list = None,
    y_domains: list = None,
):
    """Heatmap of pairwise domain co-occurrence.

    metric: 'phi' (diverging, -1 to +1 correlation) or
            'jaccard' (0–1 similarity).

    If x_domains and y_domains are both given, draws a rectangular matrix —
    x_domains as columns, y_domains as rows — so each pair's correlation
    appears exactly once (no mirrored upper/lower triangle). Use this for
    two-group comparisons, e.g. CBM domains (rows) vs GH43 domains (columns).

    Otherwise falls back to the symmetric clustered heatmap over every domain
    present in cooc_rows (single-group / full-matrix view), derived from
    whichever rows are passed in — a filtered subset (see
    filter_cooccurrence_rows) only draws the domains actually involved in it.
    """
    if x_domains is not None and y_domains is not None:
        _draw_rectangular_heatmap(cooc_rows, out_path, metric, x_domains, y_domains)
    else:
        _draw_symmetric_heatmap(cooc_rows, out_path, metric)


def _draw_rectangular_heatmap(cooc_rows, out_path, metric, x_domains, y_domains):
    value_lookup = {}
    for row in cooc_rows:
        value_lookup[(row["domain_a"], row["domain_b"])] = row[metric]
        value_lookup[(row["domain_b"], row["domain_a"])] = row[metric]

    matrix = np.full((len(y_domains), len(x_domains)), np.nan)
    for i, dy in enumerate(y_domains):
        for j, dx in enumerate(x_domains):
            if dx != dy:
                val = value_lookup.get((dx, dy))
                if val is not None:
                    matrix[i, j] = val

    cmap, center, vmin, vmax, cbar_label = _heatmap_color_scale(matrix, metric)

    fig_w = max(6, 0.45 * len(x_domains) + 2)
    fig_h = max(6, 0.45 * len(y_domains) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix,
        xticklabels=x_domains,
        yticklabels=y_domains,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.tick_params(labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path} ({len(x_domains)} x {len(y_domains)} domains, metric={metric})")


def _draw_symmetric_heatmap(cooc_rows, out_path, metric):
    domains = sorted({r["domain_a"] for r in cooc_rows} | {r["domain_b"] for r in cooc_rows})
    n = len(domains)
    idx = {d: i for i, d in enumerate(domains)}

    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 0)

    for row in cooc_rows:
        val = row[metric]
        i, j = idx[row["domain_a"]], idx[row["domain_b"]]
        matrix[i, j] = val
        matrix[j, i] = val

    cmap, center, vmin, vmax, cbar_label = _heatmap_color_scale(matrix, metric)

    g = sns.clustermap(
        matrix,
        xticklabels=domains,
        yticklabels=domains,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        figsize=(max(6, 0.45 * n + 2), max(6, 0.45 * n + 2)),
        dendrogram_ratio=0.12,
        row_cluster=True,
        col_cluster=True,
        cbar_kws={"label": cbar_label},
        method="average",
        metric="euclidean",
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(labelsize=15)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    g.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path} ({n} domains, metric={metric})")


def write_tsv(rows, path, fieldnames):
    """Write a list of dicts to a TSV file."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute domain co-occurrence and correlation statistics"
    )
    parser.add_argument("--domains", required=True, help="Resolved domain TSV file")
    parser.add_argument("--cooccurrence-out", required=True, help="Output TSV for pairwise co-occurrence stats")
    parser.add_argument("--frequency-out", required=True, help="Output TSV for per-domain frequency summary")
    parser.add_argument("--heatmap-out", help="Optional: output path for co-occurrence heatmap (.png/.pdf/.svg)")
    parser.add_argument("--heatmap-metric", choices=["phi", "jaccard"], default="phi",
                         help="Metric for heatmap cells (default: phi)")
    parser.add_argument("--group-a", nargs="+",
                         help="Restrict the co-occurrence report/plot to pairs involving domains "
                              "matching these fnmatch pattern(s), e.g. 'GH43*'. The full pairwise "
                              "matrix (and its BH q-value correction) is always computed first "
                              "over every domain — this only filters what gets written/plotted.")
    parser.add_argument("--group-b", nargs="+",
                         help="Second domain group, for cross-group filtering (e.g. --group-a "
                              "'CBM*' --group-b 'GH43*' keeps only CBM-vs-GH43 pairs). Defaults "
                              "to --group-a if omitted (i.e. only within-group pairs). Use '*' "
                              "for 'group-a vs everything else', e.g. --group-a GH43_1 --group-b '*'.")
    args = parser.parse_args()

    proteins = parse_domains(args.domains)
    n_before = len(proteins)
    proteins = {pid: info for pid, info in proteins.items() if len(info["domains"]) > 1}
    print(f"Dropped {n_before - len(proteins)} single-domain proteins, {len(proteins)} multi-domain remain")
    architectures, _ = collapse_architectures(proteins)
    rare = {pid for arch, members in architectures.items() if len(members) < 10 for pid in members}
    if rare:
        proteins = {pid: info for pid, info in proteins.items() if pid not in rare}
        print(f"Dropped {len(rare)} proteins in architectures with n < 10, {len(proteins)} remain")
        architectures, _ = collapse_architectures(proteins)
    domain_sets = build_domain_protein_sets(proteins)

    # Always compute the full pairwise matrix (and its BH correction) first —
    # --group-a/--group-b only narrow what gets written/plotted below.
    cooc_rows = compute_cooccurrence(proteins, domain_sets)

    report_rows = cooc_rows
    freq_domain_sets = domain_sets
    if args.group_a:
        report_rows = filter_cooccurrence_rows(cooc_rows, args.group_a, args.group_b)
        union_patterns = list(args.group_a) + list(args.group_b or args.group_a)
        freq_domain_sets = filter_domain_sets(domain_sets, union_patterns)
        print(f"Group filter (group_a={args.group_a}, group_b={args.group_b or args.group_a}): "
              f"{len(report_rows)}/{len(cooc_rows)} pairs kept")

    write_tsv(report_rows, args.cooccurrence_out, [
        "domain_a", "domain_b", "n_a", "n_b", "n_ab",
        "jaccard", "phi",
        "odds_ratio", "p_value", "q_value",
    ])
    print(f"Wrote {args.cooccurrence_out} ({len(report_rows)} domain pairs)")

    freq_rows = compute_frequency(proteins, freq_domain_sets)
    write_tsv(freq_rows, args.frequency_out, ["domain_name", "n_proteins", "pct_of_total"])
    print(f"Wrote {args.frequency_out} ({len(freq_rows)} domains)")

    if args.heatmap_out:
        if args.group_a and args.group_b:
            # Two explicit groups: rectangular layout (group_a on x, group_b on y)
            # so each pair is shown once, with no mirrored upper/lower triangle.
            x_domains = sorted(filter_domain_sets(domain_sets, args.group_a).keys())
            y_domains = sorted(filter_domain_sets(domain_sets, args.group_b).keys())
            draw_cooccurrence_heatmap(report_rows, args.heatmap_out, metric=args.heatmap_metric,
                                       x_domains=x_domains, y_domains=y_domains)
        else:
            draw_cooccurrence_heatmap(report_rows, args.heatmap_out, metric=args.heatmap_metric)

    print_architecture_summary(proteins, architectures)


if __name__ == "__main__":
    main()
