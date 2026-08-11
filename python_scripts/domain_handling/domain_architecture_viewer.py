#!/usr/bin/env python3
"""
domain_architecture_viewer.py — Collapse per-protein domain hits into unique
domain architectures, and draw a schematic diagram of each architecture
(domains-as-boxes-on-a-backbone), sorted by frequency.

This solves the "63k proteins is too many to visualize individually" problem
by first collapsing to the much smaller set of UNIQUE domain architectures
(ordered sequences of domain names), then drawing one schematic row per
unique architecture, annotated with how many proteins share it.

Input: TSV with an exact header row containing these columns (case-sensitive,
no auto-detection/aliasing — column names must match exactly):
    protein_name, protein_len, domain_name, env_from, env_to
    (one row per domain hit; multiple rows per protein for multi-domain proteins)

Usage:
    python domain_architecture_viewer.py --domains GH43_characterized_dbcan_resolved.domtblout \
        --out architecture_diagram.png --top 30
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from domain_handling.domain_utils import parse_domains, collapse_architectures


def assign_domain_colors(architectures: dict) -> dict:
    all_domains = sorted({d for arch in architectures for d in arch})
    palette = sns.color_palette("husl", len(all_domains)).as_hex()
    return {d: palette[i] for i, d in enumerate(all_domains)}


def cluster_order(architectures: dict) -> list:
    """Order architectures so that those sharing domains are placed near
    each other, using hierarchical clustering (Jaccard distance) on a
    binary architecture x domain presence matrix.

    Returns a list of architecture tuples in clustered (dendrogram leaf) order.
    Architectures with only one or two total distinct domains across the
    whole set fall back to frequency order (clustering is degenerate/uninformative
    with too few features).
    """
    arch_list = list(architectures.keys())
    all_domains = sorted({d for arch in arch_list for d in arch})

    if len(arch_list) < 3 or len(all_domains) < 2:
        # not enough architectures/domains to cluster meaningfully
        return sorted(arch_list, key=lambda a: len(architectures[a]), reverse=True)

    domain_index = {d: i for i, d in enumerate(all_domains)}
    matrix = np.zeros((len(arch_list), len(all_domains)), dtype=int)
    for i, arch in enumerate(arch_list):
        for d in arch:
            matrix[i, domain_index[d]] = 1

    # Jaccard distance; architectures with identical domain content collapse to distance 0
    dist = pdist(matrix, metric="jaccard")
    # rows that are all-zero (no domains) produce nan in jaccard; guard against that
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="average")
    dendro = dendrogram(Z, no_plot=True)
    order = dendro["leaves"]

    return [arch_list[i] for i in order]



def draw_architectures(
    proteins: dict,
    architectures: dict,
    representative: dict,
    out_path: str,
    top_n: int = 30,
    sort_by: str = "frequency",
):
    colors = assign_domain_colors(architectures)

    if sort_by == "similarity":
        # cluster order is computed over ALL architectures (so neighbours are
        # determined by the full domain-content landscape), then truncated
        # to the top_n MOST FREQUENT architectures, preserving cluster adjacency
        # among the ones actually shown.
        full_order = cluster_order(architectures)
        if top_n:
            keep = set(sorted(architectures, key=lambda a: len(architectures[a]), reverse=True)[:top_n])
            ranked_archs = [a for a in full_order if a in keep]
        else:
            ranked_archs = full_order
        ranked = [(a, architectures[a]) for a in ranked_archs]
    else:
        # sort by frequency descending (default)
        ranked = sorted(architectures.items(), key=lambda kv: len(kv[1]), reverse=True)
        if top_n:
            ranked = ranked[:top_n]

    n_rows = len(ranked)
    max_len = max(proteins[pid]["length"] for arch, _ in ranked for pid in [representative[arch]])

    fig_height = max(2, 0.4 * n_rows + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    row_height = 0.6
    backbone_height = 0.12

    for i, (arch, members) in enumerate(ranked):
        y = n_rows - i - 1
        rep_pid = representative[arch]
        rep_info = proteins[rep_pid]
        plen = rep_info["length"]

        # backbone
        ax.add_patch(mpatches.Rectangle(
            (0, y + row_height / 2 - backbone_height / 2), plen, backbone_height,
            facecolor="#dddddd", edgecolor="none", zorder=1
        ))

        # domains
        for dname, start, end in rep_info["domains"]:
            box_width = end - start
            ax.add_patch(mpatches.Rectangle(
                (start, y + 0.05), box_width, row_height - 0.1,
                facecolor=colors[dname], edgecolor="black", linewidth=0.5, zorder=2
            ))
            # label inside the box if it fits, rotated if box is narrow
            ax.text(
                start + box_width / 2, y + row_height / 2, dname,
                ha="center", va="center", fontsize=6.5, zorder=3,
                rotation=90 if box_width < max_len * 0.06 else 0,
                clip_on=True,
            )

        # frequency label
        n = len(members)
        ax.text(
            max_len * 1.02, y + row_height / 2,
            f"n={n}",
            va="center", ha="left", fontsize=8
        )

    ax.set_xlim(0, max_len * 1.15)
    ax.set_ylim(0, n_rows)
    ax.set_yticks([])
    ax.set_xlabel("Position (aa, N→C)")
    ax.set_title(f"Unique domain architectures (top {len(ranked)} of {len(architectures)} total, sorted by {sort_by})")

    # legend — restricted to domains actually appearing in the plotted (top_n) architectures
    shown_domains = sorted({d for arch, _ in ranked for d in arch})
    legend_handles = [
        mpatches.Patch(facecolor=colors[domain], edgecolor="black", label=domain)
        for domain in shown_domains
    ]
    ax.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(8, len(legend_handles)), fontsize=8, frameon=False
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {out_path}")
    print(f"  Total unique architectures: {len(architectures)}")
    print(f"  Total proteins: {sum(len(v) for v in architectures.values())}")
    print(f"  Top architecture: {'-'.join(ranked[0][0]) or '(none)'} (n={len(ranked[0][1])})")


def draw_frequency_rank(architectures: dict, out_path: str):
    """Bar chart of all architectures ranked by frequency (long-tail view).
    Complements the schematic diagram by showing the FULL distribution,
    not just the top N — relevant for assessing how skewed/long-tailed
    the architecture distribution is.
    """
    ranked = sorted(architectures.items(), key=lambda kv: len(kv[1]), reverse=True)
    counts = [len(v) for _, v in ranked]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(counts) + 1), counts, color=sns.color_palette("husl", 1)[0], width=1.0)
    ax.set_xlabel("Architecture rank (by frequency)")
    ax.set_ylabel("Number of proteins")
    ax.set_title(f"Domain architecture frequency distribution (n={len(counts)} unique architectures)")
    ax.set_yscale("log")

    n_singletons = sum(1 for c in counts if c == 1)
    ax.text(
        0.98, 0.95,
        f"{n_singletons}/{len(counts)} architectures are singletons (n=1)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {out_path}")
    print(f"  Singleton architectures: {n_singletons}/{len(counts)}")


def draw_accumulation_curve(proteins: dict, out_path: str, n_permutations: int = 20):
    """Pan-genome-style accumulation curve: as proteins are sampled in random
    order, how many cumulative UNIQUE architectures have been seen?

    If the curve has plateaued by the full sample size, architecture space
    is well-sampled. If it's still rising steeply, architecture space is
    undersampled — direct visual evidence relevant to H9.

    Averaged over multiple random orderings (n_permutations) to smooth
    out order-dependence, similar to standard pan-genome curve methodology.
    """
    import random

    pids = list(proteins.keys())
    n = len(pids)
    all_curves = []

    for _ in range(n_permutations):
        order = pids.copy()
        random.shuffle(order)
        seen_architectures = set()
        curve = []
        for pid in order:
            arch = tuple(d[0] for d in proteins[pid]["domains"])
            seen_architectures.add(arch)
            curve.append(len(seen_architectures))
        all_curves.append(curve)

    # average across permutations
    avg_curve = [sum(c[i] for c in all_curves) / n_permutations for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in all_curves:
        ax.plot(range(1, n + 1), c, color="#AAAAAA", alpha=0.3, linewidth=0.8)
    ax.plot(range(1, n + 1), avg_curve, color="#EE6677", linewidth=2, label="Mean over permutations")

    ax.set_xlabel("Number of proteins sampled")
    ax.set_ylabel("Cumulative unique architectures observed")
    ax.set_title("Domain architecture accumulation curve")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {out_path}")
    print(f"  Final unique architectures at n={n}: {avg_curve[-1]:.0f}")
    # rough slope check over last 10% of samples as a plateau heuristic
    tail_start = int(n * 0.9)
    tail_slope = (avg_curve[-1] - avg_curve[tail_start]) / max(1, n - tail_start)
    print(f"  Slope over final 10% of samples: {tail_slope:.4f} new architectures/protein")
    if tail_slope > 0.01:
        print("  -> Curve has NOT plateaued: architecture space appears undersampled")
    else:
        print("  -> Curve appears close to plateau at this sample size")



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize unique protein domain architectures")
    parser.add_argument("--domains", required=True, help="Resolved domain TSV file")
    parser.add_argument("--out", required=True, help="Output image path for schematic diagram (.png/.pdf/.svg)")
    parser.add_argument("--top", type=int, default=30, help="Show top N architectures by frequency in schematic (0 = all)")
    parser.add_argument("--sort-by", choices=["frequency", "similarity"], default="frequency",
                         help="Row order: 'frequency' (most common first) or 'similarity' "
                              "(hierarchical clustering on shared domains, so e.g. all CBM91-containing "
                              "architectures are grouped together) (default: frequency)")
    parser.add_argument("--subset", nargs="+", metavar="DOMAIN",
                         help="Only include proteins containing at least one of these domains "
                              "(e.g. --subset GH43_1 GH43_2)")
    parser.add_argument("--freq-out", help="Optional: output path for frequency-rank bar chart (full distribution)")
    parser.add_argument("--accumulation-out", help="Optional: output path for architecture accumulation curve")
    parser.add_argument("--permutations", type=int, default=20, help="Number of random orderings for accumulation curve (default: 20)")
    args = parser.parse_args()

    proteins = parse_domains(args.domains)

    if args.subset:
        required = set(args.subset)
        proteins = {
            pid: info for pid, info in proteins.items()
            if required & {d[0] for d in info["domains"]}
        }
        if not proteins:
            sys.exit(f"ERROR: no proteins contain any of the requested domains: {args.subset}")
        print(f"  Subset: kept {len(proteins)} proteins containing {' / '.join(sorted(required))}")

    architectures, representative = collapse_architectures(proteins)
    draw_architectures(
        proteins, architectures, representative, args.out,
        top_n=args.top if args.top > 0 else None,
        sort_by=args.sort_by,
    )

    if args.freq_out:
        draw_frequency_rank(architectures, args.freq_out)

    if args.accumulation_out:
        draw_accumulation_curve(proteins, args.accumulation_out, n_permutations=args.permutations)


if __name__ == "__main__":
    main()