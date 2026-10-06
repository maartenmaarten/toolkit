#!/usr/bin/env python3
"""Quantify how far each CAZy subfamily is confounded with EC activity.

Usage: python subf_ec_purity.py GH43_characterized.tsv [-o out_prefix] [--no-plots]

Reports, per subfamily: n sequences, n distinct EC labels, the dominant label
and its share, Shannon entropy of the EC distribution, and the full breakdown.
Multi-EC entries (e.g. "['3.2.1.37', '3.2.1.55']") are treated as their own
composite label by default; --split-multi counts them once per component EC.

Why this matters: a subfamily containing only one activity gives no
within-subfamily contrast, so any position that discriminates that activity is
indistinguishable from a subfamily (clade) marker.
"""
import argparse, math, re
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("tsv")
p.add_argument("-o", "--out-prefix", default="subf_ec")
p.add_argument("--subf-col", default="Subf")
p.add_argument("--ec-col", default="EC")
p.add_argument("--split-multi", action="store_true",
               help="count a multi-EC entry once per component EC")
p.add_argument("--no-plots", action="store_true")
p.add_argument("--min-n", type=int, default=3,
               help="subfamilies with fewer sequences are pooled as 'other' in plots")
a = p.parse_args()

df = pd.read_csv(a.tsv, sep="\t", dtype=str)
for c in (a.subf_col, a.ec_col):
    if c not in df.columns:
        raise SystemExit(f"ERROR: column '{c}' not in {a.tsv}.\n"
                         f"Available: {list(df.columns)}")

df[a.subf_col] = df[a.subf_col].fillna("unassigned").str.strip().replace("", "unassigned")

def ec_list(s):
    ecs = sorted(set(re.findall(r"\d+\.\d+\.\d+\.[\d-]+", str(s))))
    return ecs or ["unknown"]

df["_ecs"] = df[a.ec_col].apply(ec_list)
df["_label"] = df["_ecs"].apply(lambda l: "+".join(l))

# one row per (sequence, EC) if splitting, else per sequence
long = df.explode("_ecs") if a.split_multi else df.assign(_ecs=df["_label"])
key = "_ecs"

rows = []
for sf, g in long.groupby(df.loc[long.index, a.subf_col]):
    cnt = Counter(g[key])
    n = sum(cnt.values())
    dom, dom_n = cnt.most_common(1)[0]
    ent = -sum((v/n) * math.log2(v/n) for v in cnt.values())
    rows.append({
        "subfamily": sf,
        "n": n,
        "n_distinct_EC": len(cnt),
        "dominant_EC": dom,
        "dominant_frac": round(dom_n / n, 3),
        "entropy_bits": round(ent, 3),
        "pure": len(cnt) == 1,
        "breakdown": "; ".join(f"{k}:{v}" for k, v in cnt.most_common()),
    })

res = (pd.DataFrame(rows)
       .sort_values(["n_distinct_EC", "n"], ascending=[False, False])
       .reset_index(drop=True))

def subfamily_sort_key(v):
    s = str(v)
    return (not s.isdigit(), int(s) if s.isdigit() else 0, s)

res["subfamily"] = res["subfamily"].astype(str)
res = res.sort_values("subfamily", key=lambda s: s.map(subfamily_sort_key)).reset_index(drop=True)

res.to_csv(f"{a.out_prefix}_purity.tsv", sep="\t", index=False)

# subfamily x EC contingency table
tab = pd.crosstab(long[a.subf_col] if a.subf_col in long else df.loc[long.index, a.subf_col],
                  long[key])
tab.index = tab.index.astype(str)
tab = tab.reindex(sorted(tab.index, key=subfamily_sort_key)).fillna(0)
tab.to_csv(f"{a.out_prefix}_matrix.tsv", sep="\t")

# --- summary --------------------------------------------------------------
tot = len(df)
pure_sf = res[res["pure"]]
print(f"{tot} sequences | {len(res)} subfamilies "
      f"| multi-EC entries kept as composite labels: {not a.split_multi}\n")
print(f"single-activity subfamilies: {len(pure_sf)}/{len(res)} "
      f"covering {pure_sf['n'].sum()}/{tot} sequences ({pure_sf['n'].sum()/tot:.0%})")
print(f"subfamilies with >=2 distinct EC labels: {(~res['pure']).sum()} "
      f"covering {res.loc[~res['pure'],'n'].sum()} sequences\n")
with pd.option_context("display.width", 200, "display.max_colwidth", 70):
    print(res.to_string(index=False))
print(f"\nwrote {a.out_prefix}_purity.tsv and {a.out_prefix}_matrix.tsv")

# ---------------------------------------------------------------- plots ---
if not a.no_plots:
    # tab is already ordered by subfamily integer count (see reindex above)

    cmap = plt.get_cmap("tab20")
    colors = {ec: cmap(i % 20) for i, ec in enumerate(tab.columns)}

    fig, ax = plt.subplots(1, 2, figsize=(15, max(5, 0.36 * len(tab) + 2.2)),
                           gridspec_kw={"width_ratios": [2.2, 1]})

    # --- A: stacked composition -------------------------------------------
    left = np.zeros(len(tab))
    y = np.arange(len(tab))
    for ec in tab.columns:
        v = tab[ec].values
        ax[0].barh(y, v, left=left, color=colors[ec], edgecolor="white",
                   linewidth=0.5, label=ec)
        left += v
    ax[0].set_yticks(y); ax[0].set_yticklabels(tab.index)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("characterised sequences")
    ax[0].set_ylabel("CAZy subfamily")
    ax[0].set_title(f"EC composition per subfamily (n \u2265 {a.min_n})", loc="left")
    ax[0].legend(fontsize=7, ncol=4, frameon=False,
                 bbox_to_anchor=(0.5, -0.13), loc="upper center")
    for sp in ("top", "right"):
        ax[0].spines[sp].set_visible(False)

    # --- B: purity vs size -------------------------------------------------
    sub = res[res["subfamily"].astype(str).isin(tab.index)]
    sc = ax[1].scatter(sub["n"], sub["dominant_frac"], s=28 + 12 * sub["n_distinct_EC"],
                       c=sub["entropy_bits"], cmap="viridis", edgecolor="k", linewidth=0.4)
    for _, r in sub.iterrows():
        ax[1].annotate(str(r["subfamily"]), (r["n"], r["dominant_frac"]),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].axhline(1.0, ls=":", c="grey", lw=1)
    ax[1].set_xlabel("subfamily size (n)")
    ax[1].set_ylabel("dominant EC fraction")
    ax[1].set_ylim(0.3, 1.06)
    ax[1].set_title("purity vs size", loc="left")
    plt.colorbar(sc, ax=ax[1], label="EC entropy (bits)", shrink=0.8)
    for sp in ("top", "right"):
        ax[1].spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{a.out_prefix}_composition.png", dpi=200)
    fig.savefig(f"{a.out_prefix}_composition.svg")

    # --- C: heatmap --------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(1.1 + 0.55 * tab.shape[1], 1.5 + 0.30 * tab.shape[0]))
    m = ax2.imshow(tab.values, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(tab.shape[1])); ax2.set_xticklabels(tab.columns, rotation=90, fontsize=7)
    ax2.set_yticks(range(tab.shape[0])); ax2.set_yticklabels(tab.index, fontsize=7)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = int(tab.values[i, j])
            if v:
                ax2.text(j, i, v, ha="center", va="center", fontsize=6,
                         color="white" if v > tab.values.max() * 0.55 else "black")
    ax2.set_xlabel("EC label"); ax2.set_ylabel("subfamily")
    plt.colorbar(m, ax=ax2, label="sequences", shrink=0.7)
    fig2.tight_layout()
    fig2.savefig(f"{a.out_prefix}_heatmap.png", dpi=200)
    fig2.savefig(f"{a.out_prefix}_heatmap.svg")
    print(f"wrote {a.out_prefix}_composition.[png|svg] and {a.out_prefix}_heatmap.[png|svg]")