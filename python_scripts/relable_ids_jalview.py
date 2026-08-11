#!/usr/bin/env python3
"""Rewrite an MSA with names prefixed by a chosen TSV column, sorted by that label.

Usage:
  python relabel_msa.py ALN TSV OUT --target-col EC [Subf ...] [--match-col GenBank]
                                    [--no-sort] [--sep '|'] [--label-sep '_']
                                    [--keep-unlabelled]

Multiple --target-col values are concatenated into one label, in the order given,
and become the sort key in that order (first column is primary).

ALN  Stockholm (.sto/.stk/.stockholm) or aligned FASTA; OUT is aligned FASTA.
Column names must match the TSV header exactly (no alias guessing).
A trailing /start-end suffix on alignment names is stripped before matching.
"""
import sys, re, os, argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("aln"); p.add_argument("tsv"); p.add_argument("out")
p.add_argument("--target-col", required=True, nargs="+",
               help="one or more TSV columns used as the label (in order given)")
p.add_argument("--match-col", default="GenBank", help="TSV column matched to alignment names")
p.add_argument("--sep", default="|", help="separator between label and original name")
p.add_argument("--label-sep", default="_", help="separator between multiple target columns")
p.add_argument("--no-sort", action="store_true", help="keep original alignment order")
p.add_argument("--keep-unlabelled", action="store_true",
               help="write unmatched sequences with a NOLABEL prefix (default: drop)")
a = p.parse_args()

# --- alignment -------------------------------------------------------------
def read_stockholm(path):
    seqs = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        f = line.split(None, 1)
        if len(f) == 2:
            seqs[f[0]] = seqs.get(f[0], "") + f[1]
    return seqs

def read_fasta(path):
    seqs, sid = {}, None
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            sid = line[1:].split()[0]; seqs[sid] = ""
        elif sid is not None:
            seqs[sid] += line
    return seqs

ext = os.path.splitext(a.aln)[1].lower()
seqs = read_stockholm(a.aln) if ext in (".sto", ".stk", ".stockholm") else read_fasta(a.aln)
if not seqs:
    sys.exit(f"ERROR: no sequences parsed from {a.aln}")

# --- labels ----------------------------------------------------------------
df = pd.read_csv(a.tsv, sep="\t", dtype=str)
for col in list(a.target_col) + [a.match_col]:
    if col not in df.columns:
        sys.exit(f"ERROR: column '{col}' not in {a.tsv}.\nAvailable: {list(df.columns)}")

df = df.dropna(subset=[a.match_col])
key = df[a.match_col].str.strip()
if key.duplicated().any():
    dups = key[key.duplicated()].unique()[:5]
    print(f"WARNING: {a.match_col} has duplicates, last wins: {list(dups)}", file=sys.stderr)

parts = [df[c].fillna("NA").astype(str).str.strip().replace("", "NA")
         for c in a.target_col]
labmap = dict(zip(key, zip(*parts)))          # value = tuple, one per target col

bare = lambda n: re.sub(r"/\d+-\d+$", "", n)
rows, missing = [], []
for name, seq in seqs.items():
    if bare(name) in labmap:
        rows.append((labmap[bare(name)], name, seq))
    else:
        missing.append(name)
        if a.keep_unlabelled:
            rows.append((("NOLABEL",) * len(a.target_col), name, seq))

if not a.no_sort:
    rows.sort(key=lambda r: (r[0], r[1]))

# --- write -----------------------------------------------------------------
safe = lambda s: re.sub(r"\s+", "_", s)
with open(a.out, "w") as fh:
    for label, name, seq in rows:
        fh.write(f">{a.label_sep.join(safe(x) for x in label)}{a.sep}{name}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i+60] + "\n")

print(f"alignment: {len(seqs)} seqs | labelled: {len(seqs)-len(missing)} | "
      f"unmatched: {len(missing)} ({'kept' if a.keep_unlabelled else 'dropped'})")
if missing:
    print("  unmatched:", missing[:10], "..." if len(missing) > 10 else "")
print(f"wrote {len(rows)} seqs to {a.out}")