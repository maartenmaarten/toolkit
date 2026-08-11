#!/usr/bin/env python3
"""Build a Jalview sequence-features file for a GH43 MSA, coloured by EC.

Usage: python make_jalview_features.py aln.sto characterized_ALL.tsv out.features
Accepts Stockholm (.sto/.stk/.stockholm) or aligned FASTA (.afa/.fa/.fasta/.aln).
IDs are matched on the accession; a trailing /start-end coordinate suffix
(as added by hmmalign/esl-reformat) is stripped before matching.
"""
import sys, re, os
import pandas as pd, seaborn as sns
from matplotlib.colors import to_hex

aln, tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
ext = os.path.splitext(aln)[1].lower()

GAPS = str.maketrans("", "", "-.~")

def read_stockholm(path):
    """Return {seq_name: ungapped_length}; handles interleaved blocks."""
    L = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name, seq = parts
        # lowercase = insertion relative to HMM, still a real residue: keep it
        L[name] = L.get(name, 0) + len(seq.translate(GAPS))
    return L

def read_fasta(path):
    L, sid = {}, None
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            sid = line[1:].split()[0]
            L[sid] = 0
        elif sid:
            L[sid] += len(line.translate(GAPS))
    return L

lengths = read_stockholm(aln) if ext in (".sto", ".stk", ".stockholm") else read_fasta(aln)

# map alignment name -> bare accession (strip /start-end if present)
def bare(name):
    return re.sub(r"/\d+-\d+$", "", name)

# --- EC labels -------------------------------------------------------------
df = pd.read_csv(tsv, sep="\t", dtype=str)
#df = df[df["Family"].str.contains("GH43", na=False)].dropna(subset=["GenBank"])

def ec_label(s):
    ecs = sorted(set(re.findall(r"\d+\.\d+\.\d+\.[\d-]+", str(s))))
    return "+".join(ecs) if ecs else "unknown"   # no brackets/quotes/spaces

df["ec"] = df["EC"].apply(ec_label)
ecmap = dict(zip(df["GenBank"].str.strip(), df["ec"]))

matched = {n: ecmap[bare(n)] for n in lengths if bare(n) in ecmap}
missing = [n for n in lengths if bare(n) not in ecmap]
print(f"alignment: {len(lengths)} seqs | annotated: {len(matched)} | unlabelled: {len(missing)}")
if missing:
    print("  no EC for:", missing[:10], "..." if len(missing) > 10 else "")

# --- write -----------------------------------------------------------------
types = sorted(set(matched.values()))
pal = sns.color_palette("husl", len(types))
cols = {t: to_hex(c)[1:] for t, c in zip(types, pal)}

with open(out, "w") as fh:
    for t in types:
        fh.write(f"{t}\t{cols[t]}\n")
    fh.write("\n")
    for name, ec in matched.items():
        fh.write(f"{ec}\t{name}\t-1\t1\t{lengths[name]}\t{ec}\n")
print(f"wrote {out}: {len(matched)} features, {len(types)} EC types")