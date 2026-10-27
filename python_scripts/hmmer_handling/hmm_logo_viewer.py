#!/usr/bin/env python3
"""
Draw a sequence logo for every profile in a HMMER3 profile HMM file.

Per match state, the stack height is its information content in bits,
log2(alphabet size) - H(match emissions), and each letter's height is its
emission probability times that (as in WebLogo). The logo is drawn with
logomaker, coloured by its "chemistry" scheme for protein and "classic" for
nucleic acid. Insert states are not shown. The x axis is the match-state
number.

Needs logomaker (pip install logomaker, or conda install -c conda-forge
logomaker) and pyhmmer.

Writes <out-dir>/<profile name>.png (or the --format you give) per profile;
a long profile is wrapped into rows of --per-row positions.

Usage:
    python hmm_logo_viewer.py profiles.hmm
    python hmm_logo_viewer.py profiles.hmm -o logos --per-row 80 --format pdf
    python hmm_logo_viewer.py profiles.hmm --names GH43_1 GH43_10
"""
import argparse
import math
import re
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhmmer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hmm", help="HMMER3 profile HMM file (one or more profiles)")
    parser.add_argument("-o", "--out-dir", default="hmm_logos", help="Output folder (default: hmm_logos)")
    parser.add_argument("--names", nargs="+", default=None, help="Only these profiles (by NAME)")
    parser.add_argument("--per-row", type=int, default=60, help="Positions per row of the logo (default: 60)")
    parser.add_argument("--format", default="png", help="Image format: png, pdf, svg (default: png)")
    return parser.parse_args()


def letter_heights(hmm) -> pd.DataFrame:
    """Letter heights in bits: one row per match state (1..M), one column per letter."""
    k = hmm.alphabet.K
    letters = hmm.alphabet.symbols[:k]
    probs = np.asarray(hmm.match_emissions, dtype=float)[1:, :k]   # row 0 is the begin state
    probs = probs / probs.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(probs > 0, probs * np.log2(probs), 0), axis=1)
    information = math.log2(k) - entropy
    return pd.DataFrame(probs * information[:, None], columns=list(letters),
                        index=pd.RangeIndex(1, len(probs) + 1, name="pos"))


def plot_logo(heights: pd.DataFrame, name: str, path: Path, per_row: int) -> None:
    scheme = "classic" if heights.shape[1] <= 5 else "chemistry"
    n_pos = len(heights)
    n_rows = math.ceil(n_pos / per_row)
    y_max = math.log2(heights.shape[1])
    fig, axes = plt.subplots(n_rows, 1, figsize=(0.18 * per_row + 1, 1.8 * n_rows), squeeze=False)
    for row, ax in enumerate(axes[:, 0]):
        first = row * per_row
        logomaker.Logo(heights.loc[first + 1:first + per_row], ax=ax, color_scheme=scheme)
        ax.set_xlim(first + 0.5, first + per_row + 0.5)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("bits")
        ax.set_xticks([t for t in range(first + 1, first + per_row + 1) if t % 10 == 0 or t == first + 1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title(f"{name} ({n_pos} match states)", loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("match state")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with pyhmmer.plan7.HMMFile(args.hmm) as hmm_file:
        for hmm in hmm_file:
            name = hmm.name if isinstance(hmm.name, str) else hmm.name.decode()
            if args.names and name not in args.names:
                continue
            heights = letter_heights(hmm)
            path = out_dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', name)}.{args.format}"
            plot_logo(heights, name, path, args.per_row)
            print(f"{name}: {len(heights)} match states -> {path}")
            written += 1
    if args.names and written < len(args.names):
        print(f"  warning: only {written} of the {len(args.names)} --names found in {args.hmm}")
    print(f"Written {written} logos to {out_dir}")


if __name__ == "__main__":
    main()
