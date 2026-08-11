#!/usr/bin/env python3
"""
Extract pHMMs with at least one characterized member from a dbCAN sub-cluster HMM file.

A model is considered to have characterized members when its NAME field contains at least
one EC number (pattern: digits.digits.digits.digits, e.g. '3.2.1.55' or '3.2.1.55:7').

Optionally restrict to a NAME prefix pattern (e.g. 'GH43_e') to pull only a single family.

Usage:
    python extract_characterized_hmms.py \\
      --hmm-db  data/dbcan/phmm/dbCAN_sub.hmm \\
      --pattern GH43_e \\
      -o        data/dbcan/phmm/GH43_char_sub.hmm
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

EC_PATTERN = re.compile(r"\d+\.\d+\.\d+\.\d+")


def extract_hmms(hmm_db: Path, outfile: Path, name_pattern: str = "") -> int:
    """Write models that contain an EC number in their NAME field.

    If name_pattern is given, the NAME must also start with that string.
    Returns the number of models written.
    """
    n_written = 0
    buffer = []
    current_keep = False

    with open(hmm_db) as fin, open(outfile, "w") as fout:
        for line in fin:
            if line.startswith("NAME"):
                name = line.split(None, 1)[1].strip()
                matches_pattern = name.startswith(name_pattern) if name_pattern else True
                has_ec = bool(EC_PATTERN.search(name))
                current_keep = matches_pattern and has_ec

            buffer.append(line)

            if line.strip() == "//":
                if current_keep:
                    fout.writelines(buffer)
                    n_written += 1
                buffer = []
                current_keep = False

    return n_written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hmm-db", type=Path, required=True,
                        help="dbCAN sub-cluster HMM database to filter")
    parser.add_argument("--pattern", default="",
                        help="Optional NAME prefix to restrict family (e.g. 'GH43_e')")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output HMM file")
    parser.add_argument("--no-press", action="store_true",
                        help="Skip hmmpress after extraction")
    args = parser.parse_args()

    print("[1/2] Extracting HMMs with EC annotations")
    if args.pattern:
        print(f"  NAME filter: starts with '{args.pattern}'")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = extract_hmms(args.hmm_db, args.output, args.pattern)
    print(f"  {n} models written to {args.output}")

    if n == 0:
        sys.exit("No models matched — check --hmm-db and --pattern.")

    if args.no_press:
        print("Done (skipped hmmpress).")
        return

    print("[2/2] Running hmmpress")
    subprocess.run(["hmmpress", "-f", str(args.output)], check=True)
    print("Done.")


if __name__ == "__main__":
    main()
