#!/usr/bin/env python3
# Usage: python genbank_id_coverage.py <query_ids.txt> <reference_ids.txt>
#        Checks what fraction of query IDs are present in the reference file.

import sys


def read_ids(path):
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip() and line.strip() != "GenBank"}


def coverage(query_ids, reference_ids):
    found = query_ids & reference_ids
    missing = query_ids - reference_ids
    pct = len(found) / len(query_ids) * 100 if query_ids else 0.0
    return found, missing, pct


def main():
    if len(sys.argv) != 3:
        print("Usage: python genbank_id_coverage.py <query_ids.txt> <reference_ids.txt>")
        sys.exit(1)

    query_ids = read_ids(sys.argv[1])
    reference_ids = read_ids(sys.argv[2])

    if not query_ids:
        print("Query file is empty.", file=sys.stderr)
        sys.exit(1)

    found, missing, pct = coverage(query_ids, reference_ids)

    print(f"Query IDs:     {len(query_ids)}")
    print(f"Found:         {len(found)}")
    print(f"Missing:       {len(missing)}")
    print(f"Coverage:      {pct:.2f}%")

    if missing:
        print("\nMissing IDs:")
        for id_ in sorted(missing):
            print(f"  {id_}")


if __name__ == "__main__":
    main()
